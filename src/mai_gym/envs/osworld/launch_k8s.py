from __future__ import annotations

import hashlib
import json
import logging
import shlex
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from mai_gym.infra.launcher.k8s_base import K8sSandboxLauncherBase
from mai_gym.models.config import K8sLaunchConfig

logger = logging.getLogger(__name__)


def _sudo_password_pipe(launch: K8sLaunchConfig) -> str:
    return f"printf '%s\\n' {shlex.quote(launch.vm_password)} | sudo -S -p ''"


def _proxy_setup_command(launch: K8sLaunchConfig) -> str | None:
    proxy = (launch.http_proxy or "").strip()
    if not proxy:
        return None
    no_proxy = launch.no_proxy.strip()
    exports = {
        "http_proxy": proxy,
        "https_proxy": proxy,
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
    }
    if no_proxy:
        exports.update({"no_proxy": no_proxy, "NO_PROXY": no_proxy})
    env_lines = "\n".join(f"{key}={shlex.quote(value)}" for key, value in sorted(exports.items()))
    profile_lines = "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted(exports.items())
    )
    apt_conf = "\n".join(
        [
            f'Acquire::http::Proxy "{proxy}";',
            f'Acquire::https::Proxy "{proxy}";',
        ]
    )
    sudo = _sudo_password_pipe(launch)
    return (
        "tmp_env=$(mktemp); tmp_profile=$(mktemp); tmp_apt_conf=$(mktemp); "
        "grep -v -E '^(http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY|"
        'no_proxy|NO_PROXY)=\' /etc/environment 2>/dev/null > "$tmp_env" || true; '
        f"printf '%s\\n' {shlex.quote(env_lines)} >> \"$tmp_env\"; "
        f"printf '%s\\n' {shlex.quote(profile_lines)} > \"$tmp_profile\"; "
        f"printf '%s\\n' {shlex.quote(apt_conf)} > \"$tmp_apt_conf\"; "
        f'{sudo} cp "$tmp_env" /etc/environment; '
        f'{sudo} cp "$tmp_profile" /etc/profile.d/mai-gym-proxy.sh; '
        f"{sudo} chmod 0644 /etc/profile.d/mai-gym-proxy.sh; "
        f"{sudo} mkdir -p /etc/apt/apt.conf.d; "
        f'{sudo} cp "$tmp_apt_conf" /etc/apt/apt.conf.d/99-mai-gym-proxy; '
        'rm -f "$tmp_env" "$tmp_profile" "$tmp_apt_conf"'
    )


def _proxy_endpoint(proxy: str | None) -> str | None:

    if not proxy:
        return None
    parsed = urlsplit(proxy)
    if parsed.hostname is None:
        return "configured"
    try:
        parsed_port = parsed.port
    except ValueError:
        return f"{parsed.scheme or 'http'}://{parsed.hostname}:configured"
    port = f":{parsed_port}" if parsed_port is not None else ""
    return f"{parsed.scheme or 'http'}://{parsed.hostname}{port}"


def pod_bootstrap_contract(launch: K8sLaunchConfig) -> dict[str, Any]:

    fingerprint_input = {
        "enabled": launch.pod_bootstrap_enabled,
        "pip_index_url": launch.pip_index_url,
        "pip_trusted_host": launch.pip_trusted_host,
        "apt_mirror_url": launch.apt_mirror_url,
        "proxy_endpoint": _proxy_endpoint(launch.http_proxy),
        "no_proxy": launch.no_proxy,
        "pod_setup_commands": launch.pod_setup_commands,
        "pod_setup_timeout": launch.pod_setup_timeout,
        "pod_setup_max_attempts": launch.pod_setup_max_attempts,
        "pod_setup_retry_backoff": launch.pod_setup_retry_backoff,
    }
    encoded = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":")).encode()
    return {
        "enabled": launch.pod_bootstrap_enabled,
        "fingerprint": hashlib.sha256(encoded).hexdigest()[:16],
        "pip_mirror": bool(launch.pip_index_url),
        "apt_mirror": bool(launch.apt_mirror_url),
        "proxy": bool(launch.http_proxy),
        "extra_commands": len(launch.pod_setup_commands),
    }


class OSWorldK8sProfile(K8sSandboxLauncherBase):
    _ready_path = "/screenshot"
    _ready_poll_interval = 3.0
    _ready_default_timeout = 300.0

    def _ready_predicate(self, response: httpx.Response) -> bool:
        return response.status_code == 200 and bool(response.content)

    @staticmethod
    def _apt_mirror_setup_command(launch: K8sLaunchConfig) -> str | None:
        mirror = (launch.apt_mirror_url or "").strip().rstrip("/")
        if not mirror:
            return None
        replacement = f"{mirror}/"
        sed_expr = (
            r"s#https?://archive\.ubuntu\.com/ubuntu/?#" + replacement + r"#g; "
            r"s#https?://security\.ubuntu\.com/ubuntu/?#" + replacement + r"#g; "
            r"s#https?://[A-Za-z0-9.-]+\.archive\.ubuntu\.com/ubuntu/?#" + replacement + r"#g"
        )
        sudo = _sudo_password_pipe(launch)
        return (
            "if [ -f /etc/apt/sources.list ]; then "
            f"[ -f /etc/apt/sources.list.mai-gym.bak ] || {sudo} "
            "cp /etc/apt/sources.list /etc/apt/sources.list.mai-gym.bak; "
            f"{sudo} sed -i -E {shlex.quote(sed_expr)} /etc/apt/sources.list; "
            "fi"
        )

    @staticmethod
    def _pip_mirror_setup_command(launch: K8sLaunchConfig) -> str | None:
        index_url = (launch.pip_index_url or "").strip()
        if not index_url:
            return None
        conf = f"[global]\nindex-url = {index_url}\n"
        trusted_host = (launch.pip_trusted_host or "").strip()
        if trusted_host:
            conf += f"trusted-host = {trusted_host}\n"
        sudo = _sudo_password_pipe(launch)
        return (
            f"tmp_pip_conf=$(mktemp); printf '%s' {shlex.quote(conf)} "
            '> "$tmp_pip_conf"; '
            "for d in /home/user/.config/pip /home/user/.pip; "
            'do mkdir -p "$d" 2>/dev/null; done; '
            f"{sudo} mkdir -p /root/.config/pip /root/.pip; "
            f'{sudo} cp "$tmp_pip_conf" /etc/pip.conf; '
            f'{sudo} cp "$tmp_pip_conf" /root/.config/pip/pip.conf; '
            f'{sudo} cp "$tmp_pip_conf" /root/.pip/pip.conf; '
            f"{sudo} chmod 0644 /etc/pip.conf "
            "/root/.config/pip/pip.conf /root/.pip/pip.conf; "
            'cp "$tmp_pip_conf" /home/user/.config/pip/pip.conf; '
            'cp "$tmp_pip_conf" /home/user/.pip/pip.conf; '
            "chmod 0644 /home/user/.config/pip/pip.conf "
            "/home/user/.pip/pip.conf; "
            'rm -f "$tmp_pip_conf"; '
            "chown -R user:user /home/user/.config /home/user/.pip "
            "2>/dev/null || true"
        )

    @classmethod
    def _core_pod_env_setup_parts(cls, launch: K8sLaunchConfig) -> list[str]:
        parts: list[str] = []
        for command in (
            _proxy_setup_command(launch),
            cls._apt_mirror_setup_command(launch),
            cls._pip_mirror_setup_command(launch),
        ):
            if command:
                parts.append(command)
        return parts

    @staticmethod
    def _bootstrap_contract(launch: K8sLaunchConfig) -> dict[str, Any]:
        return pod_bootstrap_contract(launch)

    def _prepare_backend(
        self,
        *,
        url: str,
        namespace: str,
        sandbox_name: str,
        launch: K8sLaunchConfig,
    ) -> dict[str, Any]:
        if not launch.pod_bootstrap_enabled:
            return {}
        contract = self._bootstrap_contract(launch)
        logger.info(
            "[OSWorldK8sProfile] pod bootstrap start at %s "
            "(fingerprint=%s pip=%s apt=%s proxy=%s extra_commands=%d)",
            url,
            contract["fingerprint"],
            contract["pip_mirror"],
            contract["apt_mirror"],
            contract["proxy"],
            contract["extra_commands"],
        )
        core_ok = self._run_pod_script(
            url,
            self._core_pod_env_setup_parts(launch),
            label="core pod environment",
            launch=launch,
        )
        extra_ok = self._run_pod_script(
            url,
            list(launch.pod_setup_commands),
            label="extra pod setup",
            launch=launch,
        )
        if core_ok and extra_ok:
            logger.info(
                "[OSWorldK8sProfile] pod bootstrap complete at %s (fingerprint=%s)",
                url,
                contract["fingerprint"],
            )
        else:
            logger.warning(
                "[OSWorldK8sProfile] pod bootstrap incomplete at %s "
                "(fingerprint=%s core_ok=%s extra_ok=%s)",
                url,
                contract["fingerprint"],
                core_ok,
                extra_ok,
            )
        return {}

    @staticmethod
    def _run_pod_script(
        url: str,
        commands: list[str],
        *,
        label: str,
        launch: K8sLaunchConfig,
    ) -> bool:

        if not commands:
            return True
        script = "set -e; " + " ; ".join(commands)
        last_result = "not attempted"
        for attempt in range(1, launch.pod_setup_max_attempts + 1):
            try:
                with httpx.Client(timeout=launch.pod_setup_timeout) as client:
                    response = client.post(
                        f"{url}/execute",
                        json={"command": script, "shell": True},
                    )
                try:
                    body = response.json()
                except Exception:
                    body = {}
                rc = body.get("returncode") if isinstance(body, dict) else None
                if response.status_code == 200 and rc in (None, 0, "0"):
                    logger.info(
                        "[OSWorldK8sProfile] %s succeeded at %s on attempt %d",
                        label,
                        url,
                        attempt,
                    )
                    return True
                last_result = f"HTTP {response.status_code} rc={rc!r}"
            except Exception as exc:
                last_result = f"transport {type(exc).__name__}"
            if attempt < launch.pod_setup_max_attempts:
                delay = launch.pod_setup_retry_backoff * attempt
                logger.warning(
                    "[OSWorldK8sProfile] %s retry %d/%d at %s in %.1fs (%s)",
                    label,
                    attempt,
                    launch.pod_setup_max_attempts,
                    url,
                    delay,
                    last_result,
                )
                if delay:
                    time.sleep(delay)
        logger.warning(
            "[OSWorldK8sProfile] %s failed at %s after %d attempt(s) (%s)",
            label,
            url,
            launch.pod_setup_max_attempts,
            last_result,
        )
        return False

    def _describe_backend(
        self,
        *,
        url: str,
        pod_ip: str,
        sandbox_name: str,
        namespace: str,
        launch: K8sLaunchConfig,
    ) -> tuple[str, dict[str, Any]]:

        metadata: dict[str, Any] = {
            "pod_ip": pod_ip,
            "pod_name": sandbox_name,
            "namespace": namespace,
            "chromium_port": int(launch.chromium_port),
            "vlc_port": int(launch.vlc_port),
            "vnc_port": int(launch.vnc_port),
            "pod_bootstrap": self._bootstrap_contract(launch),
        }
        return sandbox_name, metadata

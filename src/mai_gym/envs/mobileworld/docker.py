from __future__ import annotations

import json
import logging
import socket
import subprocess
import sys
import time
from collections.abc import Iterable
from typing import Any

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "ghcr.io/tongyi-mai/mobile_world:latest"
DEFAULT_NAME_PREFIX = "mobile_world_env"


class ContainerInfo(BaseModel):
    name: str
    status: str | None = None
    running: bool = False
    started_at: str | None = None
    image: str | None = None
    backend_port: int | None = None
    viewer_port: int | None = None
    vnc_port: int | None = None
    adb_port: int | None = None


class ContainerConfig(BaseModel):
    name: str
    backend_port: int
    viewer_port: int
    vnc_port: int
    adb_port: int = 5556
    image: str = DEFAULT_IMAGE
    dev_mode: bool = False
    enable_vnc: bool = False
    env_file_path: Any | None = None
    dev_src_path: Any | None = None
    http_proxy: str | None = None


class LaunchResult(BaseModel):
    name: str
    backend_port: int
    viewer_port: int
    adb_port: int
    vnc_port: int
    success: bool = False
    ready: bool = False
    error_message: str | None = None


def run_command(
    cmd: list[str],
    capture: bool = True,
    allowed_exit_codes: set[int] | None = None,
) -> subprocess.CompletedProcess:

    try:
        if capture:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(cmd, check=True)
        return result
    except subprocess.CalledProcessError as e:
        if allowed_exit_codes and e.returncode in allowed_exit_codes:
            logger.debug(
                "Command returned allowed exit code %s: %s",
                e.returncode,
                " ".join(cmd),
            )

            return subprocess.CompletedProcess(
                args=e.cmd,
                returncode=e.returncode,
                stdout=e.stdout,
                stderr=e.stderr,
            )

        stderr_text = e.stderr or ""
        logger.error("Command failed: %s", " ".join(cmd))
        logger.error("Exit code: %s", e.returncode)
        if stderr_text:
            logger.error("Error output: %s", stderr_text)
            if "permission denied" in stderr_text.lower() and "docker" in stderr_text.lower():
                _log_docker_permission_help()
        sys.exit(1)


def _log_docker_permission_help() -> None:
    logger.error("%s", "=" * 80)
    logger.error("Docker Permission Error Detected")
    logger.error("%s", "=" * 80)
    logger.error("Your user doesn't have permission to access the Docker daemon.")
    logger.error("To fix this issue, try one of the following:")
    logger.error("  1. Add your user to the docker group:")
    logger.error("     $ sudo usermod -aG docker $USER")
    logger.error("     $ newgrp docker")
    logger.error("  2. Run the command with sudo:")
    logger.error("     $ sudo mobile-world env list")
    logger.error("  3. Check Docker daemon is running:")
    logger.error("     $ sudo systemctl status docker")


def docker_ps(include_all: bool = False) -> list[dict[str, Any]]:

    cmd = ["docker", "ps", "--format", "{{json .}}"]
    if include_all:
        cmd.insert(2, "-a")
    result = run_command(cmd)
    containers: list[dict[str, Any]] = []
    for line in (result.stdout or "").strip().split("\n"):
        if not line:
            continue
        try:
            containers.append(json.loads(line))
        except json.JSONDecodeError:
            logger.debug("Skipping unparsable docker ps line: %s", line)
    return containers


def list_containers_by_image_substring(
    image_substring: str, *, include_all: bool = False
) -> list[dict[str, Any]]:

    substring = (image_substring or "").lower()
    return [
        c for c in docker_ps(include_all=include_all) if substring in (c.get("Image", "").lower())
    ]


def docker_rm(container_name: str, *, force: bool = True, volumes: bool = False) -> None:

    cmd = ["docker", "rm"]
    if force:
        cmd.append("-f")
    if volumes:
        cmd.append("-v")
    cmd.append(container_name)
    run_command(cmd)


def build_run_command(
    *,
    name: str,
    image: str,
    port_mappings: Iterable[tuple[int, int]] | None = None,
    env_vars: dict[str, str] | None = None,
    volumes: Iterable[tuple[str, str]] | None = None,
    detach: bool = True,
    privileged: bool = True,
    remove: bool = True,
) -> list[str]:

    cmd: list[str] = [
        "docker",
        "run",
    ]
    if remove:
        cmd.append("--rm")
    if privileged:
        cmd.append("--privileged")
    cmd.extend(["--name", name])

    for host_port, container_port in port_mappings or []:
        cmd.extend(["-p", f"{host_port}:{container_port}"])

    for host_path, container_path in volumes or []:
        cmd.extend(["-v", f"{host_path}:{container_path}"])

    for key, value in (env_vars or {}).items():
        cmd.extend(["-e", f"{key}={value}"])

    if detach:
        cmd.append("-d")

    cmd.append(image)
    return cmd


def is_port_available(port: int, host: str = "0.0.0.0") -> bool:

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def find_next_container_index(prefix: str = DEFAULT_NAME_PREFIX, dev_mode: bool = False) -> int:

    containers = docker_ps(include_all=True)
    existing_indices = []
    suffix = "_dev" if dev_mode else ""

    for container in containers:
        name = container.get("Names", "")
        if name.startswith(f"{prefix}_"):
            remainder = name[len(prefix) + 1 :]
            if suffix and remainder.endswith(suffix):
                remainder = remainder[: -len(suffix)]

            try:
                idx = int(remainder)
                existing_indices.append(idx)
            except ValueError:
                continue

    if not existing_indices:
        return 0

    return max(existing_indices) + 1


def wait_for_container_ready(
    backend_port: int,
    timeout: int = 120,
    poll_interval: float = 1.0,
) -> bool:

    health_url = f"http://localhost:{backend_port}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("ok", False):
                    return True
        except (requests.ConnectionError, requests.Timeout, requests.RequestException):
            pass

        time.sleep(poll_interval)

    return False


def launch_container(
    config: ContainerConfig,
    detach: bool = True,
    wait_ready: bool = True,
    ready_timeout: int = 600,
) -> LaunchResult:

    result = LaunchResult(
        name=config.name,
        backend_port=config.backend_port,
        viewer_port=config.viewer_port,
        vnc_port=config.vnc_port,
        adb_port=config.adb_port,
    )

    envs: dict[str, str] = {}
    if config.enable_vnc or config.dev_mode:
        envs["ENABLE_VNC"] = "true"

    volumes: list[tuple[str, str]] = []
    if config.dev_src_path:
        volumes.append((str(config.dev_src_path), "/app/service/src"))
    if config.env_file_path:
        volumes.append((str(config.env_file_path.resolve()), "/app/service/.env"))

    cmd = build_run_command(
        name=config.name,
        image=config.image,
        port_mappings=[
            (config.backend_port, 6800),
            (config.viewer_port, 7860),
            (config.vnc_port, 5800),
            (config.adb_port, 5556),
        ],
        env_vars=envs,
        volumes=volumes,
        detach=detach,
        privileged=True,
        remove=True,
    )

    try:
        run_result = run_command(cmd)
        if run_result.returncode == 0:
            result.success = True
            logger.info("Container '%s' launched successfully", config.name)

            if wait_ready:
                if wait_for_container_ready(config.backend_port, timeout=ready_timeout):
                    result.ready = True
                    logger.info("Container '%s' is ready", config.name)
                else:
                    logger.warning("Container '%s' did not become ready in time", config.name)
        else:
            result.error_message = run_result.stderr
            logger.error("Failed to launch container '%s'", config.name)
    except Exception as e:
        result.error_message = str(e)
        logger.exception("Error launching container '%s'", config.name)

    return result


def list_containers(
    image_filter: str = DEFAULT_IMAGE,
    name_prefix: str | None = DEFAULT_NAME_PREFIX,
    include_all: bool = False,
) -> list[ContainerInfo]:

    containers = list_containers_by_image_substring(image_filter, include_all=include_all)

    result = []
    for container in containers:
        name = container.get("Names", "")

        if name_prefix and not name.startswith(name_prefix):
            continue

        ports_info = container.get("Ports", "")
        backend_port = None
        viewer_port = None
        vnc_port = None

        if ports_info:
            for port_mapping in ports_info.split(", "):
                if "->" in port_mapping:
                    host_part, container_port = port_mapping.split("->")
                    container_port_num = container_port.split("/")[0]
                    try:
                        host_port = int(host_part.split(":")[-1])
                        if container_port_num == "6800":
                            backend_port = host_port
                        elif container_port_num == "7860":
                            viewer_port = host_port
                        elif container_port_num == "5800":
                            vnc_port = host_port
                    except ValueError:
                        pass

        result.append(
            ContainerInfo(
                name=name,
                status=container.get("Status"),
                running="Up" in container.get("Status", ""),
                backend_port=backend_port,
                viewer_port=viewer_port,
                vnc_port=vnc_port,
            )
        )

    return result

import logging
import socket
import subprocess
import time
import uuid
from collections.abc import Callable

import httpx

from mai_gym.models.config import DockerLaunchConfig

logger = logging.getLogger(__name__)

_HEALTH_TIMEOUT = 10.0
_HEALTH_INTERVAL = 2.0
_HEALTH_RETRIES = 30
_PORT_SCAN_LIMIT = 1000
_MW_REUSE_HEALTH_TIMEOUT = 10
_OSW_REUSE_HEALTH_TIMEOUT = 10
_OSW_LAUNCH_STAGGER = 5
_OSW_FRESH_WAVE_SIZE = 5
_OSW_SERVER_PORT_INSIDE = "5000/tcp"


def _is_port_free_local(port: int) -> bool:

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _is_port_free_remote(host: str, user: str, key: str | None, port: int) -> bool:

    ssh = _ssh_base_cmd(host, user, key)
    probe = (
        f"(ss -Htln 2>/dev/null || netstat -tln 2>/dev/null) "
        f"| awk '{{print $4}}' | grep -E '[:.]{port}$' -q && echo USED || echo FREE"
    )
    ssh.append(probe)
    try:
        result = subprocess.run(ssh, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        logger.warning("[port-probe] remote probe failed on %s:%d — %s", host, port, exc)
        return True
    return "FREE" in result.stdout


def _wait_for_health(url: str, retries: int = _HEALTH_RETRIES) -> bool:

    for _ in range(retries):
        try:
            r = httpx.get(f"{url}/health", timeout=_HEALTH_TIMEOUT)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(_HEALTH_INTERVAL)
    return False


def _discover_device_id(url: str) -> str | None:

    try:
        r = httpx.get(f"{url}/devices", timeout=_HEALTH_TIMEOUT)
        if r.status_code == 200:
            devices = [d for d in r.json().get("devices", []) if d]
            if devices:
                return devices[0]
    except Exception:
        pass

    try:
        r = httpx.get(f"{url}/health", timeout=_HEALTH_TIMEOUT)
        if r.status_code == 200:
            data = r.json()
            status = data.get("device_status") or {}
            alive = [d for d, ok in status.items() if ok and d]
            if alive:
                return alive[0]
            devices = [d for d in data.get("devices", []) if d]
            if devices:
                return devices[0]
    except Exception:
        pass
    return None


def _make_backend_id() -> str:
    return f"rb-{uuid.uuid4().hex[:12]}"


def _ssh_base_cmd(host: str, user: str, key: str | None) -> list[str]:

    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10"]
    if key:
        cmd += ["-i", key]
    cmd.append(f"{user}@{host}")
    return cmd


def respect_launch_interval(
    last_launch_at: dict[str, float],
    device_name: str,
    interval: int,
    *,
    on_wait: Callable[[float], None] | None = None,
) -> None:

    if interval <= 0:
        last_launch_at[device_name] = time.monotonic()
        return
    last = last_launch_at.get(device_name)
    if last is not None:
        elapsed = time.monotonic() - last
        if elapsed < interval:
            wait = interval - elapsed
            if on_wait is not None:
                on_wait(wait)
            time.sleep(wait)
    last_launch_at[device_name] = time.monotonic()


def allocate_4port_slot(
    launch: DockerLaunchConfig,
    next_index: dict[str, int],
    claimed_ports: dict[str, set[int]],
    device_name: str,
    is_port_free: Callable[[int], bool],
    *,
    exclude_slots: frozenset[int] = frozenset(),
    seed_index: Callable[[], int] = lambda: 0,
    location_desc: str = "",
) -> tuple[int, int, int, int, int]:

    if device_name not in next_index:
        next_index[device_name] = seed_index()
        claimed_ports[device_name] = set()
    claimed = claimed_ports[device_name]
    idx = next_index[device_name]
    for attempt in range(_PORT_SCAN_LIMIT):
        slot = idx + attempt
        if slot in exclude_slots:
            continue
        backend = launch.port_range_start + slot
        viewer = launch.viewer_port_range_start + slot
        vnc = launch.vnc_port_range_start + slot
        adb = launch.adb_port_range_start + slot
        ports = (backend, viewer, vnc, adb)
        if any(p in claimed for p in ports):
            continue
        if not all(is_port_free(p) for p in ports):
            continue
        for p in ports:
            claimed.add(p)
        next_index[device_name] = slot + 1
        return (slot, backend, viewer, vnc, adb)
    raise RuntimeError(
        f"No free 4-port slot{location_desc} for device {device_name} "
        f"within {_PORT_SCAN_LIMIT} of {launch.port_range_start}"
    )

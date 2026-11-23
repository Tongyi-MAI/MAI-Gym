# Modified by MAI-Gym contributors.
import asyncio
import ipaddress
import json
import logging
import multiprocessing
import os
import random
import socket

import httpx

logger = logging.getLogger(__name__)

SLIME_HOST_IP_ENV = "SLIME_HOST_IP"


def find_available_port(base_port: int):
    port = base_port + random.randint(100, 1000)
    while True:
        if is_port_available(port):
            return port
        if port < 60000:
            port += 42
        else:
            port -= 43


def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except OSError:
            return False
        except OverflowError:
            return False


def get_host_info():
    hostname = socket.gethostname()

    if env_overwrite_local_ip := os.getenv(SLIME_HOST_IP_ENV, None):
        return hostname, env_overwrite_local_ip

    def _is_loopback(ip):
        return ip.startswith("127.") or ip == "::1"

    def _resolve_ip(family, test_target_ip):

        try:
            with socket.socket(family, socket.SOCK_DGRAM) as s:
                s.connect((test_target_ip, 80))
                ip = s.getsockname()[0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        try:
            infos = socket.getaddrinfo(hostname, None, family=family, type=socket.SOCK_STREAM)

            for info in infos:
                ip = info[4][0]
                if not _is_loopback(ip):
                    return ip
        except Exception:
            pass

        return None

    prefer_ipv6 = os.getenv("SLIME_PREFER_IPV6", "0").lower() in ("1", "true", "yes", "on")
    local_ip = None
    final_fallback = "127.0.0.1"

    if prefer_ipv6:
        local_ip = _resolve_ip(socket.AF_INET6, "2001:4860:4860::8888")
        final_fallback = "::1"
    else:
        local_ip = _resolve_ip(socket.AF_INET, "8.8.8.8")
        final_fallback = "127.0.0.1"

    return hostname, local_ip or final_fallback


def _wrap_ipv6(host):
    try:
        ipaddress.IPv6Address(host.strip("[]"))
        return f"[{host.strip('[]')}]"
    except ipaddress.AddressValueError:
        return host


def run_router(args):
    try:
        from sglang_router.launch_router import launch_router

        router = launch_router(args)
        if router is None:
            return 1
        return 0
    except Exception as e:
        logger.info(e)
        return 1


def terminate_process(process: multiprocessing.Process, timeout: float = 1.0) -> None:
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=timeout)
    if process.is_alive():
        process.kill()
        process.join()


_http_client: httpx.AsyncClient | None = None
_client_concurrency: int = 0

_distributed_post_enabled: bool = False
_post_actors: list[object] = []
_post_actor_idx: int = 0


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


async def _post(client, url, payload, max_retries=60, headers=None):
    retry_count = 0
    while retry_count < max_retries:
        response = None
        try:
            response = await client.post(url, json=payload or {}, headers=headers)
            response.raise_for_status()
            content = await response.aread()
            try:
                output = json.loads(content)
            except json.JSONDecodeError:
                output = content.decode() if isinstance(content, bytes) else content
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                if 400 <= e.response.status_code < 500 and e.response.status_code not in (408, 429):
                    raise
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        finally:
            if response is not None:
                await response.aclose()
        break

    return output


def init_http_client(args):
    global _http_client, _client_concurrency, _distributed_post_enabled
    if not args.rollout_num_gpus:
        return

    _client_concurrency = args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
            trust_env=False,
        )

    if args.use_distributed_post:
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args):
    global _post_actors
    if _post_actors:
        return

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(1, concurrency)),
                timeout=httpx.Timeout(None),
                trust_env=False,
            )

        async def do_post(self, url, payload, max_retries=60, headers=None):
            return await _post(self._client, url, payload, max_retries, headers=headers)

    created = []
    per_actor_conc = (_client_concurrency + len(nodes)) // len(nodes)

    for node in nodes:
        node_id = node["NodeID"]
        scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
        for _ in range(args.num_gpus_per_node):
            actor = _HttpPosterActor.options(
                name=None,
                lifetime="detached",
                scheduling_strategy=scheduling,
                max_concurrency=per_actor_conc,
                num_cpus=0.001,
            ).remote(per_actor_conc)
            created.append(actor)

    _post_actors = created


async def post(url, payload, max_retries=60, headers=None):
    if _distributed_post_enabled and _post_actors:
        try:
            actor = _next_actor()
            if actor is not None:
                obj_ref = actor.do_post.remote(url, payload, max_retries, headers=headers)
                return await obj_ref
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")

    return await _post(_http_client, url, payload, max_retries, headers=headers)


async def get(url):
    response = await _http_client.get(url)
    response.raise_for_status()
    content = await response.aread()
    output = json.loads(content)
    return output

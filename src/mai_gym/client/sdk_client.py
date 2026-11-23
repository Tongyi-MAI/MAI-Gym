from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import uuid

import httpx

from mai_gym.client.env_client import APIError, EnvClient
from mai_gym.models.sdk import MAX_RESULT_BYTES, RESULT_CHUNK_BYTES, SDK_CAPABILITY


class SDKInfrastructureError(RuntimeError):
    pass


class SDKManagerClient(EnvClient):
    def __init__(self, *args, **kwargs):
        kwargs["timeout"] = min(kwargs.get("timeout", 20), 20)
        super().__init__(*args, **kwargs)

    def _post(self, path, payload=None, timeout=None):
        return super()._post(path, payload, timeout=min(timeout or 20, 20))

    async def _apost(self, path, payload=None, timeout=None):
        async with asyncio.timeout(20):
            return await super()._apost(path, payload, timeout=min(timeout or 20, 20))


class SDKClient:
    def __init__(
        self,
        client: EnvClient,
        session_id: str,
        *,
        request_timeout: float = 20,
        operation_timeout: float = 630,
        poll_interval: float = 1,
    ):
        if not 0 < request_timeout < 90:
            raise ValueError("SDK HTTP deadline must be below the 90-second proxy limit")
        self.client, self.session_id = client, session_id
        self.lease_id = client._lease_cache[session_id]
        self.request_timeout = request_timeout
        self.operation_timeout = operation_timeout
        self.poll_interval = poll_interval

    @staticmethod
    async def require_capability(client: EnvClient):
        async with asyncio.timeout(20):
            response = await client._apost("/sdk/capabilities", timeout=20)
            if SDK_CAPABILITY not in response.json().get("capabilities", []):
                raise SDKInfrastructureError("Manager does not support SDK operations v1")

    async def _request(self, route, payload):

        async with asyncio.timeout(self.request_timeout):
            response = await self.client._apost(
                f"/sdk/{self.session_id}/{route}",
                payload,
                timeout=self.request_timeout,
            )
            return response.json()

    async def _query(self, route, payload):
        while True:
            try:
                return await self._request(route, payload)
            except APIError as exc:
                if exc.status_code < 500:
                    raise SDKInfrastructureError(str(exc)) from exc
            except (httpx.HTTPError, TimeoutError):
                pass
            await asyncio.sleep(self.poll_interval)

    async def operation(self, kind, *, operation_id=None, timeout=None, **kwargs):
        operation_id = operation_id or uuid.uuid4().hex
        ref = {"lease_id": self.lease_id, "operation_id": operation_id}
        async with asyncio.timeout(timeout or self.operation_timeout):
            try:
                await self._request("submit", {**ref, "kind": kind, **kwargs})
            except APIError as exc:
                if exc.status_code < 500:
                    raise SDKInfrastructureError(str(exc)) from exc
            except (httpx.HTTPError, TimeoutError):
                pass
            while True:
                status = await self._query("status", ref)
                if status.get("operation_id") != operation_id:
                    raise SDKInfrastructureError("Manager returned the wrong operation identity")
                if status["status"] == "completed":
                    break
                if status["status"] != "pending":
                    raise SDKInfrastructureError(status.get("error") or str(status))
                await asyncio.sleep(self.poll_interval)
            size, count = status["bytes"], status["chunks"]
            if not (
                0 < size <= MAX_RESULT_BYTES
                and count == (size + RESULT_CHUNK_BYTES - 1) // RESULT_CHUNK_BYTES
            ):
                raise SDKInfrastructureError("Invalid SDK result transfer manifest")
            parts = []
            for chunk in range(count):
                page = await self._query("result", {**ref, "chunk": chunk})
                if page.get("operation_id") != operation_id or page.get("chunk") != chunk:
                    raise SDKInfrastructureError("SDK result chunk identity mismatch")
                part = base64.b64decode(page["data"], validate=True)
                if len(part) > RESULT_CHUNK_BYTES:
                    raise SDKInfrastructureError("SDK result chunk exceeds transfer budget")
                parts.append(part)
            data = b"".join(parts)
            if len(data) != size or hashlib.sha256(data).hexdigest() != status["sha256"]:
                raise SDKInfrastructureError("Incomplete or corrupted SDK result transfer")
            return await asyncio.to_thread(json.loads, data)

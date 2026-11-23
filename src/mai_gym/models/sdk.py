from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mai_gym.models.requests import ResetRequest

SDK_CAPABILITY = "sdk_operations.v1"
RESULT_CHUNK_BYTES = 256 * 1024
MAX_RESULT_BYTES = 48 * 1024 * 1024


class SDKOperationRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease_id: str = Field(min_length=1, max_length=128)
    operation_id: str = Field(min_length=1, max_length=128)


class SDKResultRequest(SDKOperationRef):
    chunk: int = Field(ge=0)


class SDKOperationRequest(SDKOperationRef):
    kind: Literal["reset", "open", "execute", "finish", "cancel"]
    reset: ResetRequest | None = None
    device: str | None = None
    coordinate_space: Literal["pixels", "normalized", "normalized_1000"] = "normalized_1000"
    cell_id: str | None = Field(default=None, min_length=1, max_length=128)
    code: str | None = Field(default=None, min_length=1, max_length=262144)

    @model_validator(mode="after")
    def required_arguments(self):
        if self.kind == "reset" and (self.reset is None or not self.reset.task_id):
            raise ValueError("SDK reset requires a concrete MobileWorld task")
        if self.kind == "execute" and (self.cell_id is None or self.code is None):
            raise ValueError("execute requires cell_id and code")
        return self

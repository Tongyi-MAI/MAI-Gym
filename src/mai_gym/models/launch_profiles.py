from pydantic import BaseModel, Field, model_validator


class MobileWorldPatchConfig(BaseModel):
    enabled: bool = False

    directory: str | None = Field(default=None, min_length=1)
    bundle_path: str | None = None
    bundle_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    max_concurrent_patches: int = Field(default=64, ge=1)
    restart_timeout_seconds: float = Field(default=180.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_enabled_bundle(self) -> "MobileWorldPatchConfig":
        if self.directory is not None:
            if not self.directory.strip():
                raise ValueError("MobileWorld patch directory cannot be blank")
            if self.bundle_path is not None or self.bundle_sha256 is not None:
                raise ValueError(
                    "MobileWorld patch directory cannot combine with pinned bundle fields"
                )
        elif self.enabled and (not self.bundle_path or not self.bundle_sha256):
            raise ValueError("enabled MobileWorld patch requires bundle_path and bundle_sha256")
        return self


class MobileWorldTaskArtifactConfig(BaseModel):
    enabled: bool = False
    root_path: str | None = None
    max_concurrent_uploads: int = Field(default=16, ge=1)
    upload_timeout_seconds: float = Field(default=600.0, gt=0.0)
    chunk_size_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=64 * 1024,
        le=16 * 1024 * 1024,
    )

    @model_validator(mode="after")
    def _validate_enabled_root(self) -> "MobileWorldTaskArtifactConfig":
        if self.enabled and not self.root_path:
            raise ValueError("enabled MobileWorld task artifacts require root_path")
        return self

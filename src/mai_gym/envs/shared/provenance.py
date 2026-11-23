import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


class PreparationError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_file(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_file():
        raise PreparationError(
            f"Missing profile asset: {path}. Restore the recorded asset before preparing this profile."
        )
    return path


def verify_asset_tree(root: Path, asset: dict[str, Any]) -> None:
    directory = (root / asset.get("path", "")).resolve()
    manifest = resolve_file(root, asset.get("manifest", ""))
    if not directory.is_dir() or sha256(manifest) != asset.get("sha256"):
        raise PreparationError(f"Asset tree or manifest does not match its profile: {directory}")
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, list) or not entries:
        raise PreparationError(f"Asset tree manifest must list pinned files: {manifest}")
    seen = set()
    for entry in entries:
        relative = entry.get("path", "") if isinstance(entry, dict) else None
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise PreparationError(f"Asset tree paths must be relative strings: {manifest}")
        relative_path = Path(relative)
        path = (directory / relative_path).resolve()
        identity = relative_path.as_posix()
        if (
            not relative
            or ".." in relative_path.parts
            or identity in seen
            or not path.is_relative_to(directory)
        ):
            raise PreparationError(f"Invalid or duplicate asset tree path: {relative!r}")
        seen.add(identity)
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise PreparationError(f"Missing or mismatching pinned tree asset: {path}")
    asset.update(path=str(directory), manifest=str(manifest))


def verify_profile(benchmark: str, settings: dict[str, Any]) -> dict[str, Any]:
    value = settings.get("profile_lock")
    if not value:
        raise PreparationError(
            f"{benchmark}: env_config.profile_lock is required. Supply the recorded evaluation lock with runtime/evaluator commits, task-manifest SHA256 and asset identities; see skills/envs/{benchmark}.md."
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise PreparationError(f"{benchmark}: profile lock not found: {path}")
    profile: dict[str, Any] = json.loads(path.read_text())
    if profile.get("benchmark") != benchmark:
        raise PreparationError(f"Profile benchmark must be {benchmark!r}")
    root = path.parent
    sources = profile.get("sources", {})
    if not isinstance(sources, dict) or not {"runtime", "evaluator"}.issubset(sources):
        raise PreparationError(f"{benchmark}: sources must include runtime and evaluator pins")
    for name, source in sources.items():
        if not isinstance(source, dict):
            raise PreparationError(f"{benchmark}: sources.{name} must describe a pinned checkout")
        if source.get("kind") == "docker" and benchmark == "mobileworld":
            if profile.get("label", "paper") == "paper":
                raise PreparationError("MobileWorld paper sources require their recorded Git pins")
            image_assets = profile.get("assets")
            if not isinstance(image_assets, list) or not any(
                isinstance(asset, dict)
                and asset.get("kind") == "docker"
                and asset.get("image") == source.get("image")
                and asset.get("id") == source.get("id")
                for asset in image_assets
            ):
                raise PreparationError(
                    f"{benchmark}: bundled {name} source must match a verified Docker asset"
                )
            continue
        revision = source.get("revision", "")
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise PreparationError(
                f"{benchmark}: sources.{name}.revision requires a recorded full Git commit"
            )
        checkout = (root / source.get("path", "")).resolve()
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode or result.stdout.strip() != revision:
            raise PreparationError(
                f"{benchmark}: {name} checkout {checkout} does not match recorded commit {revision}"
            )
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        )
        if dirty.stdout.strip():
            raise PreparationError(
                f"{benchmark}: {name} has unrecorded source changes at {checkout}; restore the pinned tree or supply a separately verified patch bundle"
            )
        source["path"] = str(checkout)
    manifest = profile.get("task_manifest", {})
    manifest_path = resolve_file(root, manifest.get("path", ""))
    expected = manifest.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or sha256(manifest_path) != expected:
        raise PreparationError(
            f"{benchmark}: paper task manifest SHA256 mismatch at {manifest_path}"
        )
    manifest["path"] = str(manifest_path)
    assets = profile.get("assets")
    if not isinstance(assets, list) or not assets:
        raise PreparationError(
            f"{benchmark}: record required assets (VM/image/browser/application state) in the profile lock"
        )
    for asset in assets:
        if asset.get("kind") == "docker":
            image = asset.get("image", "")
            expected = asset.get("id")
            if not expected or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
                raise PreparationError(
                    f"{benchmark}: Docker asset requires an immutable sha256 image id"
                )
            result = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode or result.stdout.strip() != expected:
                raise PreparationError(
                    f"{benchmark}: cached Docker image {image!r} does not match its recorded identity"
                )
        elif asset.get("kind") == "tree":
            verify_asset_tree(root, asset)
        else:
            asset_path = resolve_file(root, asset.get("path", ""))
            if sha256(asset_path) != asset.get("sha256"):
                raise PreparationError(f"{benchmark}: asset SHA256 mismatch: {asset_path}")
            asset["path"] = str(asset_path)
    profile["lock_sha256"] = sha256(path)
    profile["lock_path"] = str(path)
    return profile

# Modified by MAI-Gym contributors.

import json
import math
import os
import struct
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


@contextmanager
def preserve_untrained_mtp_weights(bridge, *, enabled: bool):
    state = bridge.hf_pretrained.state
    source = state.source
    expected = set(state)
    original = source.save_generator
    previous = vars(source).get("save_generator")
    had_override = "save_generator" in vars(source)

    def save_generator(generator, output_path, strict=True):
        def complete():
            seen = set()
            for item in generator:
                seen.add(item.param_name)
                yield item
            missing = expected - seen
            unsupported = {name for name in missing if not (enabled and name.startswith("mtp."))}
            if unsupported:
                raise ValueError(f"HF export omitted {len(unsupported)} active tensors: {sorted(unsupported)[:5]}")
            for name in sorted(missing):
                yield SimpleNamespace(param_name=name, weight=state[name].detach().cpu())

        return original(complete(), output_path, strict=strict)

    source.save_generator = save_generator
    try:
        yield
    finally:
        if had_override:
            source.save_generator = previous
        else:
            del source.save_generator


def finalize_hf_export(path: Path, expected_keys: set[str]) -> dict[str, int]:
    path = Path(path)
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map", {})
    else:
        index = None
        weight_map = {name: "model.safetensors" for name in expected_keys}
    if not weight_map or set(weight_map) != expected_keys:
        raise ValueError("HF export index does not contain every expected tensor")

    item_sizes = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U64": 8,
        "U32": 4,
        "U16": 2,
        "U8": 1,
        "BOOL": 1,
        "F8_E4M3": 1,
        "F8_E5M2": 1,
    }
    total_size = 0
    filenames = set(weight_map.values())
    for filename in filenames:
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("HF export index contains an unsafe shard filename")
        shard = path / filename
        size = shard.stat().st_size
        with shard.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("HF export contains a truncated shard")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size > 64 * 1024 * 1024 or header_size + 8 > size:
                raise ValueError("HF export contains an invalid shard header")
            header = json.loads(stream.read(header_size))
        tensors = {name: spec for name, spec in header.items() if name != "__metadata__"}
        if set(tensors) != {name for name, value in weight_map.items() if value == filename}:
            raise ValueError("HF export shard contents disagree with its index")
        ranges = []
        for spec in tensors.values():
            start, end = spec["data_offsets"]
            shape = spec["shape"]
            if any(type(dim) is not int or dim < 0 for dim in shape):
                raise ValueError("HF export contains an invalid tensor shape")
            expected_bytes = math.prod(shape) * item_sizes[spec["dtype"]]
            if start < 0 or end < start or end - start != expected_bytes:
                raise ValueError("HF export contains an invalid tensor byte range")
            if end > start:
                ranges.append((start, end))
        position = 0
        for start, end in sorted(ranges):
            if start != position:
                raise ValueError("HF export contains overlapping or missing tensor data")
            position = end
        if position != size - header_size - 8:
            raise ValueError("HF export tensor data is incomplete")
        total_size += position

    if index is not None and index.get("metadata", {}).get("total_size") != total_size:
        index.setdefault("metadata", {})["total_size"] = total_size
        temporary = index_path.with_suffix(".json.partial")
        with temporary.open("w") as stream:
            json.dump(index, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(index_path)
    return {"tensors": len(weight_map), "shards": len(filenames), "total_size": total_size}

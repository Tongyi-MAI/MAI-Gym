from __future__ import annotations


def gen_header_suffix(token_ids: list[int], asst_open: list[int]) -> list[int]:

    k = len(asst_open)
    for i in range(len(token_ids) - k, -1, -1):
        if token_ids[i : i + k] == asst_open:
            return token_ids[i + k :]
    return []


def splice_assistant_output_ids(
    token_ids: list[int],
    output_ids_list: list[list[int]],
    asst_open: list[int],
    im_end_id: int,
    forbidden: frozenset[int] = frozenset(),
) -> list[int]:
    for step, ids in enumerate(output_ids_list):
        bad = [token for token in ids if token in forbidden]
        if bad:
            raise ValueError(
                f"output_ids of assistant turn {step} carry {bad[:4]}, which only the "
                "processor may place; refusing to build this prompt"
            )
    k = len(asst_open)
    out: list[int] = []
    step = 0
    i = 0
    n = len(token_ids)
    while i < n:
        is_empty_asst = (
            token_ids[i : i + k] == asst_open and i + k < n and token_ids[i + k] == im_end_id
        )
        if is_empty_asst:
            if step >= len(output_ids_list):
                raise ValueError(
                    f"more empty assistant turns than output_ids "
                    f"({step + 1} > {len(output_ids_list)})"
                )
            out.extend(asst_open)
            output_ids = output_ids_list[step]
            out.extend(output_ids)
            step += 1
            i += k
            if output_ids and output_ids[-1] == im_end_id:
                i += 1
        else:
            out.append(token_ids[i])
            i += 1
    if step != len(output_ids_list):
        raise ValueError(
            f"fewer empty assistant turns than output_ids ({step} < {len(output_ids_list)})"
        )
    return out


VISION_TOKENS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
)
TERMINAL_TOKENS = ("<|im_end|>", "<|endoftext|>")


def token_ids(tokenizer, tokens) -> frozenset[int]:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return frozenset()
    unk = getattr(tokenizer, "unk_token_id", None)
    return frozenset(
        tid for token in tokens if isinstance(tid := convert(token), int) and tid != unk
    )


def control_token_ids(tokenizer) -> frozenset[int]:
    special = {int(tid) for tid in getattr(tokenizer, "all_special_ids", None) or ()}
    for tid, token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        if getattr(token, "special", False):
            special.add(int(tid))
    return frozenset(special) | token_ids(tokenizer, VISION_TOKENS)


def first_control_index(
    output_ids: list[int],
    control: frozenset[int],
    terminal: frozenset[int],
) -> int | None:
    last = len(output_ids) - 1
    for index, tid in enumerate(output_ids):
        if tid in control and not (index == last and tid in terminal):
            return index
    return None

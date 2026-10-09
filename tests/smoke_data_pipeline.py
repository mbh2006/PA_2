"""Local data-pipeline smoke test for Task 1 (no model forward).

Exercises the real plumbing: chat template, encode_prompt_response, pad_batch,
the training collate function, and response-mask semantics — using synthetic
preference rows and the real Qwen2.5 tokenizer (small download).

Run:
    D:\\MyTools\\Anaconda\\envs\\atml-pa0\\python.exe tests/smoke_data_pipeline.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from common.data import (  # noqa: E402
    encode_prompt_response,
    preference_responses,
    prompt_messages_from_preference,
)
from common.models import load_tokenizer  # noqa: E402
from task1_dpo.train import make_collate  # noqa: E402

TOKENIZER_ID = "Qwen/Qwen2.5-0.5B-Instruct"

ROWS = [
    {
        "source_index": i,
        "prompt_id": f"p{i}",
        "prompt": f"Question {i}: what is {i}+{i}?",
        "chosen": [
            {"role": "user", "content": f"Question {i}: what is {i}+{i}?"},
            {"role": "assistant", "content": f"The answer is {2 * i}."},
        ],
        "rejected": [
            {"role": "user", "content": f"Question {i}: what is {i}+{i}?"},
            {"role": "assistant", "content": f"Probably {2 * i + 1}."},
        ],
    }
    for i in range(4)
]


def main():
    tok = load_tokenizer(TOKENIZER_ID)
    collate = make_collate(tok, 128)
    batch_c, batch_r = collate(ROWS)

    for name, b in (("chosen", batch_c), ("rejected", batch_r)):
        ids, am, rm = b["input_ids"], b["attention_mask"], b["response_mask"]
        assert ids.shape == am.shape == rm.shape, name
        assert rm.sum() > 0, f"{name}: response mask empty"
        assert (am[rm == 1] == 1).all(), f"{name}: response tokens must be attended"
        # response tokens are the right-hand side: last response token must be attended
        assert am[0, -1] == 1, f"{name}: last token must be real (left padding)"
        print(f"{name}: shape={tuple(ids.shape)} response_tokens={int(rm.sum())}")

    c, r = preference_responses(ROWS[0])
    p = prompt_messages_from_preference(ROWS[0])
    assert c == "The answer is 0." and r == "Probably 1."
    assert p == [{"role": "user", "content": "Question 0: what is 0+0?"}]

    # Truncation must preserve prompt and keep EOS.
    ids, mask = encode_prompt_response(tok, p, "word " * 1000, 64)
    assert len(ids) <= 64 and len(ids) == len(mask)
    assert mask[-1] == 1 and sum(mask) < len(mask)

    print("All data-pipeline smoke checks passed.")


if __name__ == "__main__":
    main()

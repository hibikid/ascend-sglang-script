#!/usr/bin/env python3
"""Send identical cold/reused prompts to manually launched mempool-min services.

This checks one simple pencil answer, not full model accuracy or ticket acceptance.
It owns no server processes; the user keeps the three launch terminals running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXPECTED: dict[str, int | str] = {
    "after_take": 14,
    "after_add": 18,
    "enough": "no",
    "shortage": 2,
    "final": 18,
}
EXPECTED_TEXT = "\n".join(f"{key}={value}" for key, value in EXPECTED.items())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def check_answer(text: str) -> dict[str, Any]:
    """Parse the five requested facts; formatting trouble is not a wrong answer."""
    facts: dict[str, Any] = {}
    try:
        for line in unicodedata.normalize("NFKC", text).splitlines():
            line = re.sub(r"\s+", "", line).replace("、", ",")
            if not line or line.startswith("```"):
                continue
            match = re.fullmatch(
                r"(after_take|after_add|enough|shortage|final)[=:](.+)", line
            )
            if match is None or match[1] in facts:
                raise ValueError("expected five unique answer lines")
            key, value = match.groups()
            if key == "enough":
                if value.lower() not in ("yes", "no"):
                    raise ValueError("enough: expected yes or no")
                facts[key] = value.lower()
            else:
                if not re.fullmatch(r"-?\d+", value):
                    raise ValueError(f"{key}: expected an integer quantity")
                facts[key] = int(value)
        if set(facts) != set(EXPECTED):
            raise ValueError("missing answer lines")
    except ValueError as exc:
        return dict(status="unparsed", facts=facts, error=str(exc))
    wrong = [key for key in EXPECTED if facts[key] != EXPECTED[key]]
    return dict(status="incorrect" if wrong else "correct", facts=facts, wrong=wrong)


def make_request(tokenizer: Any, prompt: str) -> tuple[dict[str, Any], str]:
    marker = "__MEMPOOL_USER_START_7F83__"
    if marker in prompt:
        raise ValueError("prompt contains reserved template marker")
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker + prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if chat.count(marker) != 1:
        raise ValueError("chat template did not preserve the user-content marker")
    before, after = chat.split(marker)
    head = tokenizer.encode(before, add_special_tokens=False)
    tail = tokenizer.encode(after, add_special_tokens=False)
    space = tokenizer.encode(" ", add_special_tokens=False)
    if len(space) != 1 or space[0] in tokenizer.all_special_ids:
        raise ValueError("tokenizer must encode one ordinary space as one token")
    padding = 4096 - len(head) - len(tail)
    if padding < 0:
        raise ValueError("pencil prompt exceeds the fixed 4096-token input")
    # Insert blank token IDs inside user content, preserving the complete question
    # and assistant generation suffix. Do not decode/re-tokenize before sending.
    ids = head + space * padding + tail
    if len(tokenizer.encode(EXPECTED_TEXT, add_special_tokens=False)) >= 128:
        raise ValueError("expected answer does not fit the 128-token output budget")
    return dict(
        input_ids=ids,
        sampling_params=dict(temperature=0, max_new_tokens=128, ignore_eos=False),
        stream=False,
    ), tokenizer.decode(ids, skip_special_tokens=False)


def http_json(url: str, payload: Any = None, *, timeout: float = 10) -> Any:
    request = urllib.request.Request(
        url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(urls: list[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    pending = set(urls)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    print("Waiting for P, D and Main /ready", flush=True)
    while pending:
        for url in list(pending):
            try:
                # Worker /ready responds with an empty HTTP 200 body.
                with opener.open(
                    url, timeout=min(5, max(0.1, deadline - time.monotonic()))
                ) as response:
                    if response.status != 200:
                        continue
                pending.remove(url)
                print("READY " + url, flush=True)
            except (OSError, ValueError):
                pass
        if pending:
            if time.monotonic() >= deadline:
                raise TimeoutError("services not ready: " + ", ".join(sorted(pending)))
            time.sleep(1)


def check_idle(state: dict[str, Any]) -> None:
    if not state.get("ready") or state.get("fault") or state.get("active") is not None:
        raise ValueError("Main must be ready, idle and fault-free")
    if any(v["outstanding"] for k, v in state["ledger"].items() if k != "p_prefix"):
        raise ValueError("request resources have not been released")


def run_pair(
    config: dict[str, Any],
    run_id: str,
    request: dict[str, Any],
    root: Path,
    report: dict[str, Any],
    fetch: Any = http_json,
) -> None:
    main = f"http://{config['prefill_host']}:{config['main_port']}"
    initial = fetch(main + "/status")
    write_json(root / "status-before.json", initial)
    check_idle(initial)
    if (
        initial["completed"]
        or initial["directory"]
        or initial["ledger"]["p_prefix"]["outstanding"]
    ):
        raise ValueError(
            "A must be cold: restart Main and both workers with a fresh RUN_ID"
        )
    report["results"] = []
    for number, (name, expected_hit) in enumerate((("A", 0), ("C", 3968)), 1):
        print(
            f"Sending {name}: 4096 input tokens, expected hit_len={expected_hit}",
            flush=True,
        )
        response = fetch(
            main + "/generate", request, timeout=config["operation_timeout"] + 30
        )
        write_json(root / f"response-{name}.json", response)
        text = response.get("text", "")
        (root / f"response-{name}.txt").write_text(text)
        print(f"\n--- {name} output ---\n{text}\n--- end {name} ---", flush=True)
        state = fetch(main + "/status")
        write_json(root / f"status-{name}.json", state)
        check_idle(state)
        history = state["completed"]
        rid = f"{run_id}-{number}"
        if len(history) != number or history[-1]["rid"] != rid:
            raise ValueError(
                "unexpected request history or RUN_ID; use a dedicated fresh service"
            )
        if (
            history[-1]["result"] != response
            or response.get("meta_info", {}).get("id") != rid
        ):
            raise ValueError("response does not match the completed Router request")
        hit = history[-1]["hit_len"]
        if hit != expected_hit:
            raise ValueError(f"{name}: expected hit_len={expected_hit}, got {hit}")
        meta = response.get("meta_info", {})
        ids = response.get("output_ids")
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError(f"{name}: missing valid output token IDs")
        if meta.get("prompt_tokens") != 4096 or meta.get("completion_tokens") != len(
            ids
        ):
            raise ValueError(
                f"{name}: response token counts do not match the request/output"
            )
        answer = check_answer(text)
        finish = meta.get("finish_reason", {}).get("type")
        # A wrong/truncated answer still permits C, so both actual texts survive.
        result = dict(
            name=name,
            rid=rid,
            hit_len=hit,
            output_ids=ids,
            completion_tokens=len(ids),
            finish_reason=finish,
            answer=answer,
        )
        report["results"].append(result)
        write_json(root / "report.json", report)
        print(
            f"{name}: hit_len={hit}, answer={answer['status']}, finish={finish}",
            flush=True,
        )
    a, c = report["results"]
    left, right = a["output_ids"], c["output_ids"]
    first = next(
        (i for i, (x, y) in enumerate(zip(left, right)) if x != y),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )
    parsed = all(r["answer"]["status"] != "unparsed" for r in (a, c))
    report.update(
        token_ids_equal=left == right,
        first_output_difference=first,
        logically_equal=a["answer"]["facts"] == c["answer"]["facts"]
        if parsed
        else None,
        status="passed"
        if all(
            r["answer"]["status"] == "correct"
            and r["finish_reason"] == "stop"
            and r["completion_tokens"] <= 128
            for r in (a, c)
        )
        else "failed",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="encode/save the request without contacting services",
    )
    args = parser.parse_args()
    run_id = os.environ.get("RUN_ID", "glm52-mempool-ac")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id):
        raise ValueError("RUN_ID must be a simple name of at most 80 characters")
    config_path = Path(
        os.environ.get("CONFIG_FILE", str(HERE / "glm52mempool-demo.json"))
    ).resolve()
    config = json.loads(config_path.read_text())
    if (
        config.get("prefix_pages"),
        config.get("match_prefix"),
        config.get("page_size"),
        config.get("decode_capacity"),
    ) != (40, True, 128, 128):
        raise ValueError(
            "A/C smoke requires prefix_pages=40, match_prefix=true, page_size=128, decode_capacity=128"
        )
    root = Path(
        os.environ.get("LOG_DIR", f"/home/cryang/log/mempool-demo/{run_id}")
    ).resolve()
    if not root.is_relative_to(Path("/home/cryang/log").resolve()):
        raise ValueError("LOG_DIR must remain under /home/cryang/log/")
    root = root / ("ac-prepared" if args.prepare_only else "ac")
    root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = dict(
        status="running",
        coverage="simple pencil A/C semantic smoke only",
        ticket_acceptance=False,
        run_id=run_id,
        expected=EXPECTED,
    )
    try:
        from transformers import AutoTokenizer

        prompt = (HERE / "mempool-demo-prompt.txt").read_text()
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], trust_remote_code=True
        )
        request, rendered = make_request(tokenizer, prompt)
        write_json(root / "config.json", config)
        write_json(root / "request.json", request)
        (root / "prompt.txt").write_text(prompt)
        (root / "rendered-prompt.txt").write_text(rendered)
        (root / "expected.txt").write_text(EXPECTED_TEXT + "\n")
        report["input_sha256"] = hashlib.sha256(
            json.dumps(request["input_ids"]).encode()
        ).hexdigest()
        if args.prepare_only:
            report["status"] = "prepared"
        else:
            wait_ready(
                [
                    f"http://{config['prefill_host']}:{config['prefill_port']}/ready",
                    f"http://{config['decode_host']}:{config['decode_port']}/ready",
                    f"http://{config['prefill_host']}:{config['main_port']}/ready",
                ],
                config["startup_timeout"],
            )
            run_pair(config, run_id, request, root, report)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        write_json(root / "report.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("results", "expected")},
            ensure_ascii=False,
        ),
        flush=True,
    )
    print(f"Saved request, A/C outputs, Router evidence and report: {root}", flush=True)
    return 0 if report["status"] in ("passed", "prepared") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

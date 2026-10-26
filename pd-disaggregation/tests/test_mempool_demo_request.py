"""CPU checks of the client contract; no model arithmetic is simulated."""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "client", ROOT / "request-mempool-demo.py"
)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


CORRECT = "after_take=14\nafter_add=18\nenough=no\nshortage=2\nfinal=18"


class Tokenizer:
    all_special_ids = [1, 2]

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and add_generation_prompt and not enable_thinking
        return "<user>" + messages[0]["content"] + "</user><assistant></think>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode())

    def decode(self, ids, *, skip_special_tokens):
        return bytes(ids).decode()


class Endpoint:
    def __init__(self, answers, hits=(0, 3968), finish="stop"):
        self.answers, self.hits, self.finish = answers, hits, finish
        self.requests = []
        self.state = dict(
            ready=True,
            fault=None,
            active=None,
            completed=[],
            directory={},
            ledger={"p_prefix": {"outstanding": 0}, "p_alive": {"outstanding": 0}},
        )

    def __call__(self, url, payload=None, **kwargs):
        if url.endswith("/status"):
            return copy.deepcopy(self.state)
        assert url.endswith("/generate")
        self.requests.append(copy.deepcopy(payload))
        number = len(self.requests)
        # Different IDs stand for punctuation-only differences in the answers.
        response = dict(
            text=self.answers[number - 1],
            output_ids=[number],
            meta_info=dict(
                id=f"test-{number}",
                prompt_tokens=4096,
                completion_tokens=1,
                finish_reason={"type": self.finish},
            ),
        )
        self.state["completed"].append(
            dict(rid=f"test-{number}", result=response, hit_len=self.hits[number - 1])
        )
        self.state["ledger"]["p_prefix"]["outstanding"] = 32
        return copy.deepcopy(response)


class ClientTests(unittest.TestCase):
    def test_answer_formats_and_wrong_fact(self):
        fullwidth = CORRECT.replace("=", "：").replace(",", "、")
        self.assertEqual(client.check_answer(fullwidth)["status"], "correct")
        wrong = client.check_answer(CORRECT.replace("final=18", "final=16"))
        self.assertEqual(wrong["status"], "incorrect")
        self.assertEqual(wrong["wrong"], ["final"])
        self.assertEqual(client.check_answer("我认为答案正确")["status"], "unparsed")

    def test_full_question_and_generation_suffix_survive_padding(self):
        prompt = (ROOT / "mempool-demo-prompt.txt").read_text()
        request, rendered = client.make_request(Tokenizer(), prompt)
        self.assertEqual(len(request["input_ids"]), 4096)
        self.assertTrue(rendered.endswith(prompt + "</user><assistant></think>"))
        self.assertEqual(rendered.split(prompt)[0].strip(), "<user>")
        self.assertFalse(request["sampling_params"]["ignore_eos"])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            client.make_request(Tokenizer(), "x" * 5000)

    def run_pair(self, endpoint):
        report = {"status": "running"}
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            root = Path(tmp)
            request = {"input_ids": [7] * 4096}
            client.run_pair(
                dict(prefill_host="host", main_port=6699, operation_timeout=600),
                "test",
                request,
                root,
                report,
                endpoint,
            )
            self.assertEqual(
                json.loads((root / "response-C.json").read_text())["text"],
                endpoint.answers[1],
            )
        return report

    def test_two_correct_answers_pass_despite_token_difference(self):
        endpoint = Endpoint([CORRECT, CORRECT.replace("=", "：")])
        report = self.run_pair(endpoint)
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["logically_equal"])
        self.assertFalse(report["token_ids_equal"])
        self.assertEqual(endpoint.requests[0], endpoint.requests[1])

    def test_wrong_a_still_runs_c_and_reports_difference(self):
        endpoint = Endpoint(
            [
                CORRECT.replace("after_take=14", "after_take=15"),
                CORRECT,
            ]
        )
        report = self.run_pair(endpoint)
        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["logically_equal"])
        self.assertEqual(len(endpoint.requests), 2)

    def test_truncated_correct_text_is_not_a_pass(self):
        report = self.run_pair(Endpoint([CORRECT] * 2, finish="length"))
        self.assertEqual(report["status"], "failed")

    def test_c_must_reuse_the_expected_prefix(self):
        with self.assertRaisesRegex(ValueError, "expected hit_len=3968"):
            self.run_pair(Endpoint([CORRECT] * 2, hits=(0, 0)))

    def test_used_service_is_rejected_before_sending(self):
        endpoint = Endpoint([CORRECT] * 2)
        endpoint.state["ledger"]["p_prefix"]["outstanding"] = 32
        with self.assertRaisesRegex(ValueError, "A must be cold"):
            self.run_pair(endpoint)
        self.assertFalse(endpoint.requests)

    def test_ready_accepts_empty_200_response(self):
        class Response(io.BytesIO):
            status = 200

        with patch.object(client.urllib.request, "build_opener") as opener:
            opener.return_value.open.side_effect = [
                Response(b""),
                Response(b""),
                Response(b"{}"),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                client.wait_ready(
                    ["http://p/ready", "http://d/ready", "http://m/ready"], 1
                )
            self.assertEqual(opener.return_value.open.call_count, 3)


if __name__ == "__main__":
    unittest.main()

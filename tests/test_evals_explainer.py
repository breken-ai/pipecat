#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import unittest

from pipecat.evals.explainer import (
    JUDGE_FINAL_SYSTEM_INSTRUCTION,
    JUDGE_SYSTEM_INSTRUCTION,
    EvalExplainer,
    _parse_run_verdicts,
    _parse_verdict,
)


class TestParseRunVerdicts(unittest.TestCase):
    def test_the_goal_and_a_verdict_per_turn_per_criterion(self):
        out = _parse_run_verdicts(
            '{"goal": {"verdict": "yes", "reason": "Berlin was named."}, '
            '"turns": {"politeness": ["yes", "no"], "brevity": ["yes", "yes"]}, '
            '"reasons": {"politeness": {"2": "curt"}}}',
            ["politeness", "brevity"],
            2,
        )
        self.assertEqual((out.goal.verdict, out.goal.reason), ("yes", "Berlin was named."))
        self.assertEqual([v.verdict for v in out.turns["politeness"]], ["yes", "no"])
        self.assertEqual(out.turns["politeness"][1].reason, "curt")
        self.assertEqual(out.turns["politeness"][0].reason, "")
        self.assertEqual([v.verdict for v in out.turns["brevity"]], ["yes", "yes"])

    def test_a_short_array_or_a_missing_criterion_fails_the_turns_it_lacks(self):
        out = _parse_run_verdicts(
            '```json\n{"goal": {"verdict": "no"}, "turns": {"Politeness": ["yes"]}}\n```',
            ["politeness", "brevity"],
            2,
        )
        self.assertEqual((out.goal.verdict, out.goal.reason), ("no", "(no reason given)"))
        self.assertEqual([v.verdict for v in out.turns["politeness"]], ["yes", "none"])
        self.assertEqual(out.turns["politeness"][1].reason, "(judge gave no verdict)")
        self.assertEqual([v.reason for v in out.turns["brevity"]], ["(judge gave no verdict)"] * 2)

    def test_a_failed_call_or_no_json_fails_everything(self):
        out = _parse_run_verdicts("\0explainer call failed: Boom", ["politeness"], 1)
        self.assertEqual(
            (out.goal.verdict, out.goal.reason), ("none", "explainer call failed: Boom")
        )
        self.assertEqual(out.turns["politeness"][0].reason, "explainer call failed: Boom")
        out = _parse_run_verdicts("no json here", ["politeness"], 1)
        self.assertEqual(out.goal.verdict, "none")
        self.assertEqual(out.turns["politeness"][0].verdict, "none")


class TestParseVerdict(unittest.TestCase):
    def test_clean_json_yes(self):
        v = _parse_verdict('{"verdict": "yes", "reason": "It mentions weather."}')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "It mentions weather.")

    def test_clean_json_no(self):
        v = _parse_verdict('{"verdict": "no", "reason": "Does not mention it."}')
        self.assertFalse(v.passed)
        self.assertEqual(v.verdict, "no")
        self.assertEqual(v.reason, "Does not mention it.")

    def test_clean_json_continue(self):
        v = _parse_verdict('{"verdict": "continue", "reason": "Just a filler so far."}')
        self.assertEqual(v.verdict, "continue")
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "Just a filler so far.")

    def test_unknown_verdict_fails_closed(self):
        v = _parse_verdict('{"verdict": "maybe", "reason": "x"}')
        self.assertEqual(v.verdict, "no")

    def test_unstructured_continue_fallback(self):
        v = _parse_verdict("continue, more text is needed")
        self.assertEqual(v.verdict, "continue")

    def test_fenced_json(self):
        v = _parse_verdict('```json\n{"verdict": "yes", "reason": "ok"}\n```')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "ok")

    def test_fenced_json_without_lang(self):
        v = _parse_verdict('```\n{"verdict": "yes", "reason": "ok"}\n```')
        self.assertTrue(v.passed)

    def test_trailing_prose_after_json(self):
        # Models sometimes append chatty text after the JSON object, and "know"
        # contains "no", so a substring-matching fallback would misread the
        # trailing sentence as a rejection.
        v = _parse_verdict(
            ' {"verdict": "yes", "reason": "The bot greets the user."}\n\n'
            "Let me know if you'd like to evaluate any further turns!"
        )
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "The bot greets the user.")

    def test_leading_prose_before_json(self):
        v = _parse_verdict('Sure, here is my verdict: {"verdict": "no", "reason": "wrong"}')
        self.assertFalse(v.passed)
        self.assertEqual(v.reason, "wrong")

    def test_unstructured_yes_fallback(self):
        v = _parse_verdict("yes, this satisfies the criterion")
        self.assertTrue(v.passed)

    def test_unstructured_no_fallback(self):
        v = _parse_verdict("no, it does not")
        self.assertFalse(v.passed)

    def test_ambiguous_response_fails_closed(self):
        v = _parse_verdict("the answer is yes or possibly no")
        self.assertFalse(v.passed)
        self.assertIn("could not parse", v.reason)

    def test_garbage_response(self):
        v = _parse_verdict("???")
        self.assertFalse(v.passed)

    def test_extra_whitespace(self):
        v = _parse_verdict('  \n {"verdict": "yes", "reason": "x"}  \n ')
        self.assertTrue(v.passed)

    def test_missing_reason(self):
        v = _parse_verdict('{"verdict": "yes"}')
        self.assertTrue(v.passed)
        self.assertEqual(v.reason, "(no reason given)")


class _FakeLLMService:
    """In-memory stand-in for a pipecat LLM service.

    Records every call and returns a queued response.
    """

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def run_inference(
        self,
        context,
        max_tokens=None,
        system_instruction=None,
        response_schema=None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(context._messages),
                "max_tokens": max_tokens,
                "system_instruction": system_instruction,
            }
        )
        if not self._responses:
            raise RuntimeError("FakeLLMService: no more queued responses")
        return self._responses.pop(0)


class TestExplain(unittest.IsolatedAsyncioTestCase):
    async def test_the_reply_is_explained_in_conversation_context(self):
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "four"}'])
        explainer = EvalExplainer(svc)
        transcript = [
            {"role": "user", "content": "What is two plus two?"},
            {"role": "assistant", "content": "That's for"},  # terse + STT homophone
        ]
        verdict = await explainer.explain(transcript, "answers that two plus two is four")
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.reason, "four")
        roles = [m["role"] for m in svc.calls[0]["messages"]]
        contents = [m["content"] for m in svc.calls[0]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])  # question, reply, verdict ask
        self.assertIn("What is two plus two?", contents)
        self.assertIn("That's for", contents)

    async def test_a_reply_is_explained_on_the_spoken_conversation_only(self):
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        transcript = [
            {"role": "tool", "content": "lookup()"},
            {"role": "assistant", "content": "It's 72 and sunny."},
        ]
        await EvalExplainer(svc).explain(transcript, "describes the weather")
        self.assertEqual([m["role"] for m in svc.calls[0]["messages"]], ["assistant", "user"])

    async def test_an_answer_is_cached_by_question_and_conversation(self):
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        explainer = EvalExplainer(svc)
        transcript = [{"role": "assistant", "content": "It rains."}]
        self.assertTrue((await explainer.explain(transcript, "mentions weather")).passed)
        self.assertTrue((await explainer.explain(transcript, "mentions weather")).passed)
        self.assertEqual(len(svc.calls), 1, "second call should be cached")

    async def test_a_service_failure_is_reported_not_raised(self):
        class _BoomService:
            async def run_inference(self, **kwargs):
                raise RuntimeError("network down")

        verdict = await EvalExplainer(_BoomService()).explain([], "anything")
        self.assertFalse(verdict.passed)
        self.assertIn("RuntimeError", verdict.reason)

    async def test_an_empty_response_is_a_no(self):
        verdict = await EvalExplainer(_FakeLLMService([""])).explain([], "anything")
        self.assertFalse(verdict.passed)

    async def test_a_reply_is_asked_yes_or_no_without_continue(self):
        svc = _FakeLLMService(['{"verdict": "no", "reason": "never names Berlin"}'])
        explainer = EvalExplainer(svc, allow_continue=False)
        transcript = [{"role": "assistant", "content": "No rush, take your time."}]
        verdict = await explainer.explain(transcript, "says the capital of Germany is Berlin")
        self.assertEqual(verdict.verdict, "no")
        self.assertEqual(svc.calls[0]["system_instruction"], JUDGE_FINAL_SYSTEM_INSTRUCTION)
        self.assertIn("Answer yes or no.", svc.calls[0]["messages"][-1]["content"])

    async def test_a_continue_answer_counts_as_no_without_continue(self):
        svc = _FakeLLMService(['{"verdict": "continue", "reason": "still going"}'])
        explainer = EvalExplainer(svc, allow_continue=False)
        verdict = await explainer.explain(
            [{"role": "assistant", "content": "Let me think."}], "gives an answer"
        )
        self.assertEqual(verdict.verdict, "no")

    async def test_continue_is_allowed_by_default(self):
        svc = _FakeLLMService(['{"verdict": "continue", "reason": "still going"}'])
        verdict = await EvalExplainer(svc).explain(
            [{"role": "assistant", "content": "Let me check."}], "gives the weather"
        )
        self.assertEqual(verdict.verdict, "continue")
        self.assertEqual(svc.calls[0]["system_instruction"], JUDGE_SYSTEM_INSTRUCTION)


class TestExplainCall(unittest.IsolatedAsyncioTestCase):
    async def test_the_ask_names_the_call_and_its_arguments(self):
        svc = _FakeLLMService(['{"verdict": "no", "reason": "wrong speaker"}'])
        verdict = await EvalExplainer(svc).explain_call(
            [], "submit", {"speaker": "Ann"}, "submitted for Bob"
        )
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.reason, "wrong speaker")
        ask = svc.calls[0]["messages"][-1]["content"]
        self.assertIn('called the function `submit` with arguments `{"speaker": "Ann"}`', ask)
        self.assertIn("Criterion: submitted for Bob", ask)

    async def test_a_call_is_asked_for_a_yes_or_no_verdict(self):
        """A call is judged under its own instructions: there is no reply to wait for."""
        svc = _FakeLLMService(['{"verdict": "yes", "reason": "ok"}'])
        await EvalExplainer(svc).explain_call([], "submit", {"speaker": "Ann"}, "submitted for Ann")
        instruction = svc.calls[0]["system_instruction"]
        self.assertIn("function call", instruction)
        self.assertNotIn("continue", instruction)


class TestExplainRun(unittest.IsolatedAsyncioTestCase):
    async def test_the_whole_conversation_goes_in_one_ask(self):
        """Reply segments merge into one numbered bot turn; tool calls sit inline."""
        svc = _FakeLLMService(
            [
                '{"goal": {"verdict": "yes", "reason": "booked"}, '
                '"turns": {"polite": ["yes", "yes"]}}'
            ]
        )
        transcript = [
            {"role": "user", "content": "Book a table at six."},
            {"role": "tool", "content": 'book({"time": "6pm"})'},
            {"role": "assistant", "content": "Let me check."},
            {"role": "assistant", "content": "Done, six o'clock."},
            {"role": "user", "content": "Thanks."},
            {"role": "assistant", "content": "You're welcome."},
        ]

        verdicts = await EvalExplainer(svc).explain_run(
            transcript, {"polite": "is polite"}, "a table is booked"
        )

        ask = svc.calls[0]["messages"][-1]["content"]
        self.assertIn("User: Book a table at six.", ask)
        self.assertIn('[tool call] book({"time": "6pm"})', ask)
        self.assertIn("Bot turn 1: Let me check. Done, six o'clock.", ask)
        self.assertIn("Bot turn 2: You're welcome.", ask)
        self.assertIn("there are 2 bot turns", ask)
        self.assertTrue(verdicts.goal.passed)
        self.assertEqual([v.verdict for v in verdicts.turns["polite"]], ["yes", "yes"])


if __name__ == "__main__":
    unittest.main()

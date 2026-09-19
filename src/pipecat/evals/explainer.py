#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The explainer: an LLM that says why a verdict went the way it did.

A classifier decides a verdict but cannot say what it saw, so the judge
(:class:`~pipecat.evals.judge.EvalJudge`) hands the verdicts that need a
reason to an explainer. The explainer is an LLM asked the same question,
with a one-shot inference outside the pipeline, and it answers with its own
verdict and a one-sentence reason. Any Pipecat LLM service with
``run_inference()`` works: OpenAI, Ollama, Together, and others.

The explainer never decides anything: the judge keeps the classifier's
verdict and reports the explainer's opinion alongside it when the two
disagree.

Example::

    from pipecat.services.ollama.llm import OLLamaLLMService

    service = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="gemma4:12b"))
    explainer = EvalExplainer(service)
    transcript = [
        {"role": "user", "content": "What can you help me with?"},
        {"role": "assistant", "content": "I can answer questions and set reminders."},
    ]
    verdict = await explainer.explain(transcript, "describes the bot's capabilities")
    print(verdict.reason)
"""

import json
import re
from collections.abc import Sequence
from typing import Any

from loguru import logger

from pipecat.evals.services import llm_service_from_config
from pipecat.evals.verdicts import NO_REASON, NO_VERDICT, JudgeVerdict, RunVerdicts, cache_key
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import LLMService

JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a conversation between a user and a "
    "bot under test. The 'user' messages are the user; the 'assistant' messages are "
    "the bot's replies. Judge only the bot's most recent reply — which may have "
    "arrived as several consecutive 'assistant' messages — against the given "
    "criterion, using the earlier turns only as context. The reply may still be "
    "streaming in. "
    "When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation. Always judge it by the intended spoken meaning, never by its exact "
    "spelling. In particular, treat a number as the same value whether it is spelled out, "
    "written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and "
    "'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error "
    "when the intended spoken meaning satisfies the criterion. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no" | "continue", "reason": "<one short sentence>"}. '
    'Use "yes" if the reply satisfies the criterion. '
    'Use "continue" if the bot has not given its answer yet: it says it is checking, '
    "looking something up, fetching, working on something, or that it will report back. "
    "The answer is still coming, so there is nothing to judge yet. This holds however "
    'long and however fluent the reply is: "The system is checking the current '
    'conditions for you right now." is waiting, not answering. A greeting or an '
    'obviously incomplete fragment is also "continue". '
    'Use "no" only when the bot has given its answer and that answer fails the '
    'criterion. If the bot has not answered yet, always use "continue", never "no". '
    "Do not include any other text, explanation, or markdown."
)

# Transient final user message appended for the explainer's call. The
# conversation it refers to ("the bot's most recent reply") is the LLMContext
# built from the judge's transcript; this just poses the question.
JUDGE_ASK_TEMPLATE = (
    "Does the bot's most recent reply satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes, no, or continue."
)

# The instructions for a reply when ``continue`` isn't allowed: every judged
# reply is a final answer, so the verdict is yes or no.
JUDGE_FINAL_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a conversation between a user and a "
    "bot under test. The 'user' messages are the user; the 'assistant' messages are "
    "the bot's replies. Judge only the bot's most recent reply — which may have "
    "arrived as several consecutive 'assistant' messages — against the given "
    "criterion, using the earlier turns only as context. The reply is the bot's final "
    "answer. "
    "When the bot spoke its reply, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation. Always judge it by the intended spoken meaning, never by its exact "
    "spelling. In particular, treat a number as the same value whether it is spelled out, "
    "written as a digit, or transcribed as a homophone: 'for' and 'fore' mean 'four' (4), and "
    "'to' and 'too' mean 'two' (2). Never answer 'no' solely because of a transcription error "
    "when the intended spoken meaning satisfies the criterion. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no", "reason": "<one short sentence>"}. '
    'Use "yes" if the reply satisfies the criterion and "no" if it does not. '
    "Do not include any other text, explanation, or markdown."
)

JUDGE_FINAL_ASK_TEMPLATE = (
    "Does the bot's most recent reply satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes or no."
)

# The instructions for an ``eval:`` on a function call. The call is the
# subject, and the conversation is context for it, so the verdict is yes or
# no: a call is not a partial reply, and there is nothing to wait for.
JUDGE_CALL_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a function call made by a bot under "
    "test in a conversation with a user. The 'user' messages are the user; the "
    "'assistant' messages are the bot's replies so far, given only as context for the "
    "call. Judge only the call you are asked about, by its name and its arguments, "
    "against the given criterion. "
    "When the bot spoke its replies, the 'assistant' text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, and "
    "missing punctuation; judge it by the intended spoken meaning. "
    "Respond ONLY with a JSON object on a single line containing two fields: "
    '{"verdict": "yes" | "no", "reason": "<one short sentence>"}. '
    'Use "yes" if the call satisfies the criterion and "no" if it does not. '
    "Do not include any other text, explanation, or markdown."
)

# The ask for an ``eval:`` on a function call. It names the call and gives its
# arguments as JSON, so the verdict is about that call rather than about what
# the bot said around it.
JUDGE_CALL_ASK_TEMPLATE = (
    "The bot called the function `{name}` with arguments `{args}`. "
    "Does this call satisfy this criterion?\n\n"
    "Criterion: {criterion}\n\n"
    "Answer yes or no."
)


RUN_JUDGE_SYSTEM_INSTRUCTION = (
    "You are a strict but fair judge evaluating a complete conversation between a user "
    "and a bot under test, given as a transcript. The bot's replies are numbered 'Bot "
    "turn 1', 'Bot turn 2', and so on; the user's lines are marked 'User'; a line "
    "marked '[tool call]' is a function the bot called at that point, and a completed "
    "call is stronger evidence of an action (a booking, a lookup) than the bot saying "
    "it did it. "
    "You are given criteria, each with a name, that every bot reply is judged against "
    "on its own, in the light of the conversation before it, and a goal that the "
    "conversation as a whole is judged against. A criterion that forbids something "
    "('never ...', 'does not ...') or that applies only in a situation ('when ...', "
    "'if ...') is satisfied by a reply that does not do the forbidden thing or is not "
    "in that situation; do not fault a reply for something the criterion does not ask "
    "of it. "
    "When the bot spoke its replies, its text is an automatic speech-to-text "
    "transcription, so it may contain homophones, misspellings, split or merged words, "
    "and missing punctuation. Always judge it by the intended spoken meaning, never by "
    "its exact spelling. "
    "Respond ONLY with a JSON object on a single line of the form "
    '{"goal": {"verdict": "yes" | "no", "reason": "<one short sentence>"}, '
    '"turns": {"<criterion name>": ["yes" | "no", ...]}, '
    '"reasons": {"<criterion name>": {"<bot turn number>": "<one short sentence>"}}}. '
    'Under "turns", give every criterion an array with exactly one entry per bot '
    'turn, in order. Under "reasons", give a reason only for the turns you '
    'answered "no". Do not include any other text, explanation, or markdown.'
)

RUN_JUDGE_ASK_TEMPLATE = (
    "Transcript:\n{transcript}\n\n"
    "Criteria for every bot reply:\n{criteria}\n\n"
    "Goal for the conversation as a whole: {success}\n\n"
    "Answer with the JSON described, one array entry per bot turn: there are "
    "{turn_count} bot turns."
)


class EvalExplainer:
    """Asks an LLM the question a classifier answered, for the reason behind it.

    Its answers are cached by question and conversation, so explaining the
    same verdict twice costs one call.

    Args:
        service: A pipecat LLM service with a ``run_inference()`` method
            (i.e. ``BaseOpenAILLMService`` or any subclass: OpenAI, Ollama, etc.).
        max_tokens: Cap on the explainer's response length. Default 200 —
            enough for a JSON verdict + short reason.
        allow_continue: Whether a reply may be judged ``continue``; when
            ``False``, a reply is ``yes`` or ``no``. It follows the judge's
            own setting, so both are asked the same question.
    """

    def __init__(
        self, service: LLMService[Any], *, max_tokens: int = 200, allow_continue: bool = True
    ):
        """Initialize the explainer with a configured pipecat LLM service.

        Args:
            service: A pipecat LLM service exposing ``run_inference()``.
            max_tokens: Cap on the explainer's response length.
            allow_continue: Whether a reply may be judged ``continue``.
        """
        self._service = service
        self._max_tokens = max_tokens
        self._allow_continue = allow_continue
        self._cache: dict[str, JudgeVerdict] = {}
        self._run_cache: dict[str, RunVerdicts] = {}

    @classmethod
    def from_config(cls, config: dict | None, *, allow_continue: bool = True) -> "EvalExplainer":
        """Build an explainer from an LLM block, as a scenario's ``judge.eval:`` takes.

        Args:
            config: Mapping with keys ``service`` (default ``"ollama"``),
                ``model`` (default ``"gemma4:12b"``), optional ``endpoint``
                (service-specific default if omitted) and an optional ``extra``
                mapping forwarded to the model as top-level request parameters.
                ``None`` uses all defaults.
            allow_continue: Whether a reply may be judged ``continue``.

        Returns:
            A configured EvalExplainer.

        Raises:
            ValueError: If ``service`` is unknown.
        """
        return cls(
            llm_service_from_config(config, where="judge.eval"), allow_continue=allow_continue
        )

    @property
    def service(self) -> LLMService[Any]:
        """The LLM service that answers."""
        return self._service

    async def explain(self, transcript: Sequence[dict], criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, and say why.

        Args:
            transcript: The conversation so far, as the judge kept it.
            criterion: Natural-language description of what the reply should express.

        Returns:
            The explainer's own verdict, with a one-sentence reason.
        """
        if self._allow_continue:
            ask = JUDGE_ASK_TEMPLATE.format(criterion=criterion)
            return await self._evaluate(transcript, criterion, JUDGE_SYSTEM_INSTRUCTION, ask)
        ask = JUDGE_FINAL_ASK_TEMPLATE.format(criterion=criterion)
        verdict = await self._evaluate(transcript, criterion, JUDGE_FINAL_SYSTEM_INSTRUCTION, ask)
        if verdict.verdict == "continue":
            # An answer that ignored the yes/no instructions counts as a no.
            return JudgeVerdict(
                verdict="no", reason=verdict.reason, raw_response=verdict.raw_response
            )
        return verdict

    async def explain_call(
        self, transcript: Sequence[dict], name: str, args: dict | None, criterion: str
    ) -> JudgeVerdict:
        """Judge whether a function call the bot made satisfies ``criterion``, and say why.

        The ask names the call and its arguments, the conversation so far is
        context, and the verdict is yes or no: a call is not a partial reply,
        so there is nothing to wait for.

        Args:
            transcript: The conversation so far, as the judge kept it.
            name: The function's name.
            args: The call's arguments, shown to the explainer as JSON.
            criterion: Natural-language description of what the call should be.

        Returns:
            The explainer's own verdict, with a one-sentence reason.
        """
        ask = JUDGE_CALL_ASK_TEMPLATE.format(
            name=name, args=json.dumps(args or {}, ensure_ascii=False), criterion=criterion
        )
        return await self._evaluate(transcript, criterion, JUDGE_CALL_SYSTEM_INSTRUCTION, ask)

    async def explain_run(
        self, transcript: Sequence[dict], criteria: dict[str, str], success: str
    ) -> RunVerdicts:
        """Judge the whole conversation in one call: every bot turn on every criterion, and the goal.

        The conversation goes in the question, bot turns numbered and tool
        calls inline. A bot turn is a run of reply segments with nothing else
        between them.

        Args:
            transcript: The conversation, as the judge kept it.
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order, each with a reason for a ``no``. A verdict of ``none`` is
            one the explainer did not give: a turn it left out, a goal it did
            not answer, or a call that failed.
        """
        lines: list[str] = []
        turn = 0
        for entry in transcript:
            if entry["role"] == "assistant":
                if lines and lines[-1].startswith(f"Bot turn {turn}:"):
                    lines[-1] += f" {entry['content']}"
                    continue
                turn += 1
                lines.append(f"Bot turn {turn}: {entry['content']}")
            elif entry["role"] == "tool":
                lines.append(f"[tool call] {entry['content']}")
            else:
                lines.append(f"User: {entry['content']}")
        listed = "\n".join(f"- {name}: {criterion}" for name, criterion in criteria.items())
        ask = RUN_JUDGE_ASK_TEMPLATE.format(
            transcript="\n".join(lines) or "(nothing was said)",
            criteria=listed or "(none)",
            success=success,
            turn_count=turn,
        )
        key = cache_key(ask)
        if key not in self._run_cache:
            # Room for a verdict per turn per criterion, a reason per "no", and
            # the goal's verdict; a budget sized for one verdict cuts it short.
            budget = max(300, 4 * turn * len(criteria) + 60 * len(criteria) + 80)
            response = await self._ask(
                success, [], RUN_JUDGE_SYSTEM_INSTRUCTION, ask, max_tokens=budget
            )
            self._run_cache[key] = _parse_run_verdicts(response, list(criteria), turn)
        return self._run_cache[key]

    async def _evaluate(
        self, transcript: Sequence[dict], criterion: str, instruction: str, ask: str
    ) -> JudgeVerdict:
        # The spoken conversation only: a reply is judged on what was said.
        messages = [e for e in transcript if e["role"] != "tool"]
        key = cache_key(ask, messages)
        if key not in self._cache:
            response = await self._ask(criterion, messages, instruction, ask)
            if response.startswith("\0"):
                self._cache[key] = JudgeVerdict(verdict="no", reason=response[1:], raw_response="")
            else:
                self._cache[key] = _parse_verdict(response)
        return self._cache[key]

    async def _ask(
        self,
        criterion: str,
        messages: list,
        instruction: str,
        ask: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        """The explainer's raw answer to ``ask``.

        A failed or empty call comes back as a NUL-prefixed reason, which no
        answer starts with, so callers can report it as a ``no``.
        """
        # Copy the conversation and append the transient ask, so neither the ask
        # nor the answer ever lands in the persistent context.
        context = LLMContext(messages=list(messages))
        context.add_message({"role": "user", "content": ask})

        # Log the conversation the explainer is about to read, before its answer,
        # so the debug log shows exactly what it saw (handy when a terse or
        # mis-transcribed reply gets an unexpected verdict). A run-level ask
        # carries the transcript itself, so that is what to show.
        transcript = "\n".join(f"  [{m.get('role')}] {m.get('content')}" for m in messages)
        logger.debug(
            "Explainer evaluating {!r} over conversation:\n{}",
            criterion,
            transcript or "\n".join(f"  {line}" for line in ask.splitlines()),
        )

        try:
            response = await self._service.run_inference(
                context=context,
                max_tokens=self._max_tokens if max_tokens is None else max_tokens,
                system_instruction=instruction,
            )
        except Exception as e:
            logger.error(f"Explainer call failed: {e.__class__.__name__} ({e})")
            return f"\0explainer call failed: {e.__class__.__name__}"

        if not response:
            return "\0explainer returned empty response"

        return response


def _parse_run_verdicts(response: str, names: list[str], turn_count: int) -> RunVerdicts:
    """Parse the run answer into the goal's verdict and one per turn per criterion.

    Anything missing or malformed is a ``none`` with a reason, and the raw
    answer is logged, so a bad answer never passes a turn silently.
    """
    if response.startswith("\0"):
        failed = JudgeVerdict(verdict="none", reason=response[1:], raw_response="")
        return RunVerdicts(goal=failed, turns={n: [failed] * turn_count for n in names})
    obj = _judge_json(response)
    goal = obj.get("goal")
    if not isinstance(goal, dict):
        goal = {}
    answer = str(goal.get("verdict", "")).strip().lower()
    goal_verdict = answer if answer in ("yes", "no") else "none"
    goal_reason = str(goal.get("reason", "")).strip()
    if goal_verdict == "none":
        goal_reason = NO_VERDICT
    elif goal_verdict == "no" and not goal_reason:
        goal_reason = NO_REASON
    return RunVerdicts(
        goal=JudgeVerdict(verdict=goal_verdict, reason=goal_reason, raw_response=response),
        turns={name: _turn_verdicts(obj, name, turn_count, response) for name in names},
    )


def _judge_json(response: str) -> dict:
    """The JSON object in the answer, or ``{}`` when there is none; a fenced or prefaced answer still parses."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    if start != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, AttributeError):
            pass
    logger.warning(f"Explainer answer was not the expected JSON: {response!r}")
    return {}


def _turn_verdicts(obj: dict, name: str, turn_count: int, response: str) -> list[JudgeVerdict]:
    """One verdict per bot turn for criterion ``name``; a turn left out is a ``none``.

    Criterion names match case-insensitively; a ``reasons`` entry, keyed by
    the turn number, gives a ``no`` its reason.
    """
    turns_by_name = {
        str(k).lower(): v for k, v in (obj.get("turns") or {}).items() if isinstance(v, list)
    }
    reasons_by_name = {
        str(k).lower(): v for k, v in (obj.get("reasons") or {}).items() if isinstance(v, dict)
    }
    answers = turns_by_name.get(name.lower(), [])
    reasons = reasons_by_name.get(name.lower(), {})
    if len(answers) != turn_count:
        logger.warning(
            f"Explainer gave {len(answers)} verdict(s) for {name!r} over {turn_count} bot "
            f"turn(s); its answer was: {response!r}"
        )
    verdicts = []
    for index in range(turn_count):
        answer = answers[index] if index < len(answers) else None
        if isinstance(answer, dict):
            answer = answer.get("verdict")
        if answer is None:
            verdicts.append(JudgeVerdict(verdict="none", reason=NO_VERDICT, raw_response=response))
            continue
        verdict = "yes" if str(answer).strip().lower() == "yes" else "no"
        reason = str(reasons.get(str(index + 1), "")).strip()
        if verdict == "no" and not reason:
            reason = NO_REASON
        verdicts.append(JudgeVerdict(verdict=verdict, reason=reason, raw_response=response))
    return verdicts


def _parse_verdict(response: str) -> JudgeVerdict:
    """Parse the explainer's response. Tolerant of extra whitespace and code fences."""
    cleaned = response.strip()
    # Strip markdown code fences if the model ignored instructions
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    # Parse the first JSON object and ignore anything around it. Some models
    # ignore "respond ONLY with JSON" and wrap the verdict in prose (e.g. a trailing
    # "Let me know if you'd like to evaluate further turns!"); raw_decode from the
    # first '{' parses the object and stops, leaving the trailing text out.
    start = cleaned.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
            verdict = str(obj.get("verdict", "")).strip().lower()
            if verdict not in ("yes", "no", "continue"):
                verdict = "no"
            reason = str(obj.get("reason", "")).strip()
            return JudgeVerdict(
                verdict=verdict,
                reason=reason or NO_REASON,
                raw_response=response,
            )
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: scan for a verdict keyword in the raw text.
    lowered = cleaned.lower()
    if "continue" in lowered:
        return JudgeVerdict(
            verdict="continue", reason="(unstructured continue)", raw_response=response
        )
    if "yes" in lowered and "no" not in lowered:
        return JudgeVerdict(verdict="yes", reason="(unstructured yes)", raw_response=response)
    if "no" in lowered and "yes" not in lowered:
        return JudgeVerdict(verdict="no", reason="(unstructured no)", raw_response=response)
    return JudgeVerdict(
        verdict="no",
        reason=f"could not parse explainer response: {response!r}",
        raw_response=response,
    )

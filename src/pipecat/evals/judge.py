#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The judge: decides whether the bot did what a scenario asks.

A judge answers a natural-language criterion, an ``eval:`` on a scripted
turn or a simulation's ``success:`` and metrics, about the conversation it
was given. It keeps that conversation, fed as it happens: the user's turns,
the bot's replies, and the tool calls the bot makes. A terse reply ("That's
four") is judged in context, and a whole run is judged over the same record.

How it decides:

Every question goes to a :class:`~pipecat.classifiers.base_classifier.BaseClassifier`,
which answers with an option and a probability for each of the options it
was offered. Two classifiers come with Pipecat:

- :class:`~pipecat.classifiers.llm.classifier.LLMClassifier`, over any
  Pipecat LLM service with ``run_inference()``: OpenAI, Ollama, Together,
  and others. This is the default.
- :class:`~pipecat.classifiers.jev.classifier.JevClassifier`, over TypeSafe's
  Jev, a hosted classification model. Jev answers in a few hundred
  milliseconds, costs little, and its probabilities are calibrated, so a
  scenario can tell a sure verdict from a close call.

A scenario picks one in its ``judge.eval:`` block. All the options::

    judge:
      eval:
        service: typesafe        # or ollama (the default), openai, ...
        model: jev-latest        # the model the service answers with
        endpoint: https://...    # optional; the API's base URL
        explainer:               # optional; the LLM that gives reasons
          service: ollama        #   (the judging LLM if omitted, and for
          model: gemma4:12b      #   service: typesafe the default LLM judge;
                                 #   none if set to false)
        explain_below: 0.75      # optional; see "The explainer" below
        allow_continue: true     # optional; false judges a reply yes or no only

``service: typesafe`` takes its API key from ``TYPESAFE_API_KEY``.

What the judge asks:

The harness asks three kinds of question, and each is one classifier call:

- A reply, for an ``eval:`` on a scripted turn. The state holds the
  conversation so far and, separately, the bot's latest reply. The answers
  are ``yes`` (the reply meets the criterion), ``no`` (it's a real answer
  that doesn't, including a reply that waits for the user instead), and
  ``continue`` (the bot is still working toward its answer: it only
  greeted, said it's checking, or the reply is still arriving). On
  ``continue`` the harness waits for more of the reply and asks again. A
  suite whose judged replies are all final answers, with nothing for the bot
  to fetch first, sets ``allow_continue: false``, and a reply is then yes or
  no; a reply that is still arriving is still judged again as more of it
  comes.
- A function call, for an ``eval:`` on a ``function_call``. The state holds
  the call's name and arguments and the conversation as context. The
  answer is yes or no.
- A whole simulation, at the end of the run. The goal is one yes/no
  question over the whole conversation, tool calls included. Each bot turn
  is asked about on its own, once per criterion, with only the conversation
  before that turn as context. The answers are ``meets``, ``fails``, and
  ``not_applicable`` (the criterion only covers some situation, like "when
  the time is taken, apologise", and this turn isn't in it); only ``fails``
  fails the turn. All of a run's questions go out at once.

How each question is written:

- The reply or function call being judged is sent separately from the
  conversation before it (``latest_bot_reply`` or ``call``), so the
  classifier knows which part to judge.
- Every possible answer is one of the options. "The bot hasn't answered
  yet" is the option ``continue``, and "this criterion doesn't apply to this
  turn" is the option ``not_applicable``. Don't write these as rules in the
  instructions instead: a classifier follows an instruction for every
  criterion, even where it doesn't fit. A rule like "a criterion about some
  situation passes when that situation doesn't come up" also passes turns
  that never state a price against "the reply states a price".
- The instructions hold only the question and the criterion.

How answers become verdicts:

Each verdict carries a confidence from 0 to 1, and its reason starts out as
the probabilities the classifier gave (for example ``P(yes)=0.97``):

- A reply's verdict is the option chosen, with the confidence in it.
- A yes/no question (a function call, a goal) is ``yes`` when the
  probability of yes is at least 0.5; its confidence is the probability of
  the answer given.
- A turn's confidence is the probability of the verdict given: for a pass,
  the probability of ``meets`` and ``not_applicable`` together.

Each question's verdict is cached by its criterion and state, so asking the
same question about the same conversation twice costs one call.

The explainer:

A classifier gives no reasons, so the explainer
(:class:`~pipecat.evals.explainer.EvalExplainer`), an LLM reading the same
conversation, supplies them. It's asked for:

- every ``no``, on a reply, a function call, a turn or a goal;
- every verdict the classifier is unsure of, meaning a confidence below
  ``explain_below`` (0.75 by default). When a verdict is close to 50/50, the
  next run can go the other way, so these are worth a second look. An LLM
  classifier's probabilities are not calibrated and rarely fall this low, so
  in practice it is the ``no`` verdicts that are explained;
- never a ``continue``, which is only a request to wait for more text.

``explain_below: 0`` limits it to the ``no`` verdicts, and a value above 1
sends it every verdict (useful for comparing a classifier against the LLM).

The explainer is asked the same question and makes its own judgement, which
costs a full LLM call. For a simulation, it judges the whole run in one call
if any verdict needs a reason, and its reasons go only to the verdicts that
need one. Either way, the explainer never changes a verdict: the
classifier's stands. If the explainer agrees, its reason is used, followed
by the probabilities. If it disagrees, the reason says so ("the explainer
judged yes: ...").

When a call fails:

A question that times out, can't connect, or comes back unusable is asked
once more. A second failure fails the question: a reply or a function call
gets a ``no`` with the reason "judge call failed", and each verdict of a
failed simulation question is a ``none`` (the verdict the harness reports as
not given).

Lifecycle:

The judge sets its classifier up when it is created and cleans it up when
the session closes it at the end of the run. For Jev that opens one HTTP/2
connection ahead of the first question and keeps it open, so a question
doesn't wait for a connection and a simulation's questions share one.

Example::

    from pipecat.classifiers.llm.classifier import LLMClassifier
    from pipecat.services.ollama.llm import OLLamaLLMService

    service = OLLamaLLMService(settings=OLLamaLLMService.Settings(model="gemma4:12b"))
    judge = EvalJudge(LLMClassifier(llm=service), explainer=EvalExplainer(service))
    judge.add_user_message("What can you help me with?")
    judge.add_assistant_message("I can answer questions, set reminders, and look things up.")
    verdict = await judge.evaluate("describes the bot's capabilities")
    if not verdict.passed:
        print(f"judge said no: {verdict.reason}")

Attributes:
    JUDGE_SYSTEM_INSTRUCTION: The explainer's instructions for a reply.

        .. deprecated:: 1.12.0
            Moved to :mod:`pipecat.evals.explainer`. Will be removed in 2.0.0.

    JUDGE_ASK_TEMPLATE: The explainer's ask for a reply.

        .. deprecated:: 1.12.0
            Moved to :mod:`pipecat.evals.explainer`. Will be removed in 2.0.0.

    JUDGE_CALL_SYSTEM_INSTRUCTION: The explainer's instructions for a function call.

        .. deprecated:: 1.12.0
            Moved to :mod:`pipecat.evals.explainer`. Will be removed in 2.0.0.

    JUDGE_CALL_ASK_TEMPLATE: The explainer's ask for a function call.

        .. deprecated:: 1.12.0
            Moved to :mod:`pipecat.evals.explainer`. Will be removed in 2.0.0.

    RUN_JUDGE_SYSTEM_INSTRUCTION: The explainer's instructions for a whole run.

        .. deprecated:: 1.12.0
            Moved to :mod:`pipecat.evals.explainer`. Will be removed in 2.0.0.

    RUN_JUDGE_ASK_TEMPLATE: The explainer's ask for a whole run.

        .. deprecated:: 1.12.0
            Moved to :mod:`pipecat.evals.explainer`. Will be removed in 2.0.0.
"""

import asyncio
import json
import os
import warnings
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, TypeVar, cast

from loguru import logger

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.llm.classifier import LLMClassifier
from pipecat.evals import explainer
from pipecat.evals.explainer import EvalExplainer
from pipecat.evals.services import DEFAULT_JEV_MODEL, llm_service_from_config
from pipecat.evals.verdicts import NO_REASON, JudgeVerdict, RunVerdicts, cache_key
from pipecat.services.llm_service import LLMService
from pipecat.utils.asyncio.task_manager import TaskManager

# Seconds to wait for Jev to answer a question. Jev answers in a few hundred
# milliseconds, so a question still waiting this long is one to ask again.
JEV_TIMEOUT = 2.5

_R = TypeVar("_R")

_TRANSCRIPTION_NOTE = (
    "The bot's text may be an automatic speech-to-text transcription: judge its intended "
    "spoken meaning, never its spelling ('for' may mean 'four', 'to' may mean 'two')."
)

_REPLY_OUTCOMES = {
    "yes": "The bot has given its answer, and the answer satisfies the criterion.",
    "no": (
        "The bot has given its answer, and the answer does not satisfy the criterion. A reply "
        "that waits for the user (asking them to take their time or to go on) instead of "
        "giving what the criterion asks for is a no."
    ),
    "continue": (
        "The bot is still working toward its answer: it only greets, says it is checking or "
        "looking something up and will report back, or the reply is an obviously incomplete "
        "fragment."
    ),
}

# A run's per-turn outcomes. A classifier can't tell by itself whether a
# criterion only applies in some situation ("when the time is taken,
# apologises"), so "doesn't apply" is an outcome of its own rather than a rule
# in the instructions.
_TURN_OUTCOMES = {
    "meets": "The reply does what the criterion asks.",
    "fails": "The criterion applies to this reply, and the reply does not do what it asks.",
    "not_applicable": (
        "The criterion only asks something of replies in a particular situation (it says "
        "'when', 'if', or similar), and that situation does not arise in this reply."
    ),
}


class EvalJudge:
    """Judges a conversation with a classifier, and asks an explainer for the reasons.

    Args:
        classifier: The classifier that decides the verdicts. The judge sets
            it up and cleans it up with itself.
        explainer: The LLM asked for the reason behind a ``no`` or an unsure
            verdict; ``None`` reports the probabilities alone.
        explain_below: A ``yes`` less sure than this is explained too.
        allow_continue: Whether a reply may be judged ``continue`` (the bot is
            still working toward its answer, so the harness waits for more of
            it). A suite whose judged replies are all final answers turns it
            off, and a reply is then ``yes`` or ``no``.
    """

    def __init__(
        self,
        classifier: BaseClassifier | LLMService[Any] | None = None,
        *,
        explainer: EvalExplainer | None = None,
        explain_below: float = 0.75,
        allow_continue: bool = True,
        max_tokens: int | None = None,
        service: LLMService[Any] | None = None,
    ):
        """Initialize the judge with an empty conversation.

        Args:
            classifier: The classifier that decides the verdicts. An LLM
                service is also accepted, and classifies and explains with
                itself.

                .. deprecated:: 1.12.0
                    Use :class:`~pipecat.classifiers.llm.classifier.LLMClassifier`
                    over the service, with an
                    :class:`~pipecat.evals.explainer.EvalExplainer` over it for
                    the reasons. Will be removed in 2.0.0.

            explainer: The LLM asked for the reason behind a verdict.
            explain_below: A ``yes`` less sure than this is explained too.
            allow_continue: Whether a reply may be judged ``continue``.
            max_tokens: Cap on the response length of the explainer the judge
                makes for an LLM service.

                .. deprecated:: 1.12.0
                    Pass ``max_tokens`` to
                    :class:`~pipecat.evals.explainer.EvalExplainer` instead.
                    Will be removed in 2.0.0.

            service: The LLM service that classifies and explains, in place of
                ``classifier``.

                .. deprecated:: 1.12.0
                    Use :class:`~pipecat.classifiers.llm.classifier.LLMClassifier`
                    over the service as the ``classifier``, with an
                    :class:`~pipecat.evals.explainer.EvalExplainer` over it for
                    the reasons. Will be removed in 2.0.0.

        Raises:
            ValueError: If no classifier is given.
        """
        classifier = classifier if classifier is not None else service
        if classifier is None:
            raise ValueError("EvalJudge needs a classifier to decide the verdicts")
        if max_tokens is not None:
            warnings.warn(
                "`max_tokens` parameter of `EvalJudge` is deprecated since 1.12.0 and will be "
                "removed in 2.0.0. Pass `max_tokens` to `EvalExplainer` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not isinstance(classifier, BaseClassifier):
            warnings.warn(
                "Passing an LLM service to `EvalJudge` is deprecated since 1.12.0 and will be "
                "removed in 2.0.0. Pass `LLMClassifier(llm=service)`, with an "
                "`EvalExplainer(service)` for the reasons.",
                DeprecationWarning,
                stacklevel=2,
            )
            service = cast("LLMService[Any]", classifier)
            classifier = LLMClassifier(llm=service)
            if explainer is None:
                explainer = EvalExplainer(
                    service, max_tokens=max_tokens or 200, allow_continue=allow_continue
                )
        self._classifier = classifier
        self._explainer = explainer
        self._explain_below = explain_below
        self._allow_continue = allow_continue
        self._reply_outcomes = (
            _REPLY_OUTCOMES
            if allow_continue
            else {k: v for k, v in _REPLY_OUTCOMES.items() if k != "continue"}
        )
        # The conversation the judge evaluates against, grown by the harness over
        # the scenario: dicts with a ``role`` of ``user``, ``assistant`` (a
        # segment of a reply), or ``tool`` (a call the bot made, one line), and
        # the ``content``.
        self._transcript: list[dict] = []
        self._cache: dict[str, JudgeVerdict] = {}
        self._run_cache: dict[str, RunVerdicts] = {}
        # A client the judge built for its classifier, which the classifier
        # itself does not own, so the judge is the one to close it.
        self._own_client: Any = None
        self._setup_task: asyncio.Task | None = None
        try:
            self._setup_task = asyncio.get_running_loop().create_task(self._setup())
        except RuntimeError:
            pass  # no running loop: the first question sets the classifier up

    @classmethod
    def from_config(cls, judge_config: dict | None) -> "EvalJudge":
        """Build a judge from a scenario's ``judge.eval:`` block.

        ``service: typesafe`` judges with Jev and explains with the LLM the
        ``explainer:`` block names, the default LLM when it names none.
        Any other block judges and explains with one LLM: a ``factory`` (a
        dotted path to a callable taking the config) builds the service,
        otherwise the ``service`` name picks a provider, ``ollama`` by
        default. For a fully custom judge, construct ``EvalJudge`` directly
        and pass it to the session.

        Args:
            judge_config: Mapping with keys ``service`` (default ``"ollama"``),
                ``model`` (default ``"gemma4:12b"``), optional ``endpoint``
                (service-specific default if omitted), an optional ``extra``
                mapping forwarded to the model as top-level request parameters,
                an optional ``explainer`` block (``false`` for verdicts without
                reasons), an optional ``explain_below``, and an optional
                ``allow_continue`` (``false`` judges a reply yes or no only).
                ``None`` uses all defaults.

        Returns:
            A configured EvalJudge.

        Raises:
            ValueError: If ``service`` is unknown (matching
                :func:`pipecat.evals.services.tts_service_from_config` and
                :func:`pipecat.evals.services.stt_service_from_config`), or if
                ``service: typesafe`` has no API key.

        Example::

            # In the scenario: judge.eval.factory: "my_pkg.make_judge_llm"
            def make_judge_llm(config):
                return TogetherLLMService(...)  # any service exposing run_inference()
        """
        config = judge_config or {}
        allow_continue = config.get("allow_continue", True) is not False
        explain_below = float(config.get("explain_below", 0.75))
        explainer_config = config.get("explainer")
        # The explainer is asked the same questions, so it follows the same rule.
        explains = explainer_config is not False
        explainer = None
        client = None
        if str(config.get("service", "")).lower() == "typesafe":
            classifier, client = _jev_classifier(config)
            if explains:
                explainer = EvalExplainer.from_config(
                    explainer_config, allow_continue=allow_continue
                )
        else:
            service = llm_service_from_config(judge_config, where="judge.eval")
            classifier = LLMClassifier(llm=service)
            if explains and explainer_config is None:
                # Without a block of its own, the explainer is the classifying LLM.
                explainer = EvalExplainer(service, allow_continue=allow_continue)
            elif explains:
                explainer = EvalExplainer.from_config(
                    explainer_config, allow_continue=allow_continue
                )
        judge = cls(
            classifier,
            explainer=explainer,
            explain_below=explain_below,
            allow_continue=allow_continue,
        )
        # A Jev classifier is given a client it does not own, so the judge closes it.
        judge._own_client = client
        return judge

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that decides the verdicts."""
        return self._classifier

    def add_user_message(self, text: str | None) -> None:
        """Record a user turn, so a later reply is judged in context.

        Args:
            text: The user's utterance, or ``None`` for a bot-first turn (ignored).
        """
        if text and text.strip():
            self._transcript.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str | None) -> None:
        """Add a segment of the bot's current reply to the conversation the judge sees.

        Consecutive segments are one reply: a judged run counts them as one
        bot turn.

        Args:
            text: The new reply segment; empty or ``None`` is ignored.
        """
        if text and text.strip():
            self._transcript.append({"role": "assistant", "content": text})

    def add_tool_call(self, text: str | None) -> None:
        """Record a tool call the bot made, as evidence for a judged run.

        Args:
            text: The call on one line, e.g. ``book({"time": "6pm"})`` or
                ``book was cancelled``; empty or ``None`` is ignored.
        """
        if text and text.strip():
            self._transcript.append({"role": "tool", "content": text})

    async def evaluate(self, criterion: str) -> JudgeVerdict:
        """Judge whether the bot's latest reply satisfies ``criterion``, in the conversation so far.

        Args:
            criterion: Natural-language description of what the reply should express.

        Returns:
            A ``yes``, a ``no``, or, when ``allow_continue`` is on, a
            ``continue`` when the bot is still working toward its answer,
            cached by criterion and conversation. A final verdict that needs a
            reason is explained, and a failed call is a ``no`` with the failure
            as its reason.
        """
        entries = _numbered_turns(e for e in self._transcript if e["role"] != "tool")
        latest = entries.pop()["content"] if entries and entries[-1]["role"] == "bot" else ""
        state = {"conversation": _conversation(entries), "latest_bot_reply": latest}
        key = cache_key("reply", criterion, state)
        if key not in self._cache:
            answers = await self._choices(
                state,
                {
                    "verdict": (
                        "Does `latest_bot_reply`, the bot's most recent reply, following "
                        f"`conversation`, satisfy this criterion? Criterion: "
                        f"{_sentence(criterion)} {_TRANSCRIPTION_NOTE}",
                        self._reply_outcomes,
                    )
                },
            )
            if answers is None:
                verdict = _failed("no")
            else:
                verdict = _reply_verdict(answers["verdict"])
                if verdict.verdict != "continue":
                    verdict = await self._explain(
                        verdict, lambda e: e.explain(self._transcript, criterion)
                    )
            self._cache[key] = verdict
        return self._cache[key]

    async def evaluate_call(self, name: str, args: dict | None, criterion: str) -> JudgeVerdict:
        """Judge whether a function call the bot made satisfies ``criterion``, in the conversation so far.

        Args:
            name: The function's name.
            args: The call's arguments.
            criterion: Natural-language description of what the call should be.

        Returns:
            A ``yes`` or a ``no``, cached by call, criterion and conversation,
            and explained when it needs a reason.
        """
        entries = _numbered_turns(e for e in self._transcript if e["role"] != "tool")
        state = {
            "conversation": _conversation(entries),
            "call": {"name": name, "arguments": args or {}},
        }
        key = cache_key("call", criterion, state)
        if key not in self._cache:
            answer = await self._yes_no(
                state,
                "Does the bot's function `call`, judged by its name and arguments, satisfy "
                f"this criterion? Criterion: {_sentence(criterion)} `conversation` is context "
                f"only. {_TRANSCRIPTION_NOTE}",
            )
            if answer is None:
                verdict = _failed("no")
            else:
                verdict = await self._explain(
                    _yes_no_verdict(answer),
                    lambda e: e.explain_call(self._transcript, name, args, criterion),
                )
            self._cache[key] = verdict
        return self._cache[key]

    async def evaluate_run(
        self,
        criteria: dict[str, str],
        success: str,
        transcript: Sequence[dict] | None = None,
    ) -> RunVerdicts:
        """Judge the whole conversation: every bot turn on every criterion, and the goal.

        The goal is asked over the whole conversation. Each bot turn is its
        own call, judged in the light of the conversation before it, with the
        turn itself as ``latest_bot_reply``; the calls run concurrently. When
        any verdict needs a reason, the explainer judges the run once and its
        reasons go with the classifier's verdicts.

        Args:
            criteria: The per-turn criteria to decide, by name.
            success: The goal criterion, decided over the whole conversation.
            transcript: A conversation to judge in place of the one the judge
                kept: dicts with a ``role`` of ``user``, ``assistant`` or
                ``tool`` and the ``content``. Also accepted first, before
                ``criteria`` and ``success``.

                .. deprecated:: 1.11.0
                    Feed the judge with :meth:`add_user_message`,
                    :meth:`add_assistant_message` and :meth:`add_tool_call`
                    instead. Will be removed in 2.0.0.

        Returns:
            The goal's verdict and, per criterion, a verdict per bot turn in
            order. A failed call is a ``none`` for each verdict it asked for.
        """
        if not isinstance(criteria, dict):
            # The (transcript, criteria, success) order of the deprecated form.
            transcript, criteria, success = (
                cast("Sequence[dict]", criteria),
                cast("dict[str, str]", success),
                cast("str", transcript),
            )
        if transcript is not None:
            warnings.warn(
                "`transcript` parameter of `EvalJudge.evaluate_run` is deprecated since 1.11.0 "
                "and will be removed in 2.0.0. Feed the judge with `add_user_message`, "
                "`add_assistant_message` and `add_tool_call` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        conversation = list(self._transcript if transcript is None else transcript)
        entries = _numbered_turns(conversation)
        names = list(criteria)
        key = cache_key("run", criteria, success, entries)
        if key in self._run_cache:
            return self._run_cache[key]

        goal_instructions = (
            f"Does the conversation as a whole achieve this goal? Goal: {_sentence(success)} "
            "A `tool` entry is a function the bot called at that point; a completed "
            "call is stronger evidence of an action than the bot saying it did it. "
            f"{_TRANSCRIPTION_NOTE}"
        )
        turn_questions = {
            name: (
                "`latest_bot_reply` is the bot's reply following `conversation`. How does it "
                f"stand against this criterion? Criterion: {_sentence(criteria[name])} "
                f"{_TRANSCRIPTION_NOTE}",
                _TURN_OUTCOMES,
            )
            for name in names
        }
        turn_states = [
            {"conversation": _conversation(entries[:i]), "latest_bot_reply": entry["content"]}
            for i, entry in enumerate(entries)
            if entry["role"] == "bot"
        ]

        # The goal, and every criterion on each turn: one call each, all at once.
        goal_answer, turn_answers = await asyncio.gather(
            self._yes_no({"conversation": _conversation(entries)}, goal_instructions),
            asyncio.gather(
                *(
                    self._choices(state, turn_questions)
                    for state in (turn_states if turn_questions else [])
                )
            ),
        )

        failed = _failed("none")
        verdicts = RunVerdicts(
            goal=_yes_no_verdict(goal_answer) if goal_answer else failed,
            turns={
                name: [
                    _turn_verdict(answers[name]) if answers else failed for answers in turn_answers
                ]
                for name in names
            },
        )
        verdicts = await self._explain_run(verdicts, conversation, criteria, success)
        self._run_cache[key] = verdicts
        return verdicts

    async def close(self) -> None:
        """Release what the judge holds open; the session calls it when the run ends."""
        if self._setup_task is not None and not self._setup_task.done():
            self._setup_task.cancel()
        await self._classifier.cleanup()
        if self._own_client is not None:
            await self._own_client.close()

    async def _setup(self) -> None:
        """Wire the classifier up, ahead of the first question.

        The judge runs outside any pipeline, so it gives the classifier a task
        manager of its own. A classifier that cannot be set up now is only a
        warning: the first question goes out regardless.
        """
        try:
            await self._classifier.setup(TaskManager())
        except Exception as e:
            logger.debug(f"Judge couldn't set its classifier up early: {e}")

    def _needs_reason(self, verdict: JudgeVerdict) -> bool:
        """Whether a verdict is worth explaining: a ``no``, or an unsure ``yes``."""
        unsure = verdict.confidence is not None and verdict.confidence < self._explain_below
        return verdict.verdict == "no" or unsure

    async def _explain(
        self,
        verdict: JudgeVerdict,
        ask: Callable[[EvalExplainer], Awaitable[JudgeVerdict]],
    ) -> JudgeVerdict:
        """The verdict with the explainer's reason for it, when it needs one and there is an explainer."""
        if self._explainer is None or not self._needs_reason(verdict):
            return verdict
        return _explained(verdict, await ask(self._explainer))

    async def _explain_run(
        self,
        verdicts: RunVerdicts,
        transcript: Sequence[dict],
        criteria: dict[str, str],
        success: str,
    ) -> RunVerdicts:
        """The run's verdicts, those that need a reason given the explainer's; it judges the run once."""
        everything = [verdicts.goal, *(v for vs in verdicts.turns.values() for v in vs)]
        if self._explainer is None or not any(self._needs_reason(v) for v in everything):
            return verdicts
        explanation = await self._explainer.explain_run(transcript, criteria, success)

        def explain(verdict: JudgeVerdict, reasoned: JudgeVerdict | None) -> JudgeVerdict:
            if reasoned is None or reasoned.verdict == "none" or not self._needs_reason(verdict):
                return verdict
            return _explained(verdict, reasoned)

        turns: dict[str, list[JudgeVerdict]] = {}
        for name, turn_verdicts in verdicts.turns.items():
            reasoned = explanation.turns.get(name, [])
            turns[name] = [
                explain(verdict, reasoned[i] if i < len(reasoned) else None)
                for i, verdict in enumerate(turn_verdicts)
            ]
        return RunVerdicts(goal=explain(verdicts.goal, explanation.goal), turns=turns)

    async def _yes_no(self, state, instructions: str) -> YesNoResult | None:
        """The probability of yes, or ``None`` when the question failed."""
        answers = await self._ask(
            state,
            lambda: self._classifier.yes_no(
                state, {"answer": YesNoQuestion(instructions=instructions)}
            ),
        )
        return answers["answer"] if answers else None

    async def _choices(
        self, state, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceResult] | None:
        """The option chosen for each question, or ``None`` when they failed.

        The questions share the state, so they go in one call.
        """
        return await self._ask(
            state,
            lambda: self._classifier.choice(
                state,
                {
                    name: ChoiceQuestion(instructions=instructions, options=dict(options))
                    for name, (instructions, options) in questions.items()
                },
            ),
        )

    async def _ask(self, state, ask: Callable[[], Awaitable[_R]]) -> "_R | None":
        """The classifier's answer, asked once more if it failed, or ``None``."""
        if self._setup_task is None:
            self._setup_task = asyncio.get_running_loop().create_task(self._setup())
        if not self._setup_task.done():
            await asyncio.shield(self._setup_task)
        logger.debug(f"Judge asking over state:\n{json.dumps(state)}")
        for attempt in (1, 2):
            try:
                return await ask()
            except ClassifierError as e:
                if attempt == 1:
                    logger.warning(f"Judge question failed, asking again: {e}")
                else:
                    logger.error(f"Judge question failed: {e}")
        return None


def _jev_classifier(config: dict) -> tuple["BaseClassifier", Any]:
    """A classifier over Jev, and the client it was given, which it does not own.

    Args:
        config: The ``judge.eval:`` block, for the ``model`` and the
            ``endpoint`` the questions go to.

    Returns:
        The classifier and its client.

    Raises:
        ValueError: If there is no ``TYPESAFE_API_KEY``.
    """
    from pipecat.classifiers.jev.classifier import JevClassifier
    from pipecat.classifiers.jev.client import JevClient

    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        raise ValueError("Judging with Jev needs an API key: set TYPESAFE_API_KEY.")
    endpoint = config.get("endpoint")
    # Only a block that names an endpoint overrides the client's own default.
    client_args: dict[str, Any] = {"base_url": str(endpoint).rstrip("/")} if endpoint else {}
    client = JevClient(
        api_key=api_key,
        model=config.get("model") or DEFAULT_JEV_MODEL,
        timeout=JEV_TIMEOUT,
        **client_args,
    )
    return JevClassifier(client=client), client


def _numbered_turns(transcript: Iterable[dict]) -> list[dict]:
    """The conversation with each bot reply joined into one numbered turn.

    A bot turn is a run of reply segments with nothing else between them.

    Args:
        transcript: The judge's conversation entries.

    Returns:
        The entries, each a dict with a ``role`` of ``user``, ``bot`` or
        ``tool`` and the ``content``, a bot entry also carrying its ``turn``
        (from 1).
    """
    entries: list[dict] = []
    turn = 0
    for entry in transcript:
        if entry["role"] == "assistant":
            if entries and entries[-1]["role"] == "bot":
                entries[-1]["content"] += f" {entry['content']}"
                continue
            turn += 1
            entries.append({"role": "bot", "turn": turn, "content": entry["content"]})
        else:
            entries.append({"role": entry["role"], "content": entry["content"]})
    return entries


def _conversation(entries: list[dict]) -> list[dict]:
    """The conversation as the state carries it: a speaker, the text, and a bot turn's number."""
    return [
        {"speaker": e["role"], **({"turn": e["turn"]} if "turn" in e else {}), "text": e["content"]}
        for e in entries
    ]


def _reply_verdict(answer: ChoiceResult) -> JudgeVerdict:
    """A reply's verdict: the choice of ``yes``, ``no`` or ``continue``."""
    return JudgeVerdict(
        verdict=answer.choice,
        reason=_probabilities(answer.probabilities),
        raw_response=answer.model_dump_json(),
        confidence=answer.confidence,
    )


def _yes_no_verdict(answer: YesNoResult) -> JudgeVerdict:
    """A yes/no verdict from the probability of yes."""
    probability = answer.probability
    return JudgeVerdict(
        verdict="yes" if answer.is_yes else "no",
        reason=f"P(yes)={probability:.2f}",
        raw_response=answer.model_dump_json(),
        confidence=max(probability, 1 - probability),
    )


def _turn_verdict(answer: ChoiceResult) -> JudgeVerdict:
    """A turn's verdict: a ``no`` only when the criterion applies to the turn and fails.

    Its confidence is the probability of the verdict given: for a pass, the
    probability of either outcome that passes.
    """
    fails = answer.probabilities.get("fails", 0.0)
    passed = answer.choice != "fails"
    return JudgeVerdict(
        verdict="yes" if passed else "no",
        reason=_probabilities(answer.probabilities),
        raw_response=answer.model_dump_json(),
        confidence=1 - fails if passed else fails,
    )


def _sentence(text: str) -> str:
    """``text`` ending in punctuation, so the instruction after it reads as a new sentence."""
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else f"{text}."


def _probabilities(probabilities: dict[str, float]) -> str:
    """The probability of each outcome, as a verdict's reason."""
    return ", ".join(f"P({k})={v:.2f}" for k, v in probabilities.items())


def _failed(verdict: str) -> JudgeVerdict:
    """The verdict a failed call gives."""
    return JudgeVerdict(verdict=verdict, reason="judge call failed", raw_response="")


def _explained(verdict: JudgeVerdict, explanation: JudgeVerdict) -> JudgeVerdict:
    """The verdict with the explainer's reason, noting when the explainer disagreed."""
    if explanation.verdict == verdict.verdict:
        given = explanation.reason and explanation.reason != NO_REASON
        reason = f"{explanation.reason} ({verdict.reason})" if given else verdict.reason
    else:
        reason = (
            f"{verdict.reason}; the explainer judged {explanation.verdict}: {explanation.reason}"
        )
    return JudgeVerdict(
        verdict=verdict.verdict,
        reason=reason,
        raw_response=verdict.raw_response,
        confidence=verdict.confidence,
    )


# The prompts these named are the explainer's now, and are read from there for
# as long as this module answers for them.
_MOVED_TO_EXPLAINER = (
    "JUDGE_SYSTEM_INSTRUCTION",
    "JUDGE_ASK_TEMPLATE",
    "JUDGE_CALL_SYSTEM_INSTRUCTION",
    "JUDGE_CALL_ASK_TEMPLATE",
    "RUN_JUDGE_SYSTEM_INSTRUCTION",
    "RUN_JUDGE_ASK_TEMPLATE",
)


def __getattr__(name: str):
    """Answer for a prompt constant at its former path in this module."""
    if name in _MOVED_TO_EXPLAINER:
        warnings.warn(
            f"`pipecat.evals.judge.{name}` is deprecated since 1.12.0 and will be removed in "
            f"2.0.0. Use `pipecat.evals.explainer.{name}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(explainer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

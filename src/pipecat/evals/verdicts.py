#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""What a judge decides: a verdict per question, and a whole run's verdicts.

:class:`~pipecat.evals.judge.EvalJudge` answers with these, and the
explainer (:class:`~pipecat.evals.explainer.EvalExplainer`) fills in the
reasons behind them.
"""

import hashlib
import json
from dataclasses import dataclass

#: The reason a verdict carries when the judge gave the verdict without one.
NO_REASON = "(no reason given)"

#: The reason a verdict carries when the judge gave no verdict at all.
NO_VERDICT = "(judge gave no verdict)"


@dataclass
class JudgeVerdict:
    """Outcome of a single judge call.

    Parameters:
        verdict: ``"yes"`` (satisfies), ``"no"`` (substantive answer that fails),
            or ``"continue"`` (interim/filler/incomplete — re-judge once more text
            arrives).
        reason: One-sentence justification.
        raw_response: The judge's raw answer, for diagnostics.
        confidence: How sure the judge is of the verdict, from 0 to 1.
    """

    verdict: str
    reason: str
    raw_response: str
    confidence: float | None = None

    @property
    def passed(self) -> bool:
        """True only when the verdict is a definite ``"yes"``."""
        return self.verdict == "yes"


@dataclass
class RunVerdicts:
    """A whole simulation run's verdicts.

    Parameters:
        goal: The verdict on the goal, over the whole conversation.
        turns: Per criterion name, a verdict per bot turn, in order.
    """

    goal: JudgeVerdict
    turns: dict[str, list[JudgeVerdict]]


def cache_key(*parts) -> str:
    """Hash what a question was about, so asking it again is free.

    Args:
        *parts: Everything that decides the answer: the kind of question, the
            criterion, and the conversation it is asked over.

    Returns:
        A key for the verdict cache.
    """
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()

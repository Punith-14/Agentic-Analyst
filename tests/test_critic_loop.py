"""The critic path: does it stop runs, and can it take one down?

No trained model is loaded here. These use fake scorers so the tests run in
CI without lightgbm, torch or a 600 MB download — what's being checked is the
loop's contract with the critic, not the model's accuracy. Accuracy lives in
notebooks 02 and 05, and feature parity in test_critic_infer.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent._stubs import TOOL_SPECS, TOOLS, set_db
from agent.llm import scripted
from agent.loop import AgentConfig, run_agent


@pytest.fixture(autouse=True)
def use_chinook():
    set_db("chinook_1")


class FakeCritic:
    """Returns a fixed score. version is read by the loop for provenance."""

    version = "fake-v1"

    def __init__(self, score):
        self.score_value = score
        self.calls = 0

    def score(self, question, steps):
        self.calls += 1
        return self.score_value


class BrokenCritic:
    version = "broken"

    def score(self, question, steps):
        raise RuntimeError("the critic exploded")


def go(scenario="keeps_failing", critic=None, **kw):
    cfg = AgentConfig(max_steps=6, verbose=False, critic=critic, **kw)
    return run_agent(
        question="How many albums are there?",
        llm=scripted(scenario), tools=TOOLS, tool_specs=TOOL_SPECS,
        table_names=["Album", "Artist", "Track"], config=cfg,
    )


# ==========================================================================
# it stops runs
# ==========================================================================


def test_high_score_stops_the_run():
    rec = go(critic=FakeCritic(0.99))
    assert rec.termination == "critic_stop"
    assert len(rec.steps) == 1, "should stop at the first scored step"


def test_low_score_does_not_stop():
    rec = go(critic=FakeCritic(0.01))
    assert rec.termination != "critic_stop"


def test_threshold_is_respected():
    """0.694 is a chosen operating point, not a constant baked into the loop."""
    assert go(critic=FakeCritic(0.5), critic_threshold=0.4).termination == "critic_stop"
    assert go(critic=FakeCritic(0.5), critic_threshold=0.9).termination != "critic_stop"


def test_min_step_delays_stopping():
    """Raising critic_min_step buys the agent room to recover."""
    late = go(critic=FakeCritic(0.99), critic_min_step=2)
    if late.termination == "critic_stop":
        assert len(late.steps) >= 3, "stopped earlier than critic_min_step allows"


def test_score_is_recorded_on_the_step():
    """Needed to analyse the critic's behaviour from the logs afterwards."""
    rec = go(critic=FakeCritic(0.99))
    assert rec.steps[-1].critic_score == pytest.approx(0.99)


def test_provenance_records_the_critic():
    """Runs shortened by a critic must be filterable out of future training
    sets — otherwise the next critic learns from this one's decisions."""
    rec = go(critic=FakeCritic(0.99))
    assert rec.critic_version == "fake-v1"
    assert "critic@" in rec.context_policy


# ==========================================================================
# it cannot take the run down
# ==========================================================================


def test_a_crashing_critic_does_not_kill_the_run():
    """A broken critic is worse than no critic if it loses the trajectory."""
    rec = go("clean_success", critic=BrokenCritic())
    assert rec.termination == "final_answer"
    assert rec.final_answer is not None


def test_none_score_means_do_not_stop():
    """`None` is the critic saying "no opinion" — it must never be treated as
    a high score by accident."""
    rec = go("clean_success", critic=FakeCritic(None))
    assert rec.termination == "final_answer"


def test_no_critic_is_unchanged_behaviour():
    with_none = go("clean_success", critic=None)
    assert with_none.termination == "final_answer"
    assert with_none.critic_version is None
    assert "critic@" not in with_none.context_policy
    assert all(s.critic_score is None for s in with_none.steps)


def test_critic_sees_only_the_history_so_far():
    """If it were handed future steps the offline scores would be optimistic
    and the loop would behave differently from the evaluation."""
    seen = []

    class Recording:
        version = "recording"

        def score(self, question, steps):
            seen.append([s.step_index for s in steps])
            return 0.0

    rec = go("recovers_from_bad_column", critic=Recording())
    for call, indices in enumerate(seen):
        assert indices == list(range(len(indices))), "history must be contiguous"
        assert max(indices) <= call + 1, "critic was shown a step from the future"


def test_final_answer_is_not_pre_empted():
    """The critic runs after tool steps, never on the answer step — stopping a
    run that has already answered saves nothing and loses the answer."""
    rec = go("clean_success", critic=FakeCritic(0.99))
    if rec.termination == "critic_stop":
        assert rec.final_answer is None, "stopped a run that had already answered"

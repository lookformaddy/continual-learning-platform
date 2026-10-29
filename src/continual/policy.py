"""Drift-triggered retraining policy.

Drift signals are noisy, so a raw fire never retrains on its own. The
policy gates on: a fired detector, a cooldown since the last retrain
(avoiding retrain storms), and a minimum batch count since the last
action. Severity decides the action: a distribution-only signal gets a
light replay-mix refresh; an accuracy collapse gets a full replay
retrain; a collapse below the rollback floor asks for a rollback check.
"""

from __future__ import annotations

from dataclasses import dataclass

from continual.drift import DriftSignal


@dataclass
class PolicyDecision:
    action: str  # "none" | "replay_mix" | "retrain" | "rollback_check"
    reason: str
    signals: list[DriftSignal]


@dataclass
class RetrainPolicy:
    cooldown_batches: int = 3
    min_batches_since_action: int = 2
    rollback_accuracy_floor: float = 0.55
    replay_passes: int = 2

    def decide(
        self,
        signals: list[DriftSignal],
        batches_since_action: int,
        window_accuracy: float,
    ) -> PolicyDecision:
        if batches_since_action < self.min_batches_since_action:
            return PolicyDecision("none", "within min-batches-since-action gate", signals)
        fired = [s for s in signals if s.fired]
        if not fired:
            return PolicyDecision("none", "no detector fired", signals)
        if batches_since_action < self.cooldown_batches:
            return PolicyDecision("none", "within retrain cooldown", signals)
        accuracy_fired = any(s.detector == "accuracy_window" for s in fired)
        if accuracy_fired and window_accuracy < self.rollback_accuracy_floor:
            return PolicyDecision(
                "rollback_check",
                f"accuracy {window_accuracy:.3f} below rollback floor "
                f"{self.rollback_accuracy_floor}",
                signals,
            )
        if accuracy_fired:
            return PolicyDecision("retrain", "accuracy window fired", signals)
        return PolicyDecision("replay_mix", "distribution shift only", signals)

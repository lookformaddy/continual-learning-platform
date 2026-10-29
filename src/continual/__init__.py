"""Continual / online learning platform for tabular models.

Pure-stdlib core (no sklearn): online learners, drift detection,
replay-buffer mitigation, drift-triggered retraining policy, versioned
model snapshots with rollback, and a drift eval harness.
"""

from continual.data import Batch, generate_stream
from continual.drift import (
    AccuracyWindowDetector,
    DistributionShiftDetector,
    DriftSignal,
)
from continual.learners import GaussianNaiveBayesOnline, OnlineLogisticRegression
from continual.metrics import accuracy, confusion, log_loss
from continual.pipeline import BatchReport, ContinualPipeline, PipelineConfig
from continual.policy import PolicyDecision, RetrainPolicy
from continual.registry import ModelRegistry, ModelSnapshot
from continual.replay import ReservoirReplayBuffer

__all__ = [
    "AccuracyWindowDetector",
    "Batch",
    "BatchReport",
    "ContinualPipeline",
    "DistributionShiftDetector",
    "DriftSignal",
    "GaussianNaiveBayesOnline",
    "ModelRegistry",
    "ModelSnapshot",
    "OnlineLogisticRegression",
    "PipelineConfig",
    "PolicyDecision",
    "ReservoirReplayBuffer",
    "RetrainPolicy",
    "accuracy",
    "confusion",
    "generate_stream",
    "log_loss",
]

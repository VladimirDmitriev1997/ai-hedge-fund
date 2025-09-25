"""
Subpackage: hedge.fitting
- Learning/optimization helpers and differentiable losses.
"""

from .learning import LearningConfig, OptimizationConfig, FitResult, fit_strategy
from .losses import Objective, evaluate_objective
from .cost_models import COST_MODELS

__all__ = [
    "LearningConfig",
    "OptimizationConfig",
    "FitResult",
    "fit_strategy",
    "Objective",
    "evaluate_objective",
    "COST_MODELS",
]

"""
Expert Group-Specific Metrics System

Each expert group has unique evaluation criteria tailored to their domain:
- Math: Correctness, numerical precision, step-by-step reasoning
- Agentic: Tool selection accuracy, API call validity, multi-step coherence
- Planning: Long-horizon success, uncertainty calibration, recovery ability

This ensures each expert group optimizes for its specific strengths.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


class ExpertGroup(Enum):
    """Expert group identifiers."""
    MATH = 0
    AGENTIC = 1
    PLANNING = 2
    DUMMY = 3  # For testing


@dataclass
class ExpertMetrics:
    """Base metrics common to all expert groups."""
    val_loss: float
    val_aux_loss: float
    expert_diversity_score: float
    experts_active_ratio: float
    composite_score: float
    expert_group_id: int


@dataclass
class MathExpertMetrics(ExpertMetrics):
    """Math-specific metrics."""
    numerical_accuracy: float = 0.0  # % of numbers within tolerance
    equation_validity: float = 0.0  # % of valid mathematical expressions
    step_coherence: float = 0.0  # Quality of step-by-step reasoning
    answer_extraction_rate: float = 0.0  # % where answer can be extracted

    # Domain-specific penalties
    arithmetic_errors: int = 0
    unit_inconsistencies: int = 0

    def to_dict(self) -> dict[str, float]:
        base = {
            "val_loss": self.val_loss,
            "val_aux_loss": self.val_aux_loss,
            "expert_diversity_score": self.expert_diversity_score,
            "experts_active_ratio": self.experts_active_ratio,
            "expert_group_id": self.expert_group_id,
        }
        math_specific = {
            "math_numerical_accuracy": self.numerical_accuracy,
            "math_equation_validity": self.equation_validity,
            "math_step_coherence": self.step_coherence,
            "math_answer_extraction_rate": self.answer_extraction_rate,
            "math_arithmetic_errors": float(self.arithmetic_errors),
            "math_unit_inconsistencies": float(self.unit_inconsistencies),
        }

        # Compute math-specific composite score
        # Lower is better (like loss)
        math_penalty = (
            (1.0 - self.numerical_accuracy) * 0.3 +
            (1.0 - self.equation_validity) * 0.2 +
            (1.0 - self.step_coherence) * 0.2 +
            (1.0 - self.answer_extraction_rate) * 0.1 +
            self.arithmetic_errors * 0.1 +
            self.unit_inconsistencies * 0.1
        )

        base["composite_score"] = self.val_loss + math_penalty
        base.update(math_specific)
        return base


@dataclass
class AgenticExpertMetrics(ExpertMetrics):
    """Agentic/Tool-use specific metrics."""
    tool_selection_accuracy: float = 0.0  # % correct tool chosen
    api_call_validity: float = 0.0  # % of syntactically valid API calls
    parameter_correctness: float = 0.0  # % correct parameter types
    multi_step_coherence: float = 0.0  # Quality of tool chaining
    json_parse_rate: float = 0.0  # % of outputs that parse as valid JSON

    # Domain-specific penalties
    hallucinated_tools: int = 0  # Non-existent tools referenced
    parameter_type_errors: int = 0
    incomplete_tool_calls: int = 0

    def to_dict(self) -> dict[str, float]:
        base = {
            "val_loss": self.val_loss,
            "val_aux_loss": self.val_aux_loss,
            "expert_diversity_score": self.expert_diversity_score,
            "experts_active_ratio": self.experts_active_ratio,
            "expert_group_id": self.expert_group_id,
        }
        agentic_specific = {
            "agentic_tool_selection_accuracy": self.tool_selection_accuracy,
            "agentic_api_call_validity": self.api_call_validity,
            "agentic_parameter_correctness": self.parameter_correctness,
            "agentic_multi_step_coherence": self.multi_step_coherence,
            "agentic_json_parse_rate": self.json_parse_rate,
            "agentic_hallucinated_tools": float(self.hallucinated_tools),
            "agentic_parameter_type_errors": float(self.parameter_type_errors),
            "agentic_incomplete_calls": float(self.incomplete_tool_calls),
        }

        # Compute agentic-specific composite score
        agentic_penalty = (
            (1.0 - self.tool_selection_accuracy) * 0.3 +
            (1.0 - self.api_call_validity) * 0.25 +
            (1.0 - self.parameter_correctness) * 0.15 +
            (1.0 - self.multi_step_coherence) * 0.15 +
            (1.0 - self.json_parse_rate) * 0.05 +
            self.hallucinated_tools * 0.05 +
            self.parameter_type_errors * 0.05
        )

        base["composite_score"] = self.val_loss + agentic_penalty
        base.update(agentic_specific)
        return base


@dataclass
class PlanningExpertMetrics(ExpertMetrics):
    """Planning-specific metrics."""
    plan_length_appropriateness: float = 0.0  # Neither too short nor too long
    step_dependency_validity: float = 0.0  # Steps in logical order
    uncertainty_calibration: float = 0.0  # Confidence matches actual success
    recovery_capability: float = 0.0  # Can handle failures/rollbacks
    checkpoint_quality: float = 0.0  # Quality of intermediate checkpoints

    # Uncertainty decay metrics
    avg_confidence_per_step: list[float] = field(default_factory=list)
    confidence_decay_rate: float = 0.0  # Learned decay factor

    # Domain-specific penalties
    circular_dependencies: int = 0
    missing_prerequisites: int = 0
    unrealistic_assumptions: int = 0

    def to_dict(self) -> dict[str, float]:
        base = {
            "val_loss": self.val_loss,
            "val_aux_loss": self.val_aux_loss,
            "expert_diversity_score": self.expert_diversity_score,
            "experts_active_ratio": self.experts_active_ratio,
            "expert_group_id": self.expert_group_id,
        }
        planning_specific = {
            "planning_length_appropriateness": self.plan_length_appropriateness,
            "planning_dependency_validity": self.step_dependency_validity,
            "planning_uncertainty_calibration": self.uncertainty_calibration,
            "planning_recovery_capability": self.recovery_capability,
            "planning_checkpoint_quality": self.checkpoint_quality,
            "planning_confidence_decay_rate": self.confidence_decay_rate,
            "planning_circular_dependencies": float(self.circular_dependencies),
            "planning_missing_prerequisites": float(self.missing_prerequisites),
            "planning_unrealistic_assumptions": float(self.unrealistic_assumptions),
        }

        # Compute planning-specific composite score
        planning_penalty = (
            (1.0 - self.plan_length_appropriateness) * 0.15 +
            (1.0 - self.step_dependency_validity) * 0.25 +
            (1.0 - self.uncertainty_calibration) * 0.25 +
            (1.0 - self.recovery_capability) * 0.15 +
            (1.0 - self.checkpoint_quality) * 0.1 +
            self.circular_dependencies * 0.05 +
            self.missing_prerequisites * 0.05
        )

        base["composite_score"] = self.val_loss + planning_penalty
        base.update(planning_specific)
        return base


class ExpertMetricsComputer:
    """Computes domain-specific metrics for each expert group."""

    def __init__(self, expert_group_id: int):
        self.expert_group_id = expert_group_id
        self.group = ExpertGroup(expert_group_id)

    def compute_metrics(
        self,
        predictions: list[str],
        references: list[str],
        base_metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute expert-specific metrics.

        Args:
            predictions: Model outputs
            references: Ground truth/expected outputs
            base_metrics: Base metrics (loss, diversity, etc.)

        Returns:
            Dictionary of metrics including domain-specific ones
        """
        if self.group == ExpertGroup.MATH:
            return self._compute_math_metrics(predictions, references, base_metrics)
        elif self.group == ExpertGroup.AGENTIC:
            return self._compute_agentic_metrics(predictions, references, base_metrics)
        elif self.group == ExpertGroup.PLANNING:
            return self._compute_planning_metrics(predictions, references, base_metrics)
        else:
            # Dummy/fallback
            return base_metrics

    def _compute_math_metrics(
        self,
        predictions: list[str],
        references: list[str],
        base_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Compute math-specific metrics."""
        numerical_correct = 0
        equations_valid = 0
        answers_extracted = 0
        arithmetic_errors = 0

        for pred, ref in zip(predictions, references):
            # Extract numbers from prediction and reference
            pred_numbers = self._extract_numbers(pred)
            ref_numbers = self._extract_numbers(ref)

            # Check numerical accuracy (within 1% tolerance)
            if pred_numbers and ref_numbers:
                if self._numbers_match(pred_numbers[-1], ref_numbers[-1], tolerance=0.01):
                    numerical_correct += 1
                else:
                    arithmetic_errors += 1

            # Check equation validity
            if self._contains_valid_equations(pred):
                equations_valid += 1

            # Check if answer can be extracted
            if self._can_extract_answer(pred):
                answers_extracted += 1

        n = len(predictions) if predictions else 1

        metrics = MathExpertMetrics(
            val_loss=base_metrics["val_loss"],
            val_aux_loss=base_metrics.get("val_aux_loss", 0.0),
            expert_diversity_score=base_metrics.get("expert_diversity_score", 0.0),
            experts_active_ratio=base_metrics.get("experts_active_ratio", 0.0),
            expert_group_id=self.expert_group_id,
            numerical_accuracy=numerical_correct / n,
            equation_validity=equations_valid / n,
            answer_extraction_rate=answers_extracted / n,
            step_coherence=self._assess_step_coherence(predictions),
            arithmetic_errors=arithmetic_errors,
            composite_score=0.0,  # Will be computed in to_dict()
        )

        return metrics.to_dict()

    def _compute_agentic_metrics(
        self,
        predictions: list[str],
        references: list[str],
        base_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Compute agentic/tool-use metrics."""
        valid_api_calls = 0
        json_parseable = 0
        hallucinated_tools = 0
        param_errors = 0

        known_tools = {"calculator", "search", "database_query", "file_read", "api_call"}  # Example

        for pred in predictions:
            # Check JSON validity
            if self._is_valid_json_or_function_call(pred):
                json_parseable += 1
                valid_api_calls += 1

            # Check for hallucinated tools
            mentioned_tools = self._extract_tool_names(pred)
            for tool in mentioned_tools:
                if tool not in known_tools:
                    hallucinated_tools += 1

            # Check parameter types
            if not self._check_parameter_types(pred):
                param_errors += 1

        n = len(predictions) if predictions else 1

        metrics = AgenticExpertMetrics(
            val_loss=base_metrics["val_loss"],
            val_aux_loss=base_metrics.get("val_aux_loss", 0.0),
            expert_diversity_score=base_metrics.get("expert_diversity_score", 0.0),
            experts_active_ratio=base_metrics.get("experts_active_ratio", 0.0),
            expert_group_id=self.expert_group_id,
            api_call_validity=valid_api_calls / n,
            json_parse_rate=json_parseable / n,
            tool_selection_accuracy=self._assess_tool_selection(predictions, references),
            parameter_correctness=1.0 - (param_errors / n),
            multi_step_coherence=self._assess_tool_chain_coherence(predictions),
            hallucinated_tools=hallucinated_tools,
            parameter_type_errors=param_errors,
            composite_score=0.0,
        )

        return metrics.to_dict()

    def _compute_planning_metrics(
        self,
        predictions: list[str],
        references: list[str],
        base_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Compute planning-specific metrics."""
        circular_deps = 0
        missing_prereqs = 0
        confidences_per_step = []

        for pred in predictions:
            # Extract confidence markers
            confidences = self._extract_confidence_markers(pred)
            if confidences:
                confidences_per_step.append(np.mean(confidences))

            # Check for circular dependencies
            if self._has_circular_dependencies(pred):
                circular_deps += 1

            # Check for missing prerequisites
            if self._has_missing_prerequisites(pred):
                missing_prereqs += 1

        # Compute uncertainty decay rate
        decay_rate = 0.9  # Default
        if confidences_per_step:
            # Fit exponential decay
            decay_rate = self._fit_confidence_decay(confidences_per_step)

        n = len(predictions) if predictions else 1

        metrics = PlanningExpertMetrics(
            val_loss=base_metrics["val_loss"],
            val_aux_loss=base_metrics.get("val_aux_loss", 0.0),
            expert_diversity_score=base_metrics.get("expert_diversity_score", 0.0),
            experts_active_ratio=base_metrics.get("experts_active_ratio", 0.0),
            expert_group_id=self.expert_group_id,
            plan_length_appropriateness=self._assess_plan_length(predictions),
            step_dependency_validity=1.0 - (circular_deps / n),
            uncertainty_calibration=self._assess_uncertainty_calibration(predictions, references),
            recovery_capability=self._assess_recovery_capability(predictions),
            checkpoint_quality=self._assess_checkpoint_quality(predictions),
            confidence_decay_rate=decay_rate,
            avg_confidence_per_step=confidences_per_step,
            circular_dependencies=circular_deps,
            missing_prerequisites=missing_prereqs,
            composite_score=0.0,
        )

        return metrics.to_dict()

    # Helper methods for math metrics
    def _extract_numbers(self, text: str) -> list[float]:
        """Extract all numbers from text."""
        pattern = r'-?\d+\.?\d*'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches if m]

    def _numbers_match(self, a: float, b: float, tolerance: float = 0.01) -> bool:
        """Check if two numbers match within tolerance."""
        if abs(b) < 1e-10:  # Avoid division by zero
            return abs(a - b) < tolerance
        return abs(a - b) / abs(b) < tolerance

    def _contains_valid_equations(self, text: str) -> bool:
        """Check if text contains valid mathematical equations."""
        # Simple heuristic: look for = and valid operators
        return '=' in text and any(op in text for op in ['+', '-', '*', '/'])

    def _can_extract_answer(self, text: str) -> bool:
        """Check if a final answer can be extracted."""
        # Look for common answer patterns
        patterns = [
            r'answer is\s+(-?\d+\.?\d*)',
            r'= (-?\d+\.?\d*)\s*$',
            r'result:\s+(-?\d+\.?\d*)',
            r'\*\*(-?\d+\.?\d*)\*\*',  # Bold numbers
        ]
        return any(re.search(p, text.lower()) for p in patterns)

    def _assess_step_coherence(self, predictions: list[str]) -> float:
        """Assess quality of step-by-step reasoning."""
        # Heuristic: count steps, check for logical flow markers
        coherent_count = 0
        for pred in predictions:
            steps = len(re.findall(r'step \d+|first,|then,|next,|finally,', pred.lower()))
            if steps >= 2:  # At least 2 steps shown
                coherent_count += 1
        return coherent_count / len(predictions) if predictions else 0.0

    # Helper methods for agentic metrics
    def _is_valid_json_or_function_call(self, text: str) -> bool:
        """Check if text is valid JSON or function call syntax."""
        try:
            json.loads(text)
            return True
        except:
            # Check for function call pattern
            return bool(re.match(r'\w+\([^)]*\)', text))

    def _extract_tool_names(self, text: str) -> list[str]:
        """Extract mentioned tool names."""
        # Look for function call patterns
        pattern = r'(\w+)\s*\('
        return re.findall(pattern, text)

    def _check_parameter_types(self, text: str) -> bool:
        """Basic parameter type checking."""
        # Heuristic: look for common type errors
        errors = [
            'null',  # Unexpected null
            'undefined',
            'NaN',
        ]
        return not any(err in text.lower() for err in errors)

    def _assess_tool_selection(self, predictions: list[str], references: list[str]) -> float:
        """Assess if correct tools were selected."""
        # Simplified: check if mentioned tools match reference
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_tools = set(self._extract_tool_names(pred))
            ref_tools = set(self._extract_tool_names(ref))
            if pred_tools & ref_tools:  # Any overlap
                correct += 1
        return correct / len(predictions) if predictions else 0.0

    def _assess_tool_chain_coherence(self, predictions: list[str]) -> float:
        """Assess quality of tool chaining."""
        coherent = 0
        for pred in predictions:
            tools = self._extract_tool_names(pred)
            # Multiple tools used in sequence
            if len(tools) >= 2:
                coherent += 1
        return coherent / len(predictions) if predictions else 0.0

    # Helper methods for planning metrics
    def _extract_confidence_markers(self, text: str) -> list[float]:
        """Extract confidence scores from text."""
        pattern = r'\[CONF:(\d+\.?\d*)\]'
        matches = re.findall(pattern, text)
        return [float(m) for m in matches]

    def _has_circular_dependencies(self, text: str) -> bool:
        """Check for circular dependencies in plan."""
        # Simplified: look for step references that go backwards
        steps = re.findall(r'step (\d+)', text.lower())
        if len(steps) >= 2:
            # Check if later steps reference earlier ones incorrectly
            for i, step in enumerate(steps[1:], 1):
                if int(step) < i:
                    return True
        return False

    def _has_missing_prerequisites(self, text: str) -> bool:
        """Check for missing prerequisites."""
        # Heuristic: look for "requires" without corresponding step
        requires = re.findall(r'requires (.+?)(?:\n|$)', text.lower())
        return len(requires) > 0  # Simplified check

    def _fit_confidence_decay(self, confidences: list[float]) -> float:
        """Fit exponential decay to confidence sequence."""
        if len(confidences) < 2:
            return 0.9
        # Simple exponential fit
        try:
            ratios = [confidences[i+1] / confidences[i] for i in range(len(confidences)-1) if confidences[i] > 0]
            return np.mean(ratios) if ratios else 0.9
        except:
            return 0.9

    def _assess_plan_length(self, predictions: list[str]) -> float:
        """Assess if plan length is appropriate."""
        appropriate = 0
        for pred in predictions:
            steps = len(re.findall(r'step \d+', pred.lower()))
            # Appropriate: 3-15 steps
            if 3 <= steps <= 15:
                appropriate += 1
        return appropriate / len(predictions) if predictions else 0.0

    def _assess_uncertainty_calibration(self, predictions: list[str], references: list[str]) -> float:
        """Assess if uncertainty matches actual success."""
        # Simplified placeholder
        return 0.7  # Would need actual success/failure data

    def _assess_recovery_capability(self, predictions: list[str]) -> float:
        """Assess ability to handle failures."""
        has_recovery = 0
        for pred in predictions:
            recovery_markers = ['[CHECKPOINT]', '[ALT]', 'fallback', 'rollback', 'retry']
            if any(marker in pred for marker in recovery_markers):
                has_recovery += 1
        return has_recovery / len(predictions) if predictions else 0.0

    def _assess_checkpoint_quality(self, predictions: list[str]) -> float:
        """Assess quality of checkpoints."""
        quality_score = 0
        for pred in predictions:
            checkpoints = pred.count('[CHECKPOINT]')
            # Good: 1 checkpoint per 3-5 steps
            steps = len(re.findall(r'step \d+', pred.lower()))
            if steps > 0:
                ratio = checkpoints / steps
                if 0.15 <= ratio <= 0.35:  # ~1 per 3-7 steps
                    quality_score += 1
        return quality_score / len(predictions) if predictions else 0.0


def get_expert_metrics_computer(expert_group_id: int) -> ExpertMetricsComputer:
    """Factory function to get metrics computer for expert group."""
    return ExpertMetricsComputer(expert_group_id)

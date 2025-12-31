"""
Uncertainty Decay Modeling for Multi-Step Planning

Core concept: As plans get longer, uncertainty compounds.
Model explicitly tracks and decays confidence at each step.

P(success at step n) = P_base × decay^n × confidence(state_n)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty decay."""
    base_confidence: float = 0.95  # Starting confidence
    decay_rate: float = 0.92  # Per-step decay
    min_confidence: float = 0.3  # Floor
    checkpoint_bonus: float = 0.1  # Recovery at checkpoints
    branching_penalty: float = 0.05  # Uncertainty from alternatives


# =============================================================================
# Special Tokens for Uncertainty Tracking
# =============================================================================

UNCERTAINTY_TOKENS = {
    # Confidence levels
    "[CONF:HIGH]": 0.9,
    "[CONF:MED]": 0.7,
    "[CONF:LOW]": 0.5,
    "[CONF:UNCERTAIN]": 0.3,
    
    # Step markers
    "[STEP:1]": None,
    "[STEP:2]": None,
    "[STEP:3]": None,
    "[STEP:N]": None,
    
    # Control flow
    "[CHECKPOINT]": "checkpoint",  # Recovery point
    "[ALT]": "alternative",  # Branch
    "[RISK:HIGH]": "high_risk",
    "[RISK:MED]": "medium_risk",
    "[RISK:LOW]": "low_risk",
    
    # Failure modes
    "[FAIL:POSSIBLE]": "possible_failure",
    "[RECOVER]": "recovery_action",
    "[ABORT]": "abort_plan",
}


def add_uncertainty_tokens(tokenizer):
    """Add uncertainty tokens to tokenizer vocabulary."""
    new_tokens = list(UNCERTAINTY_TOKENS.keys())
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    return num_added


# =============================================================================
# Uncertainty Computation
# =============================================================================

class UncertaintyTracker:
    """
    Tracks uncertainty through a multi-step plan.
    """
    
    def __init__(self, config: UncertaintyConfig | None = None):
        self.config = config or UncertaintyConfig()
        self.reset()
    
    def reset(self):
        """Reset tracker for new plan."""
        self.current_step = 0
        self.confidence = self.config.base_confidence
        self.checkpoints: list[float] = []
        self.history: list[dict] = []
    
    def step(
        self,
        step_confidence: float | None = None,
        is_checkpoint: bool = False,
        is_branch: bool = False,
        risk_level: str = "low",
    ) -> float:
        """
        Advance one step and compute new confidence.
        
        Args:
            step_confidence: Override confidence for this step
            is_checkpoint: If True, this is a recovery point
            is_branch: If True, this step has alternatives
            risk_level: "low", "medium", or "high"
        
        Returns:
            Updated confidence score
        """
        self.current_step += 1
        
        # Base decay
        self.confidence *= self.config.decay_rate
        
        # Apply step confidence if provided
        if step_confidence is not None:
            self.confidence *= step_confidence
        
        # Risk adjustment
        risk_multipliers = {"low": 1.0, "medium": 0.9, "high": 0.75}
        self.confidence *= risk_multipliers.get(risk_level, 1.0)
        
        # Branching adds uncertainty
        if is_branch:
            self.confidence -= self.config.branching_penalty
        
        # Checkpoints allow recovery
        if is_checkpoint:
            self.checkpoints.append(self.confidence)
            self.confidence += self.config.checkpoint_bonus
        
        # Apply floor
        self.confidence = max(self.confidence, self.config.min_confidence)
        
        # Record history
        self.history.append({
            "step": self.current_step,
            "confidence": self.confidence,
            "is_checkpoint": is_checkpoint,
            "is_branch": is_branch,
            "risk_level": risk_level,
        })
        
        return self.confidence
    
    def rollback_to_checkpoint(self) -> float:
        """Rollback to last checkpoint."""
        if self.checkpoints:
            self.confidence = self.checkpoints.pop() + self.config.checkpoint_bonus
        return self.confidence
    
    def get_confidence_token(self) -> str:
        """Get appropriate confidence token for current state."""
        if self.confidence >= 0.85:
            return "[CONF:HIGH]"
        elif self.confidence >= 0.65:
            return "[CONF:MED]"
        elif self.confidence >= 0.45:
            return "[CONF:LOW]"
        else:
            return "[CONF:UNCERTAIN]"


# =============================================================================
# Text Processing with Uncertainty
# =============================================================================

def annotate_plan_with_uncertainty(
    plan_text: str,
    tracker: UncertaintyTracker | None = None,
) -> str:
    """
    Add uncertainty annotations to a plan.
    
    Detects steps, checkpoints, and risks, then injects confidence tokens.
    """
    if tracker is None:
        tracker = UncertaintyTracker()
        tracker.reset()
    
    lines = plan_text.split("\n")
    annotated_lines = []
    
    step_patterns = [
        r"^(\d+)\.",  # "1. Do something"
        r"^Step (\d+)",  # "Step 1:"
        r"^First|Second|Third|Then|Next|Finally",  # Sequential markers
    ]
    
    checkpoint_keywords = ["verify", "check", "confirm", "validate", "ensure"]
    risk_keywords = {
        "high": ["dangerous", "critical", "risky", "careful", "warning"],
        "medium": ["might", "could", "possibly", "may fail"],
        "low": ["simple", "straightforward", "easy"],
    }
    branch_keywords = ["alternatively", "or", "option", "either"]
    
    for line in lines:
        line_lower = line.lower()
        
        # Detect step
        is_step = any(re.match(pattern, line, re.IGNORECASE) for pattern in step_patterns)
        
        if is_step:
            # Detect checkpoint
            is_checkpoint = any(kw in line_lower for kw in checkpoint_keywords)
            
            # Detect branching
            is_branch = any(kw in line_lower for kw in branch_keywords)
            
            # Detect risk level
            risk = "low"
            for level, keywords in risk_keywords.items():
                if any(kw in line_lower for kw in keywords):
                    risk = level
                    break
            
            # Update tracker
            confidence = tracker.step(
                is_checkpoint=is_checkpoint,
                is_branch=is_branch,
                risk_level=risk,
            )
            
            # Add annotations
            conf_token = tracker.get_confidence_token()
            step_token = f"[STEP:{tracker.current_step}]"
            
            annotations = [step_token, conf_token]
            if is_checkpoint:
                annotations.append("[CHECKPOINT]")
            if is_branch:
                annotations.append("[ALT]")
            if risk != "low":
                annotations.append(f"[RISK:{risk.upper()}]")
            
            annotated_line = " ".join(annotations) + " " + line
        else:
            annotated_line = line
        
        annotated_lines.append(annotated_line)
    
    return "\n".join(annotated_lines)


# =============================================================================
# Loss Weighting by Uncertainty
# =============================================================================

class UncertaintyWeightedLoss(nn.Module):
    """
    Weight loss by inverse uncertainty.
    
    High-confidence steps get more loss weight (model should be more certain).
    Low-confidence steps get less weight (expected to be harder).
    """
    
    def __init__(self, base_loss_fn: nn.Module | None = None):
        super().__init__()
        self.base_loss_fn = base_loss_fn or nn.CrossEntropyLoss(reduction="none")
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        confidence_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            logits: Model outputs [batch, seq, vocab]
            targets: Target token IDs [batch, seq]
            confidence_scores: Per-token confidence [batch, seq]
        
        Returns:
            Weighted loss scalar
        """
        # Flatten for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        flat_logits = logits.view(-1, vocab_size)
        flat_targets = targets.view(-1)
        
        # Base loss per token
        token_losses = self.base_loss_fn(flat_logits, flat_targets)
        token_losses = token_losses.view(batch_size, seq_len)
        
        if confidence_scores is not None:
            # Weight by confidence (higher confidence = higher weight)
            weights = confidence_scores.clamp(0.1, 1.0)
            weighted_loss = (token_losses * weights).sum() / weights.sum()
        else:
            weighted_loss = token_losses.mean()
        
        return weighted_loss


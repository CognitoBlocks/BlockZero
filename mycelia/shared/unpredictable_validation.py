"""
Unpredictable Validation Dataset System

Anti-cheat measures to prevent miners from gaming validation metrics:
1. Cryptographic randomness from blockchain data
2. Dynamic dataset rotation
3. Multiple validation strategies (never the same twice)
4. Synthetic perturbations
5. Compositional generalization tests
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


class ValidationStrategy(Enum):
    """Different validation sampling strategies to rotate between."""
    RANDOM_HOLDOUT = "random_holdout"  # Pure random from holdout
    STRATIFIED_DIFFICULTY = "stratified_difficulty"  # Sample across difficulty levels
    COMPOSITIONAL = "compositional"  # Unseen combinations
    PERTURBED = "perturbed"  # Modified versions of training data
    ADVERSARIAL = "adversarial"  # Known failure modes
    TEMPORAL_SPLIT = "temporal_split"  # Recent data only
    CROSS_DOMAIN = "cross_domain"  # Different domain mix


@dataclass
class ValidationSeed:
    """
    Cryptographically secure seed for validation dataset selection.

    Combines multiple sources of entropy to make prediction impossible:
    - Blockchain block hash (unpredictable until block is mined)
    - Validator consensus seed (multiple validators must agree)
    - Cycle number (changes each validation round)
    - Secret salt (known only to validators)
    """
    block_hash: str  # Current blockchain block hash
    validator_seed: str  # Combined validator seeds from commits
    cycle_number: int  # Current cycle/epoch
    secret_salt: str | None = None  # Optional additional entropy

    def to_int(self) -> int:
        """Convert to deterministic integer seed."""
        combined = f"{self.block_hash}{self.validator_seed}{self.cycle_number}{self.secret_salt or ''}"
        return int(hashlib.sha256(combined.encode()).hexdigest(), 16)

    def to_rng(self) -> np.random.Generator:
        """Create numpy RNG from this seed."""
        # Use only bottom 64 bits for numpy (it doesn't accept larger seeds)
        seed_64bit = self.to_int() % (2**64)
        return np.random.default_rng(seed_64bit)

    def derive_subseed(self, context: str) -> int:
        """Derive a subseed for specific purpose."""
        combined = f"{self.to_int()}{context}"
        return int(hashlib.sha256(combined.encode()).hexdigest(), 16) % (2**64)


class UnpredictableValidator:
    """
    Orchestrates unpredictable validation to prevent gaming.

    Key principles:
    1. Miners cannot predict which data will be used for validation
    2. Strategy changes each cycle (random holdout, perturbed, compositional, etc.)
    3. Uses blockchain randomness that doesn't exist during training
    4. Multiple validators must agree on seed (no single point of failure)
    """

    def __init__(
        self,
        expert_group_id: int,
        dataset_sources: list[str],  # HF dataset names for this expert group
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int = 1024,
        samples_per_validation: int = 500,
    ):
        self.expert_group_id = expert_group_id
        self.dataset_sources = dataset_sources
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.samples_per_validation = samples_per_validation

        self._holdout_cache: dict[str, Dataset] = {}

    def select_strategy(self, validation_seed: ValidationSeed) -> ValidationStrategy:
        """
        Unpredictably select which validation strategy to use.

        Rotates between strategies to prevent overfitting to any single approach.
        """
        rng = validation_seed.to_rng()
        strategies = list(ValidationStrategy)

        # Weight different strategies
        weights = np.array([
            3.0,  # RANDOM_HOLDOUT - most common
            2.0,  # STRATIFIED_DIFFICULTY
            1.5,  # COMPOSITIONAL
            2.0,  # PERTURBED
            1.0,  # ADVERSARIAL - less common
            1.5,  # TEMPORAL_SPLIT
            1.0,  # CROSS_DOMAIN
        ])
        weights = weights / weights.sum()

        idx = rng.choice(len(strategies), p=weights)
        selected = strategies[idx]

        logger.info(
            "Selected validation strategy",
            strategy=selected.value,
            cycle=validation_seed.cycle_number,
        )
        return selected

    def get_validation_dataset(
        self,
        validation_seed: ValidationSeed,
        strategy: ValidationStrategy | None = None,
    ) -> Dataset:
        """
        Generate unpredictable validation dataset for this cycle.

        Args:
            validation_seed: Cryptographic seed from blockchain + validators
            strategy: Override automatic strategy selection (for testing)

        Returns:
            Dataset with validation examples (different each time)
        """
        if strategy is None:
            strategy = self.select_strategy(validation_seed)

        rng = validation_seed.to_rng()

        # Dispatch to strategy-specific implementation
        if strategy == ValidationStrategy.RANDOM_HOLDOUT:
            return self._random_holdout(validation_seed, rng)
        elif strategy == ValidationStrategy.STRATIFIED_DIFFICULTY:
            return self._stratified_difficulty(validation_seed, rng)
        elif strategy == ValidationStrategy.COMPOSITIONAL:
            return self._compositional(validation_seed, rng)
        elif strategy == ValidationStrategy.PERTURBED:
            return self._perturbed(validation_seed, rng)
        elif strategy == ValidationStrategy.ADVERSARIAL:
            return self._adversarial(validation_seed, rng)
        elif strategy == ValidationStrategy.TEMPORAL_SPLIT:
            return self._temporal_split(validation_seed, rng)
        elif strategy == ValidationStrategy.CROSS_DOMAIN:
            return self._cross_domain(validation_seed, rng)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _random_holdout(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Pure random sampling from holdout set.

        This is the baseline - completely unpredictable selection.
        """
        # Load or retrieve holdout dataset
        holdout = self._get_or_create_holdout()

        # Random sample using cryptographic seed
        n_samples = min(self.samples_per_validation, len(holdout))
        indices = rng.choice(len(holdout), size=n_samples, replace=False)

        logger.info(
            "Random holdout sampling",
            n_samples=n_samples,
            total_holdout=len(holdout),
        )

        return holdout.select(indices.tolist())

    def _stratified_difficulty(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Sample across different difficulty levels.

        Prevents overfitting to easy examples by ensuring coverage.
        """
        holdout = self._get_or_create_holdout()

        # Simple difficulty heuristic: length of text
        # More sophisticated: use perplexity, answer length, etc.
        def get_difficulty(example: dict) -> float:
            text = example.get("text", example.get("question", ""))
            return len(text)

        difficulties = [get_difficulty(holdout[i]) for i in range(len(holdout))]

        # Divide into quintiles
        percentiles = np.percentile(difficulties, [20, 40, 60, 80])

        # Sample from each quintile
        samples_per_quintile = self.samples_per_validation // 5
        selected_indices = []

        for i in range(5):
            if i == 0:
                mask = np.array(difficulties) <= percentiles[0]
            elif i == 4:
                mask = np.array(difficulties) > percentiles[3]
            else:
                mask = (np.array(difficulties) > percentiles[i-1]) & (np.array(difficulties) <= percentiles[i])

            quintile_indices = np.where(mask)[0]
            if len(quintile_indices) > 0:
                n_sample = min(samples_per_quintile, len(quintile_indices))
                selected = rng.choice(quintile_indices, size=n_sample, replace=False)
                selected_indices.extend(selected.tolist())

        logger.info(
            "Stratified difficulty sampling",
            n_samples=len(selected_indices),
            percentiles=percentiles.tolist(),
        )

        return holdout.select(selected_indices)

    def _compositional(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Test compositional generalization.

        Combine known primitives in unseen ways.
        Example: If trained on "A+B" and "B+C", test on "A+C"
        """
        # This requires domain-specific logic
        # For now, approximate with cross-domain mixing
        return self._cross_domain(validation_seed, rng)

    def _perturbed(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Apply random perturbations to validation data.

        Domain-specific transformations:
        - Math: Change numbers while preserving structure
        - Code: Rename variables
        - Text: Paraphrase
        """
        holdout = self._get_or_create_holdout()
        n_samples = min(self.samples_per_validation, len(holdout))
        indices = rng.choice(len(holdout), size=n_samples, replace=False)

        subset = holdout.select(indices.tolist())

        # Apply perturbations (domain-specific)
        perturbed = subset.map(
            lambda ex: self._apply_perturbation(ex, rng),
            desc="Applying perturbations",
        )

        logger.info("Perturbed validation sampling", n_samples=n_samples)
        return perturbed

    def _adversarial(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Test known failure modes and adversarial examples.

        Examples:
        - Math: Edge cases (negative numbers, zero, very large)
        - Code: Unusual syntax, edge case inputs
        - Planning: Long-horizon scenarios
        """
        # Would require a curated adversarial dataset
        # For now, fall back to perturbed
        return self._perturbed(validation_seed, rng)

    def _temporal_split(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Use only recent data (temporal validation).

        Tests if model generalizes to new examples from same distribution.
        """
        # This requires dataset timestamps
        # For now, approximate with random sampling
        return self._random_holdout(validation_seed, rng)

    def _cross_domain(self, validation_seed: ValidationSeed, rng: np.random.Generator) -> Dataset:
        """
        Mix examples from different datasets in unexpected ratios.

        Prevents overfitting to specific dataset blend used in training.
        """
        all_examples = []

        # Load small samples from each source
        for source in self.dataset_sources[:3]:  # Limit to first 3 for speed
            try:
                ds = load_dataset(source, split="train", streaming=True)
                # Take small sample
                samples = list(ds.take(100))
                all_examples.extend(samples)
            except Exception as e:
                logger.warning(f"Failed to load {source} for cross-domain: {e}")
                continue

        if not all_examples:
            # Fallback to holdout
            return self._random_holdout(validation_seed, rng)

        # Random subsample
        n_samples = min(self.samples_per_validation, len(all_examples))
        indices = rng.choice(len(all_examples), size=n_samples, replace=False)

        selected = [all_examples[i] for i in indices]

        logger.info("Cross-domain sampling", n_samples=n_samples, sources=len(self.dataset_sources))
        return Dataset.from_list(selected)

    def _apply_perturbation(self, example: dict, rng: np.random.Generator) -> dict:
        """
        Apply domain-specific perturbation to example.

        Override in subclasses for expert-specific logic.
        """
        # Math group: perturb numbers
        if self.expert_group_id == 0:  # Math group
            return self._perturb_math_example(example, rng)
        # Agentic group: perturb tool names
        elif self.expert_group_id == 1:  # Agentic group
            return self._perturb_agentic_example(example, rng)
        # Planning group: shuffle steps
        elif self.expert_group_id == 2:  # Planning group
            return self._perturb_planning_example(example, rng)
        else:
            return example

    def _perturb_math_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Perturb numbers in math problems."""
        import re

        text = example.get("text", example.get("question", ""))

        def perturb_number(match):
            num = float(match.group())
            # Small random perturbation
            offset_pct = rng.uniform(-0.2, 0.2)
            new_num = num * (1 + offset_pct)

            # Keep int if original was int
            if "." not in match.group():
                return str(int(round(new_num)))
            return f"{new_num:.2f}"

        perturbed_text = re.sub(r"\d+\.?\d*", perturb_number, text)

        result = example.copy()
        if "text" in result:
            result["text"] = perturbed_text
        if "question" in result:
            result["question"] = perturbed_text

        return result

    def _perturb_agentic_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Randomize tool/function names."""
        # Would require parsing function calls and remapping names
        # For now, return unchanged
        return example

    def _perturb_planning_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Shuffle non-dependent planning steps."""
        # Would require dependency parsing
        # For now, return unchanged
        return example

    def _get_or_create_holdout(self) -> Dataset:
        """
        Get or create holdout dataset for this expert group.

        Holdout is created once and cached.
        """
        cache_key = f"expert_{self.expert_group_id}"

        if cache_key in self._holdout_cache:
            return self._holdout_cache[cache_key]

        # Load first dataset source as holdout base
        # In production, would merge all sources
        if not self.dataset_sources:
            raise ValueError("No dataset sources configured")

        logger.info(f"Creating holdout dataset from {self.dataset_sources[0]}")

        # Load full dataset (not streaming for holdout)
        try:
            ds = load_dataset(self.dataset_sources[0], split="train")
        except Exception:
            # Try with streaming and take sample
            ds_stream = load_dataset(self.dataset_sources[0], split="train", streaming=True)
            samples = list(ds_stream.take(10000))  # Take 10k examples
            ds = Dataset.from_list(samples)

        # Take 1% as holdout using deterministic hash
        def is_holdout(example, idx):
            text = str(example.get("text", example.get("question", "")))[:100]
            h = hashlib.md5(f"{text}{idx}".encode()).hexdigest()
            hash_val = int(h, 16) / (16 ** 32)
            return hash_val < 0.01

        holdout = ds.filter(is_holdout, with_indices=True)

        logger.info(f"Created holdout dataset", size=len(holdout))

        self._holdout_cache[cache_key] = holdout
        return holdout


def get_blockchain_entropy(block_hash: str, validator_seeds: dict[str, int], cycle: int) -> ValidationSeed:
    """
    Create validation seed from blockchain and validator consensus.

    Args:
        block_hash: Current block hash from chain
        validator_seeds: Dict of {validator_hotkey: seed} from commits
        cycle: Current validation cycle number

    Returns:
        Cryptographically secure ValidationSeed
    """
    # Combine all validator seeds
    combined_validator = hashlib.sha256(
        "".join(str(s) for s in sorted(validator_seeds.values())).encode()
    ).hexdigest()

    # Additional secret salt (could be loaded from secure config)
    secret_salt = secrets.token_hex(16)

    return ValidationSeed(
        block_hash=block_hash,
        validator_seed=combined_validator,
        cycle_number=cycle,
        secret_salt=secret_salt,
    )


# Example usage:
"""
# In validator code:
from mycelia.shared.unpredictable_validation import UnpredictableValidator, get_blockchain_entropy

# Setup
validator = UnpredictableValidator(
    expert_group_id=0,  # Math group
    dataset_sources=["openai/gsm8k", "lighteval/MATH"],
    tokenizer=tokenizer,
    samples_per_validation=500,
)

# Each validation cycle:
block_hash = subtensor.get_block_hash(current_block)
validator_seeds = get_validator_seeds_from_commits(config, subtensor)

validation_seed = get_blockchain_entropy(block_hash, validator_seeds, cycle_number)
validation_dataset = validator.get_validation_dataset(validation_seed)

# Now evaluate miners on this unpredictable dataset
evaluate_model(model, validation_dataset, ...)
"""

"""
Integration Module for Unpredictable Validation

Connects the UnpredictableValidator to the existing subnet validation flow.
"""
from __future__ import annotations

import bittensor
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from mycelia.shared.app_logging import structlog
from mycelia.shared.config import ValidatorConfig
from mycelia.shared.cycle import get_combined_validator_seed, get_phase, get_validator_seed_from_commit
from mycelia.shared.chain import get_chain_commits
from mycelia.shared.unpredictable_validation import (
    UnpredictableValidator,
    ValidationSeed,
    get_blockchain_entropy,
)

logger = structlog.get_logger(__name__)


# Expert group dataset mappings
EXPERT_DATASETS = {
    0: [  # Math group
        "openai/gsm8k",
        "lighteval/MATH",
        "meta-math/MetaMathQA",
        "nvidia/OpenMathInstruct-2",
    ],
    1: [  # Agentic group
        "glaiveai/glaive-function-calling-v2",
        "Salesforce/xlam-function-calling-60k",
        "NousResearch/hermes-function-calling-v1",
    ],
    2: [  # Planning group
        "kaist-ai/CoT-Collection",
        "allenai/WildChat",
        "bigcode/self-oss-instruct-sc2",
    ],
    # Dummy group for testing
    3: ["allenai/c4"],
}


def get_validation_seed_from_chain(
    config: ValidatorConfig,
    subtensor: bittensor.Subtensor,
    use_secret_salt: bool = True,
) -> ValidationSeed:
    """
    Generate validation seed from blockchain state.

    This seed is cryptographically unpredictable because:
    1. Block hash doesn't exist until block is mined (during training phase)
    2. Requires consensus from multiple validators
    3. Changes every cycle
    4. Optionally includes secret salt

    Args:
        config: Validator configuration
        subtensor: Bittensor subtensor instance
        use_secret_salt: Whether to add additional entropy (recommended for production)

    Returns:
        ValidationSeed that changes each cycle
    """
    # Get current block hash
    phase = get_phase(config)
    current_block = phase.block

    try:
        # Get block hash (this is unpredictable during training)
        block_hash = subtensor.get_block_hash(current_block)
        if block_hash is None:
            logger.warning(f"Could not get block hash for {current_block}, using block number")
            block_hash = str(current_block)
    except Exception as e:
        logger.warning(f"Error getting block hash: {e}, using block number")
        block_hash = str(current_block)

    # Get validator seeds from chain commits
    commits = get_chain_commits(config, subtensor)
    validator_seeds = get_validator_seed_from_commit(config, commits)

    if not validator_seeds:
        logger.warning("No validator seeds found, using fallback")
        validator_seeds = {"fallback": current_block}

    # Combine into cryptographic seed
    validation_seed = get_blockchain_entropy(
        block_hash=block_hash,
        validator_seeds=validator_seeds,
        cycle=phase.cycle_index,
    )

    if not use_secret_salt:
        validation_seed.secret_salt = None

    logger.info(
        "Generated validation seed",
        cycle=phase.cycle_index,
        block=current_block,
        n_validators=len(validator_seeds),
        seed_preview=str(validation_seed.to_int())[:16] + "...",
    )

    return validation_seed


def create_unpredictable_validator(
    config: ValidatorConfig,
    tokenizer: PreTrainedTokenizerBase,
    samples_per_validation: int = 500,
) -> UnpredictableValidator:
    """
    Create UnpredictableValidator instance for this expert group.

    Args:
        config: Validator configuration (contains expert_group_id)
        tokenizer: Tokenizer for dataset formatting
        samples_per_validation: How many examples per validation run

    Returns:
        Configured UnpredictableValidator
    """
    expert_group_id = config.task.expert_group_id

    if expert_group_id not in EXPERT_DATASETS:
        logger.warning(
            f"Unknown expert group {expert_group_id}, using fallback dataset",
        )
        dataset_sources = EXPERT_DATASETS[3]  # Fallback to dummy
    else:
        dataset_sources = EXPERT_DATASETS[expert_group_id]

    validator = UnpredictableValidator(
        expert_group_id=expert_group_id,
        dataset_sources=dataset_sources,
        tokenizer=tokenizer,
        sequence_length=config.task.data.sequence_length,
        samples_per_validation=samples_per_validation,
    )

    logger.info(
        "Created unpredictable validator",
        expert_group_id=expert_group_id,
        n_datasets=len(dataset_sources),
        samples_per_validation=samples_per_validation,
    )

    return validator


def get_unpredictable_validation_dataset(
    config: ValidatorConfig,
    subtensor: bittensor.Subtensor,
    tokenizer: PreTrainedTokenizerBase,
    samples_per_validation: int = 500,
    force_strategy: str | None = None,
) -> tuple[Dataset, ValidationSeed]:
    """
    High-level function to get unpredictable validation dataset.

    This is the main entry point for validator evaluation.

    Args:
        config: Validator configuration
        subtensor: Subtensor instance for blockchain access
        tokenizer: Tokenizer for formatting
        samples_per_validation: Number of validation examples
        force_strategy: Override automatic strategy selection (for testing)

    Returns:
        (validation_dataset, validation_seed) tuple
    """
    # Get cryptographic seed from blockchain
    validation_seed = get_validation_seed_from_chain(config, subtensor)

    # Create validator
    validator = create_unpredictable_validator(
        config,
        tokenizer,
        samples_per_validation,
    )

    # Generate unpredictable dataset
    strategy = None
    if force_strategy:
        from mycelia.shared.unpredictable_validation import ValidationStrategy
        strategy = ValidationStrategy(force_strategy)

    validation_dataset = validator.get_validation_dataset(
        validation_seed,
        strategy=strategy,
    )

    logger.info(
        "Generated validation dataset",
        size=len(validation_dataset),
        cycle=validation_seed.cycle_number,
    )

    return validation_dataset, validation_seed


# Example integration into existing validator code:
"""
# In mycelia/validator/run.py, replace the eval_dataloader creation:

# OLD:
eval_dataloader = get_dataloader(
    config,
    rank=config.local_par.world_size,
    world_size=config.local_par.world_size + 1,
    tokenizer=tokenizer,
)

# NEW:
from mycelia.shared.validation_integration import get_unpredictable_validation_dataset

validation_dataset, validation_seed = get_unpredictable_validation_dataset(
    config=config,
    subtensor=subtensor,
    tokenizer=tokenizer,
    samples_per_validation=500,
)

# Convert to dataloader
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

eval_dataloader = DataLoader(
    validation_dataset,
    batch_size=config.task.data.per_device_train_batch_size,
    collate_fn=data_collator,
)

# Log the seed for reproducibility/debugging
logger.info("Validation seed", seed_int=validation_seed.to_int())
"""

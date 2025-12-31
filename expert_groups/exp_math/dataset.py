"""
Math Expert Dataset

Merges top mathematical reasoning datasets:
- GSM8K, MATH, MetaMathQA, OpenMathInstruct, etc.

Total: ~15M+ examples for anti-cheat protection
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from mycelia.shared.dataloader import DefaultStreamingTorchDataset
from mycelia.shared.merged_dataset import BenchmarkSampler, DatasetSource, MergedStreamingDataset


# =============================================================================
# Dataset Sources Configuration
# =============================================================================

MATH_SOURCES = [
    # Core datasets
    DatasetSource(
        name="openai/gsm8k",
        subset="main",
        split="train",
        weight=2.0,  # High quality, oversample
        text_field="question",
    ),
    DatasetSource(
        name="lighteval/MATH",
        subset="all",
        split="train",
        weight=2.0,
        text_field="problem",
    ),
    DatasetSource(
        name="meta-math/MetaMathQA",
        subset=None,
        split="train",
        weight=1.0,  # Large, normal weight
        text_field="query",
    ),
    # Augmentation datasets
    DatasetSource(
        name="nvidia/OpenMathInstruct-2",
        subset=None,
        split="train",
        weight=0.5,  # Very large, downsample slightly
        text_field="problem",
    ),
    DatasetSource(
        name="TIGER-Lab/MathInstruct",
        subset=None,
        split="train",
        weight=1.0,
        text_field="instruction",
    ),
    # Diverse math
    DatasetSource(
        name="allenai/lila",
        subset=None,
        split="train",
        weight=1.0,
        text_field="input",
    ),
]


# =============================================================================
# Custom Formatters
# =============================================================================

def format_gsm8k(example: dict, tokenizer: PreTrainedTokenizerBase, seq_len: int) -> dict:
    """Format GSM8K with chain-of-thought."""
    question = example.get("question", "")
    answer = example.get("answer", "")
    
    messages = [
        {"role": "user", "content": f"Solve this math problem step by step:\n\n{question}"},
        {"role": "assistant", "content": answer},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    toks = tokenizer(text, truncation=True, max_length=seq_len, padding="max_length")
    
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }


def format_math_problem(example: dict, tokenizer: PreTrainedTokenizerBase, seq_len: int) -> dict:
    """Format MATH dataset with solution."""
    problem = example.get("problem", example.get("question", example.get("query", "")))
    solution = example.get("solution", example.get("answer", example.get("response", "")))
    
    messages = [
        {"role": "user", "content": f"Solve:\n{problem}"},
        {"role": "assistant", "content": solution},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    toks = tokenizer(text, truncation=True, max_length=seq_len, padding="max_length")
    
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }


# =============================================================================
# Main Dataset Class
# =============================================================================

class MergedMathDataset(MergedStreamingDataset):
    """
    Merged math dataset with anti-cheat measures.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        seed: int = 42,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        # Add custom formatters to sources
        sources = MATH_SOURCES.copy()
        for source in sources:
            if source.name == "openai/gsm8k":
                source.format_fn = format_gsm8k
            else:
                source.format_fn = format_math_problem

        super().__init__(
            sources=sources,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            seed=seed,
            holdout_fraction=0.01,  # 1% holdout for benchmark
            rank=rank,
            world_size=world_size,
        )

    @classmethod
    def get_tokenised_dataset(
        cls,
        config,
        tokenizer: PreTrainedTokenizerBase,
        rank: int | None = None,
        world_size: int | None = None,
        train: bool = True,
        seed: str | None = None,
        fraction: float | None = None,
    ):
        """
        Create and return tokenised dataset instance.

        This method is called by the dataloader to instantiate the dataset.
        """
        return cls(
            tokenizer=tokenizer,
            sequence_length=config.task.data.sequence_length,
            seed=int(seed) if seed else 42,
            rank=rank,
            world_size=world_size,
        )


class MathBenchmarkSampler(BenchmarkSampler):
    """
    Math-specific benchmark with anti-cheat perturbations.
    """
    
    def perturb_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Perturb numbers in math problems."""
        text = example.get("text", example.get("question", ""))
        
        # Find all numbers and perturb them slightly
        def perturb_number(match):
            num = float(match.group())
            # Add small random offset
            offset = rng.uniform(-0.1, 0.1) * abs(num) if num != 0 else rng.uniform(-1, 1)
            new_num = num + offset
            # Keep as int if original was int
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


# =============================================================================
# Legacy Compatibility (for existing configs)
# =============================================================================

class StreamingTorchDataset(DefaultStreamingTorchDataset):
    """
    Backward-compatible wrapper.
    
    For new deployments, use MergedMathDataset directly.
    """
    
    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        """Format for training."""
        # Handle various field names
        if "messages" in example:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif "problem" in example:
            problem = example["problem"]
            solution = example.get("solution", "")
            text = f"Problem: {problem}\n\nSolution: {solution}"
        elif "question" in example:
            question = example["question"]
            answer = example.get("answer", "")
            text = f"Question: {question}\n\nAnswer: {answer}"
        elif "query" in example:
            query = example["query"]
            response = example.get("response", "")
            text = f"Query: {query}\n\nResponse: {response}"
        elif "text" in example:
            text = example["text"]
        else:
            # Fallback: concatenate all string fields
            text = " ".join(str(v) for v in example.values() if isinstance(v, str))
        
        toks = tokenizer(
            text,
            truncation=True,
            max_length=sequence_length,
            padding="max_length",
            add_special_tokens=True,
        )
        
        return {
            "input_ids": toks["input_ids"],
            "attention_mask": toks["attention_mask"],
        }

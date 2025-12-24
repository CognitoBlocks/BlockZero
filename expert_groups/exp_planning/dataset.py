"""
Planning Expert Dataset with Uncertainty Decay

Merges multi-step reasoning and planning datasets:
- Chain-of-thought, task decomposition, natural instructions

Total: ~7M+ examples with uncertainty annotations
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from mycelia.shared.dataloader import DefaultStreamingTorchDataset
from mycelia.shared.merged_dataset import BenchmarkSampler, DatasetSource, MergedStreamingDataset

from .uncertainty import UncertaintyTracker, annotate_plan_with_uncertainty


# =============================================================================
# Dataset Sources Configuration
# =============================================================================

PLANNING_SOURCES = [
    # Chain-of-thought
    DatasetSource(
        name="kaist-ai/CoT-Collection",
        subset=None,
        split="train",
        weight=2.0,
        text_field="source",
    ),
    # Task decomposition
    DatasetSource(
        name="google/natural-instructions",
        subset=None,
        split="train",
        weight=1.0,
        text_field="definition",
    ),
    # Multi-step reasoning
    DatasetSource(
        name="allenai/natural-instructions-v2",
        subset=None,
        split="train",
        weight=1.0,
        text_field="definition",
    ),
    # Code planning
    DatasetSource(
        name="bigcode/self-oss-instruct-sc2-exec-filter-50k",
        subset=None,
        split="train",
        weight=1.5,
        text_field="instruction",
    ),
    # Reasoning
    DatasetSource(
        name="Open-Orca/OpenOrca",
        subset=None,
        split="train",
        weight=0.5,  # Large, downsample
        text_field="question",
    ),
    # Real dialogues with planning
    DatasetSource(
        name="allenai/WildChat-1M",
        subset=None,
        split="train",
        weight=0.3,  # Very large
        text_field="conversation",
    ),
]


# =============================================================================
# Custom Formatters with Uncertainty
# =============================================================================

def format_with_uncertainty(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> dict:
    """
    Format planning example with uncertainty annotations.
    """
    tracker = UncertaintyTracker()
    
    # Extract instruction and response
    instruction = example.get("instruction", example.get("definition", example.get("source", "")))
    response = example.get("response", example.get("output", example.get("rationale", "")))
    
    # If no response, check for input/output pairs
    if not response and "input" in example:
        instruction = example.get("instruction", "") + "\n\nInput: " + example.get("input", "")
        response = example.get("output", "")
    
    # Handle conversation format
    if "conversation" in example:
        conv = example["conversation"]
        if isinstance(conv, list) and len(conv) >= 2:
            instruction = conv[0].get("content", "") if isinstance(conv[0], dict) else str(conv[0])
            response = conv[1].get("content", "") if isinstance(conv[1], dict) else str(conv[1])
    
    # Annotate response with uncertainty if it looks like a plan
    if response and has_plan_structure(response):
        response = annotate_plan_with_uncertainty(response, tracker)
    
    # Build messages
    messages = [
        {"role": "system", "content": "You are a planning assistant. Break down complex tasks into steps with confidence markers."},
        {"role": "user", "content": instruction},
    ]
    if response:
        messages.append({"role": "assistant", "content": response})
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not response)
    toks = tokenizer(text, truncation=True, max_length=seq_len, padding="max_length")
    
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }


def has_plan_structure(text: str) -> bool:
    """Check if text looks like a multi-step plan."""
    indicators = [
        r"^\d+\.",  # Numbered lists
        r"^Step \d+",
        r"^First|Second|Third",
        r"^\s*-\s+",  # Bullet points
        r"Then|Next|Finally|After that",
    ]
    lines = text.split("\n")
    step_count = sum(1 for line in lines if any(re.match(p, line.strip(), re.IGNORECASE) for p in indicators))
    return step_count >= 2


def format_cot_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
) -> dict:
    """Format chain-of-thought with step markers."""
    tracker = UncertaintyTracker()
    
    source = example.get("source", "")
    rationale = example.get("rationale", "")
    target = example.get("target", "")
    
    # Add step markers to rationale
    if rationale:
        annotated_rationale = annotate_plan_with_uncertainty(rationale, tracker)
    else:
        annotated_rationale = ""
    
    full_response = annotated_rationale
    if target:
        full_response += f"\n\n[CONF:{tracker.get_confidence_token()[6:-1]}] Final Answer: {target}"
    
    messages = [
        {"role": "user", "content": source},
        {"role": "assistant", "content": full_response},
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

class MergedPlanningDataset(MergedStreamingDataset):
    """
    Merged planning dataset with uncertainty annotations.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        seed: int = 42,
        rank: int | None = None,
        world_size: int | None = None,
    ):
        # Add custom formatters
        sources = []
        for source in PLANNING_SOURCES:
            source_copy = DatasetSource(
                name=source.name,
                subset=source.subset,
                split=source.split,
                weight=source.weight,
                text_field=source.text_field,
                format_fn=format_with_uncertainty,
            )
            # Use CoT formatter for CoT-Collection
            if "CoT" in source.name:
                source_copy.format_fn = format_cot_example
            sources.append(source_copy)
        
        super().__init__(
            sources=sources,
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            seed=seed,
            holdout_fraction=0.01,
            rank=rank,
            world_size=world_size,
        )


class PlanningBenchmarkSampler(BenchmarkSampler):
    """
    Planning-specific benchmark with anti-cheat perturbations.
    """
    
    def perturb_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Shuffle step order and perturb step descriptions."""
        result = example.copy()
        
        for field in ["response", "output", "rationale"]:
            if field in result and result[field]:
                text = result[field]
                
                # Find numbered steps
                step_pattern = r"(\d+)\.\s*(.+?)(?=\d+\.|$)"
                steps = re.findall(step_pattern, text, re.DOTALL)
                
                if len(steps) >= 3:
                    # Shuffle middle steps (keep first and last)
                    first_step = steps[0]
                    last_step = steps[-1]
                    middle_steps = steps[1:-1]
                    
                    rng.shuffle(middle_steps)
                    
                    # Rebuild
                    shuffled = [first_step] + list(middle_steps) + [last_step]
                    new_text = ""
                    for i, (_, content) in enumerate(shuffled, 1):
                        new_text += f"{i}. {content.strip()}\n"
                    
                    result[field] = new_text
                break
        
        return result


# =============================================================================
# Legacy Compatibility
# =============================================================================

class StreamingTorchDataset(DefaultStreamingTorchDataset):
    """Backward-compatible wrapper."""
    
    @staticmethod
    def tokenize_and_format(
        example: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
    ) -> dict[str, Any]:
        """Format for training with uncertainty."""
        tracker = UncertaintyTracker()
        
        # Get text from various fields
        text_fields = ["instruction", "definition", "source", "question", "input"]
        response_fields = ["response", "output", "rationale", "answer", "target"]
        
        instruction = ""
        for field in text_fields:
            if field in example and example[field]:
                instruction = str(example[field])
                break
        
        response = ""
        for field in response_fields:
            if field in example and example[field]:
                response = str(example[field])
                break
        
        # Annotate if plan-like
        if response and has_plan_structure(response):
            response = annotate_plan_with_uncertainty(response, tracker)
        
        # Format
        if instruction and response:
            text = f"Task: {instruction}\n\nPlan:\n{response}"
        elif instruction:
            text = f"Task: {instruction}\n\nPlan:"
        else:
            text = response or str(example)
        
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


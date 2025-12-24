"""
Agentic Expert Dataset

Merges top tool-use and function-calling datasets:
- Glaive, xLAM, Gorilla, ToolBench, AgentInstruct, etc.

Total: ~800K+ examples for anti-cheat protection
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase

from mycelia.shared.dataloader import DefaultStreamingTorchDataset
from mycelia.shared.merged_dataset import BenchmarkSampler, DatasetSource, MergedStreamingDataset


# =============================================================================
# Dataset Sources Configuration
# =============================================================================

AGENTIC_SOURCES = [
    # Function calling
    DatasetSource(
        name="glaiveai/glaive-function-calling-v2",
        subset=None,
        split="train",
        weight=2.0,  # High quality
        text_field="system_prompt",
    ),
    DatasetSource(
        name="Salesforce/xlam-function-calling-60k",
        subset=None,
        split="train",
        weight=2.0,
        text_field="query",
    ),
    # API/Tool use
    DatasetSource(
        name="NousResearch/hermes-function-calling-v1",
        subset=None,
        split="train",
        weight=1.5,
        text_field="conversations",
    ),
    # Agent behavior
    DatasetSource(
        name="THUDM/AgentInstruct",
        subset=None,
        split="train",
        weight=1.0,
        text_field="conversations",
    ),
    # Structured output
    DatasetSource(
        name="NousResearch/json-mode-eval",
        subset=None,
        split="train",
        weight=1.0,
        text_field="prompt",
    ),
]


# =============================================================================
# Custom Formatters
# =============================================================================

def format_function_call(example: dict, tokenizer: PreTrainedTokenizerBase, seq_len: int) -> dict:
    """Format function calling examples."""
    
    # Handle different field structures
    if "conversations" in example:
        messages = []
        for conv in example["conversations"]:
            role = conv.get("from", conv.get("role", "user"))
            content = conv.get("value", conv.get("content", ""))
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"
            messages.append({"role": role, "content": content})
    else:
        # Build from individual fields
        system = example.get("system_prompt", example.get("system", ""))
        query = example.get("query", example.get("user", example.get("instruction", "")))
        response = example.get("response", example.get("assistant", example.get("output", "")))
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})
        if response:
            messages.append({"role": "assistant", "content": response})
    
    # Add tool use markers if tools are present
    if "tools" in example or "functions" in example:
        tools = example.get("tools", example.get("functions", []))
        if tools and messages:
            tool_desc = json.dumps(tools, indent=2) if isinstance(tools, (list, dict)) else str(tools)
            messages[0]["content"] = f"Available tools:\n```json\n{tool_desc}\n```\n\n" + messages[0]["content"]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    toks = tokenizer(text, truncation=True, max_length=seq_len, padding="max_length")
    
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }


def format_agent_trace(example: dict, tokenizer: PreTrainedTokenizerBase, seq_len: int) -> dict:
    """Format ReAct-style agent traces."""
    
    # Build thought-action-observation chain
    if "trajectory" in example:
        trajectory = example["trajectory"]
        text = "Agent Trace:\n\n"
        for step in trajectory:
            if "thought" in step:
                text += f"Thought: {step['thought']}\n"
            if "action" in step:
                text += f"Action: {step['action']}\n"
            if "observation" in step:
                text += f"Observation: {step['observation']}\n"
            text += "\n"
    else:
        # Fallback to simple format
        text = format_function_call(example, tokenizer, seq_len)
        return text
    
    messages = [
        {"role": "user", "content": example.get("task", "Complete the task.")},
        {"role": "assistant", "content": text},
    ]
    
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    toks = tokenizer(formatted_text, truncation=True, max_length=seq_len, padding="max_length")
    
    return {
        "input_ids": toks["input_ids"],
        "attention_mask": toks["attention_mask"],
    }


# =============================================================================
# Main Dataset Class
# =============================================================================

class MergedAgenticDataset(MergedStreamingDataset):
    """
    Merged agentic dataset with anti-cheat measures.
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
        for source in AGENTIC_SOURCES:
            source_copy = DatasetSource(
                name=source.name,
                subset=source.subset,
                split=source.split,
                weight=source.weight,
                text_field=source.text_field,
                format_fn=format_function_call,
            )
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


class AgenticBenchmarkSampler(BenchmarkSampler):
    """
    Agentic-specific benchmark with anti-cheat perturbations.
    """
    
    # Fake tool names for perturbation
    FAKE_TOOLS = [
        "query_database_v2", "fetch_user_profile", "send_notification_async",
        "calculate_metrics", "validate_input_schema", "transform_data_pipeline",
        "invoke_external_api", "cache_result", "log_event_stream",
    ]
    
    def perturb_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Randomize tool/function names."""
        result = example.copy()
        
        # Replace tool names with random alternatives
        if "tools" in result:
            tools = result["tools"]
            if isinstance(tools, list):
                for tool in tools:
                    if isinstance(tool, dict) and "name" in tool:
                        # Append random suffix to tool name
                        suffix = rng.choice(["_v2", "_new", "_alt", "_beta"])
                        tool["name"] = tool["name"] + suffix
        
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
        """Format for training."""
        
        # Handle conversations format
        if "conversations" in example:
            messages = []
            for conv in example["conversations"]:
                role = conv.get("from", conv.get("role", "user"))
                content = conv.get("value", conv.get("content", ""))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "model"):
                    role = "assistant"
                messages.append({"role": role, "content": content})
            
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            # Concatenate available fields
            parts = []
            for field in ["system_prompt", "query", "instruction", "response", "output"]:
                if field in example and example[field]:
                    parts.append(f"{field}: {example[field]}")
            text = "\n\n".join(parts) if parts else str(example)
        
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


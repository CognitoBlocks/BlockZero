"""
Merged Dataset Infrastructure

Combines multiple HuggingFace datasets with:
- Weighted sampling
- Anti-cheat benchmark holdout
- Dynamic validation sampling
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np
import torch
from datasets import Dataset, IterableDataset, interleave_datasets, load_dataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import PreTrainedTokenizerBase

from mycelia.shared.app_logging import structlog

logger = structlog.get_logger(__name__)


@dataclass
class DatasetSource:
    """Configuration for a single dataset source."""
    name: str  # HuggingFace dataset name
    subset: str | None = None  # Dataset config/subset
    split: str = "train"
    weight: float = 1.0  # Sampling weight relative to others
    text_field: str = "text"  # Field containing text (or "messages" for chat)
    format_fn: Callable | None = None  # Custom formatter


class MergedStreamingDataset(TorchIterableDataset):
    """
    Merges multiple HuggingFace datasets into a single streaming dataset.
    
    Features:
    - Weighted interleaving of sources
    - Holdout benchmark extraction
    - Per-epoch random validation sampling
    """
    
    def __init__(
        self,
        sources: list[DatasetSource],
        tokenizer: PreTrainedTokenizerBase,
        sequence_length: int,
        seed: int = 42,
        holdout_fraction: float = 0.01,  # 1% for benchmark
        rank: int | None = None,
        world_size: int | None = None,
    ):
        self.sources = sources
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.seed = seed
        self.holdout_fraction = holdout_fraction
        self.rank = rank
        self.world_size = world_size
        
        self._datasets: list[IterableDataset] = []
        self._benchmark_hashes: set[str] = set()
        
    def _load_sources(self) -> list[IterableDataset]:
        """Load all dataset sources as streaming datasets."""
        datasets = []
        weights = []
        
        for source in self.sources:
            try:
                logger.info(f"Loading dataset: {source.name}", subset=source.subset)
                ds = load_dataset(
                    source.name,
                    source.subset,
                    split=source.split,
                    streaming=True,
                    trust_remote_code=True,
                )
                datasets.append(ds)
                weights.append(source.weight)
            except Exception as e:
                logger.warning(f"Failed to load {source.name}: {e}")
                continue
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Normalize weights
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        # Interleave with probabilities
        merged = interleave_datasets(
            datasets,
            probabilities=probs,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )
        
        return merged
    
    def _hash_example(self, example: dict) -> str:
        """Create deterministic hash for deduplication/holdout."""
        # Use first 100 chars of text content
        text = str(example.get("text", example.get("messages", "")))[:100]
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def _is_holdout(self, example: dict) -> bool:
        """Deterministically decide if example is in holdout set."""
        h = self._hash_example(example)
        # Use hash to deterministically select holdout
        hash_val = int(h, 16) / (16 ** 16)
        return hash_val < self.holdout_fraction
    
    def _format_example(self, example: dict, source_idx: int) -> dict | None:
        """Format example for tokenization."""
        source = self.sources[source_idx % len(self.sources)]
        
        # Skip holdout examples during training
        if self._is_holdout(example):
            self._benchmark_hashes.add(self._hash_example(example))
            return None
        
        # Apply custom formatter if provided
        if source.format_fn:
            return source.format_fn(example, self.tokenizer, self.sequence_length)
        
        # Default: handle both text and chat formats
        if "messages" in example:
            text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=True,
            )
        elif source.text_field in example:
            text = example[source.text_field]
        else:
            # Try common field names
            for field in ["text", "content", "input", "instruction"]:
                if field in example:
                    text = example[field]
                    break
            else:
                return None
        
        # Tokenize
        toks = self.tokenizer(
            text,
            truncation=True,
            max_length=self.sequence_length,
            padding="max_length",
            add_special_tokens=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "labels": toks["input_ids"].squeeze(0).clone(),
        }
    
    def __iter__(self) -> Iterator[dict]:
        merged = self._load_sources()
        
        # Shard for distributed training
        if self.rank is not None and self.world_size is not None:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # Multi-worker within single process
                per_worker = self.world_size * worker_info.num_workers
                worker_id = self.rank * worker_info.num_workers + worker_info.id
            else:
                per_worker = self.world_size
                worker_id = self.rank
        else:
            per_worker = 1
            worker_id = 0
        
        for idx, example in enumerate(merged):
            # Shard across workers
            if idx % per_worker != worker_id:
                continue
            
            formatted = self._format_example(example, idx)
            if formatted is not None:
                yield formatted


class BenchmarkSampler:
    """
    Generates random validation samples from holdout set.
    
    Anti-cheat: Different sample each epoch, never same twice.
    """
    
    def __init__(
        self,
        holdout_dataset: Dataset,
        samples_per_epoch: int = 1000,
    ):
        self.holdout = holdout_dataset
        self.samples_per_epoch = samples_per_epoch
        self._epoch = 0
    
    def get_epoch_sample(self, epoch: int | None = None) -> Dataset:
        """Get random sample for this epoch."""
        if epoch is None:
            epoch = self._epoch
            self._epoch += 1
        
        rng = np.random.default_rng(seed=42 + epoch * 7919)  # Different each epoch
        n_samples = min(self.samples_per_epoch, len(self.holdout))
        indices = rng.choice(len(self.holdout), size=n_samples, replace=False)
        
        return self.holdout.select(indices.tolist())
    
    def perturb_example(self, example: dict, rng: np.random.Generator) -> dict:
        """Apply anti-cheat perturbations."""
        # Override in subclasses for domain-specific perturbations
        return example


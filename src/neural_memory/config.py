"""Configuration management for Neural Memory."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DecayConfig:
    base_decay_rate: float = 0.15
    stability_growth_on_review: float = 0.3
    max_stability: float = 10.0
    prune_threshold: float = 0.05
    importance_shield: float = 0.3
    emotional_arousal_shield: float = 0.5


@dataclass
class ActivationConfig:
    initial_activation: float = 1.0
    decay_per_hop: float = 0.6
    min_activation: float = 0.05
    max_hops: int = 4
    max_nodes_visited: int = 200
    top_k: int = 20


@dataclass
class RankerConfig:
    semantic_weight: float = 0.40
    fts_weight: float = 0.15
    activation_weight: float = 0.30
    recency_weight: float = 0.10
    strength_weight: float = 0.05


@dataclass
class WorkingMemoryConfig:
    capacity: int = 20
    ttl_hours: float = 2.0
    promotion_strength_threshold: float = 0.6
    short_term_to_long_term_hours: float = 48.0
    short_term_access_threshold: int = 3


@dataclass
class ConsolidationConfig:
    similarity_threshold: float = 0.82
    min_cluster_size: int = 3
    max_cluster_size: int = 15


@dataclass
class Config:
    """Top-level configuration."""

    storage_dir: str = "./storage"
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    log_level: str = "INFO"

    decay: dict[str, Any] = field(default_factory=dict)
    activation: dict[str, Any] = field(default_factory=dict)
    ranker: dict[str, Any] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    consolidation: dict[str, Any] = field(default_factory=dict)

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)

    @property
    def db_path(self) -> Path:
        return self.storage_path / "neural_memory.db"

    @property
    def files_path(self) -> Path:
        return self.storage_path / "files"

    @property
    def decay_config(self) -> DecayConfig:
        return DecayConfig(**self.decay)

    @property
    def activation_config(self) -> ActivationConfig:
        return ActivationConfig(**self.activation)

    @property
    def ranker_config(self) -> RankerConfig:
        return RankerConfig(**self.ranker)

    @property
    def working_memory_config(self) -> WorkingMemoryConfig:
        return WorkingMemoryConfig(**self.working_memory)

    @property
    def consolidation_config(self) -> ConsolidationConfig:
        return ConsolidationConfig(**self.consolidation)


def load_config(config_path: str | None = None) -> Config:
    """Load config from YAML file with environment variable overrides."""
    base: dict[str, Any] = {}

    # Default: look for config.yaml next to the package root (project dir)
    _pkg_dir = Path(__file__).resolve().parent  # src/neural_memory/
    _project_dir = _pkg_dir.parent.parent       # neural-memory-mcp/
    default_path = str(_project_dir / "config.yaml")

    path = config_path or os.environ.get("NEURAL_MEMORY_CONFIG", default_path)
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    # Environment variable overrides
    if env_storage := os.environ.get("NEURAL_MEMORY_STORAGE_DIR"):
        base["storage_dir"] = env_storage
    if env_model := os.environ.get("NEURAL_MEMORY_EMBEDDING_MODEL"):
        base["embedding_model"] = env_model
    if env_log := os.environ.get("NEURAL_MEMORY_LOG_LEVEL"):
        base["log_level"] = env_log

    return Config(**base)

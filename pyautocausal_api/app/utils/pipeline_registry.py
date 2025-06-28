from pathlib import Path
import yaml
from functools import lru_cache
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

_YAML_FILE = Path(__file__).resolve().parent.parent / "deployed_pipelines.yaml"

@lru_cache(maxsize=1)
def load_deployed_pipelines() -> Dict[str, Dict[str, List[str]]]:
    """Load deployed_pipelines.yaml and return mapping.

    Returns
    -------
    dict
        {pipeline_name: {"required_columns": [...], "optional_columns": [...]}}
    """
    if not _YAML_FILE.exists():
        logger.error("deployed_pipelines.yaml not found at %s", _YAML_FILE)
        return {}
    try:
        with _YAML_FILE.open("r") as f:
            raw_yaml = yaml.safe_load(f)
    except Exception as exc:
        logger.error("Failed to read deployed_pipelines.yaml: %s", exc)
        return {}
    mapping: Dict[str, Dict[str, List[str]]] = {}
    if isinstance(raw_yaml, list):
        for entry in raw_yaml:
            if isinstance(entry, dict):
                mapping.update(entry)
    elif isinstance(raw_yaml, dict):
        mapping = raw_yaml
    # Ensure lists exist
    for name, cfg in mapping.items():
        mapping[name] = {
            "required_columns": cfg.get("required_columns", []) or [],
            "optional_columns": cfg.get("optional_columns", []) or [],
            "graph_uri": cfg.get("graph_uri"),
        }
    return mapping 
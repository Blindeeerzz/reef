"""Parsing and validation for recipe YAML configuration and environment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from reef.recipe.errors import RecipeConfigError


def load_recipe_config(path: str | Path) -> dict[str, Any]:
    """Load a recipe file in the public layout and return its flat recipe config."""
    config_path = Path(path)
    try:
        loaded = yaml.safe_load(config_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RecipeConfigError(f"cannot load recipe config {config_path}: {exc}") from exc
    return public_recipe_config(loaded)


def recipe_config_from_mapping(loaded: Any) -> dict[str, Any]:
    """Validate a flat recipe config: a composite component or a deployment's resolved recipe settings."""
    if not isinstance(loaded, Mapping):
        raise RecipeConfigError("recipe config must be a YAML object")
    config = dict(loaded)
    if not isinstance(config.get("implementation"), str) or not config["implementation"]:
        raise RecipeConfigError("recipe config must contain a non-empty 'implementation'")
    for section in ("data", "artifact", "runtime", "model", "rollout", "optimization"):
        value = config.get(section, {})
        if not isinstance(value, Mapping):
            raise RecipeConfigError(f"recipe config '{section}' must be an object")
        config[section] = dict(value)
    return config


def config_positive_int(config: Mapping[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RecipeConfigError(f"{name} must be a positive integer")
    return value


def public_recipe_config(loaded: Any) -> dict[str, Any]:
    """Read a recipe file's public sections into a flat recipe config.

    The file selects its class under ``recipe.implementation``, its fields under
    ``recipe.config`` and its runtime under ``recipe.runtime``. A standalone
    file names its model in the ``inference`` section; a deployment supplies
    its already resolved runtime instead.
    """
    from reef.recipe.registry import recipe_class_for

    if not isinstance(loaded, Mapping):
        raise RecipeConfigError("recipe config must be a YAML object")
    config = dict(loaded)
    if "service" in config or "services" in config:
        raise RecipeConfigError("service/services are not accepted; use reef settings and automatic assembly")
    if isinstance(config.get("execution"), Mapping) and "services" in config["execution"]:
        raise RecipeConfigError("execution.services is not accepted; configure component resources instead")
    recipe = config.get("recipe")
    if not isinstance(recipe, Mapping) or not isinstance(recipe.get("implementation"), str):
        raise RecipeConfigError(
            "recipe.implementation must name a recipe class; "
            "see the configuration layout in docs/reference/configuration.rst"
        )
    recipe_type = recipe_class_for(recipe["implementation"])
    if recipe_type is None:
        raise RecipeConfigError("recipe.implementation must name a recipe class")
    supplied = recipe.get("config", {})
    if not isinstance(supplied, Mapping):
        raise RecipeConfigError("recipe.config must be an object")
    data: dict[str, Any] = {}
    resolved: dict[str, Any] = {"implementation": recipe["implementation"], "data": data}
    for name, value in supplied.items():
        if not isinstance(name, str):
            raise RecipeConfigError("recipe config field names must be strings")
        name = name.replace("-", "_")
        target = resolved if name in recipe_type.config_sections else data
        if name in target:
            raise RecipeConfigError(f"duplicate recipe config field: {name}")
        target[name] = value
    for section in recipe_type.config_sections:
        if section in resolved and not isinstance(resolved[section], Mapping):
            raise RecipeConfigError(f"recipe.config.{section} must be an object")
    if "runtime" in recipe:
        resolved["runtime"] = recipe["runtime"]
    for section in ("execution", "executors", "pipeline"):
        if section in config:
            resolved[section] = config[section]
    # A deployment provides its already resolved runtime. Standalone recipe
    # files carry the model identifier for their embedding driver instead.
    inference = config.get("inference", {})
    if not isinstance(inference, Mapping):
        raise RecipeConfigError("inference must be an object")
    if "reef" not in config:
        # An explicit model, even null, is kept so that it is not replaced by the deployment's model.
        supplied_names = [
            name for name in ("upstream-model", "upstream_model", "model-path", "model_path") if name in inference
        ]
        if supplied_names:
            resolved["model"] = {"path": inference[supplied_names[0]]}
    return recipe_config_from_mapping(resolved)

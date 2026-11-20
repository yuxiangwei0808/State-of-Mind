"""Restricted loader for the trusted local TRIBEv2 configuration file."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping

import yaml


class StimulusConfigError(ValueError):
    """Raised when the local feature configuration cannot be read safely."""


class _RestrictedConfigLoader(yaml.SafeLoader):
    """Accept only the tuple/path tags emitted by TRIBEv2 and Exca."""


def _construct_python_tuple(
    loader: _RestrictedConfigLoader,
    node: Any,
) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node, deep=True))


def _construct_posix_path(loader: _RestrictedConfigLoader, node: Any) -> str:
    parts = loader.construct_sequence(node, deep=True)
    return str(PurePosixPath(*(str(part) for part in parts)))


def _construct_windows_path(loader: _RestrictedConfigLoader, node: Any) -> str:
    parts = loader.construct_sequence(node, deep=True)
    return str(PureWindowsPath(*(str(part) for part in parts)))


_RestrictedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    _construct_python_tuple,
)
for _path_class in ("Path", "PurePath", "PosixPath", "PurePosixPath"):
    _RestrictedConfigLoader.add_constructor(
        f"tag:yaml.org,2002:python/object/apply:pathlib.{_path_class}",
        _construct_posix_path,
    )
for _path_class in ("WindowsPath", "PureWindowsPath"):
    _RestrictedConfigLoader.add_constructor(
        f"tag:yaml.org,2002:python/object/apply:pathlib.{_path_class}",
        _construct_windows_path,
    )


def load_checkpoint_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    """Load known TRIBEv2 YAML values without arbitrary object construction."""

    source = Path(path)
    try:
        value = yaml.load(
            source.read_text(encoding="utf-8"),
            Loader=_RestrictedConfigLoader,
        )
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StimulusConfigError(
            f"Cannot parse checkpoint YAML {source}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise StimulusConfigError(
            f"Expected a YAML mapping in {source}, got {type(value).__name__}"
        )
    return value


__all__ = ["StimulusConfigError", "load_checkpoint_yaml_mapping"]

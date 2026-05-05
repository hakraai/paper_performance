from __future__ import annotations

from datetime import date, datetime
from pathlib import Path


def normalize_timeframe_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raise ValueError(f"Expected ISO date string or YAML date value, got {value!r}")


def require_timeframe_config(
    config: dict[str, object],
    key: str,
    config_path: Path,
) -> list[str]:
    value = config.get(key)
    if value is None:
        raise ValueError(f"Missing required timeframe setting '{key}' in {config_path}")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(
            f"Expected '{key}' in {config_path} to be a list of two ISO date strings, got {value!r}"
        )
    return [normalize_timeframe_value(item) for item in value]


def build_perspective_filter_attrs(
    base_attrs: dict[str, object],
    timeframe_testing: list[str],
    timeframe_forecast: list[str],
) -> dict[str, dict[str, object]]:
    retrospective_filter_attrs = dict(base_attrs)
    retrospective_filter_attrs["timeframe"] = timeframe_forecast

    prospective_filter_attrs = dict(base_attrs)
    prospective_filter_attrs["timeframe"] = timeframe_testing

    time_series_filter_attrs = dict(base_attrs)
    time_series_filter_attrs["timeframe"] = timeframe_forecast

    return {
        "retrospective": retrospective_filter_attrs,
        "prospective": prospective_filter_attrs,
        "time_series": time_series_filter_attrs,
    }


__all__ = [
    "build_perspective_filter_attrs",
    "normalize_timeframe_value",
    "require_timeframe_config",
]
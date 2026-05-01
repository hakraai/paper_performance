from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import matplotlib as mpl
import yaml

from run_performance_assessment import (
    DEFAULT_MODEL_NAMES,
    REPO_ROOT,
    build_testsuite_artifact,
    collect_calibrations,
    filter_from_attrs,
    generate_adaptive_analysis_plots_all_models,
    generate_bs_basis_plot,
    generate_bs_spatial_plot,
    generate_cell_covering_plot,
    generate_groningen_three_panel_plot,
    generate_loo_plot,
    generate_model_time_series_plot,
    generate_spatial_performance_plots,
    get_assessment_path,
    get_cell_covering_path,
    has_perspective_models,
    load_assessment,
    load_cell_covering,
    load_context,
    log_skipped_figure,
    maybe_render_formats,
    normalize_figure_formats,
    plot_likelihood_tests_2x4,
    resolve_path,
)
from workflow_support.logging import configure_logging, get_logger
from workflow_support.paths import default_model_specs_path, default_source_data_root


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render figures from cached assessment artifacts without running the assessment workflow."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "figure_generation.yaml",
        help="YAML configuration file.",
    )
    parser.add_argument(
        "--cache",
        choices=["error", "reuse", "refresh"],
        default="reuse",
        help="How to handle existing figure outputs: error, reuse them, or refresh them.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def get_cache_mode(args: argparse.Namespace) -> str:
    return "refresh" if args.force else args.cache


def configure_export_style(figure_dpi: int, config: dict[str, object]) -> None:
    pdf_fonttype = int(config.get("pdf_fonttype", 42))
    ps_fonttype = int(config.get("ps_fonttype", 42))
    line_joinstyle = str(config.get("line_joinstyle", "round"))
    mpl.rcParams.update(
        {
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.dpi": figure_dpi,
            "savefig.dpi": figure_dpi,
            "savefig.transparent": False,
            "pdf.fonttype": pdf_fonttype,
            "ps.fonttype": ps_fonttype,
            "ps.useafm": False,
            "legend.framealpha": 1.0,
            "lines.solid_joinstyle": line_joinstyle,
            "lines.dash_joinstyle": line_joinstyle,
        }
    )


def render_time_series_figure(
    perspective: str,
    model_ids: list[str],
    calibrations: dict[str, dict[str, az.InferenceData]],
    model_specs: dict[str, dict[str, object]],
    grid_data,
    event_data,
    filterset,
    model_names: dict[str, str],
    figure_dir: Path,
    timeframe_plotting: list[str],
    figure_formats: list[str],
) -> None:
    suites = {
        model_id: build_testsuite_artifact(
            model_id,
            calibrations[perspective][model_id],
            model_specs,
            grid_data,
            event_data,
            filterset,
        )
        for model_id in model_ids
    }
    generate_model_time_series_plot(
        model_ids,
        f"timeseries_{perspective}",
        suites,
        model_names,
        figure_dir,
        timeframe_plotting,
        figure_formats,
    )


def main() -> None:
    args = parse_args()
    configure_logging()
    config = yaml.safe_load(args.config.read_text())
    cache_mode = get_cache_mode(args)

    repo_root = resolve_path(REPO_ROOT, config.get("repo_root")) or REPO_ROOT
    source_data_root = resolve_path(repo_root, config.get("source_data_root")) or default_source_data_root(repo_root)
    calibration_root = resolve_path(repo_root, config.get("calibration_root")) or (repo_root / "data" / "generated_calibrations")
    artifact_dir = resolve_path(repo_root, config.get("artifact_dir")) or (repo_root / "data" / "generated_assessment")
    figure_dir = resolve_path(repo_root, config.get("figure_dir")) or (repo_root / "figures" / "generated_paper")
    experiment = config.get("experiment", "groningen_1995_2025")
    model_specs_file = resolve_path(repo_root, config.get("model_specs_file")) or default_model_specs_path(repo_root, experiment)
    model_specs = yaml.safe_load(model_specs_file.read_text())
    perspectives = config.get("perspectives", ["prospective", "retrospective"])
    model_names = config.get("model_names", DEFAULT_MODEL_NAMES)
    model_ids = config.get("models", list(model_names.keys()))
    timeframe_testing = config.get(
        "timeframe_testing",
        config.get("timeframe_prospective_testing", ["2020-10-01", "2025-10-01"]),
    )
    timeframe_forecast = config.get("timeframe_forecast")
    timeframe_plotting = config.get("timeframe_plotting", ["1995-01-01", "2028-01-01"])
    figure_formats = normalize_figure_formats(config.get("figure_formats", config.get("figure_extension")))
    figure_dpi = int(config.get("figure_dpi", 600))

    LOGGER.info(
        "stage=figures configured experiment=%s perspectives=%s models=%s cache=%s artifact_dir=%s figure_dir=%s",
        experiment,
        perspectives,
        model_ids,
        cache_mode,
        artifact_dir,
        figure_dir,
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    configure_export_style(figure_dpi, config)

    groningen_contour, event_data, fault_data, grid_data, _ = load_context(source_data_root)
    rc = collect_calibrations(
        calibration_root,
        experiment,
        perspectives,
        model_ids,
    )

    forecast_source = "retrospective" if "retrospective" in rc else "prospective"
    forecast_filter_attrs = dict(rc[forecast_source][model_ids[0]].attrs)
    retrospective_filter_attrs = dict(forecast_filter_attrs)
    if timeframe_forecast is not None:
        retrospective_filter_attrs["timeframe"] = timeframe_forecast
    prospective_filter_attrs = dict(forecast_filter_attrs)
    prospective_filter_attrs["timeframe"] = timeframe_testing
    time_series_filter_attrs = dict(forecast_filter_attrs)
    if timeframe_forecast is not None:
        time_series_filter_attrs["timeframe"] = timeframe_forecast
    time_series_filterset = filter_from_attrs(time_series_filter_attrs).sel(purpose="calibration")
    LOGGER.info(
        "stage=figures status=filters retrospective_timeframe=%s prospective_timeframe=%s time_series_timeframe=%s",
        retrospective_filter_attrs["timeframe"],
        prospective_filter_attrs["timeframe"],
        time_series_filter_attrs["timeframe"],
    )

    cell_covering_path = get_cell_covering_path(artifact_dir, experiment)
    if not cell_covering_path.exists():
        raise FileNotFoundError(f"Missing required cached artifact: {cell_covering_path}")
    cell_covering = load_cell_covering(cell_covering_path)

    temporal_results = {perspective: {} for perspective in perspectives}
    spatial_results = {perspective: {} for perspective in perspectives}
    adaptive_results = {perspective: {} for perspective in perspectives}

    for perspective in perspectives:
        for model_id in model_ids:
            assessment_path = get_assessment_path(artifact_dir, experiment, perspective, model_id)
            if not assessment_path.exists():
                raise FileNotFoundError(f"Missing required cached assessment artifact: {assessment_path}")
            temporal, spatial, adaptive = load_assessment(assessment_path)
            temporal_results[perspective][model_id] = temporal
            spatial_results[perspective][model_id] = spatial
            adaptive_results[perspective][model_id] = adaptive

    maybe_render_formats(
        figure_dir,
        "bs_basis",
        figure_formats,
        cache_mode,
        lambda: generate_bs_basis_plot(fault_data, figure_dir, figure_formats),
    )
    maybe_render_formats(
        figure_dir,
        "bs_spatial_4",
        figure_formats,
        cache_mode,
        lambda: generate_bs_spatial_plot(grid_data, groningen_contour, figure_dir, figure_formats),
    )
    maybe_render_formats(
        figure_dir,
        "groningen_three_panel",
        figure_formats,
        cache_mode,
        lambda: generate_groningen_three_panel_plot(event_data, fault_data, grid_data, groningen_contour, figure_dir, figure_formats),
    )
    if cell_covering is not None and has_perspective_models(adaptive_results, "retrospective", [model_ids[0]]):
        maybe_render_formats(
            figure_dir,
            "cell_covering",
            figure_formats,
            cache_mode,
            lambda: generate_cell_covering_plot(cell_covering, adaptive_results, model_ids, groningen_contour, figure_dir, figure_formats),
        )
    else:
        log_skipped_figure("cell_covering", "missing-cell-covering-or-retrospective-adaptive-stats")
    if has_perspective_models(rc, "retrospective", model_ids):
        maybe_render_formats(
            figure_dir,
            "timeseries_retrospective",
            figure_formats,
            cache_mode,
            lambda: render_time_series_figure(
                "retrospective",
                model_ids,
                rc,
                model_specs,
                grid_data,
                event_data,
                time_series_filterset,
                model_names,
                figure_dir,
                timeframe_plotting,
                figure_formats,
            ),
        )
    else:
        log_skipped_figure("timeseries_retrospective", "missing-retrospective-calibrations")
    if has_perspective_models(rc, "prospective", model_ids):
        maybe_render_formats(
            figure_dir,
            "timeseries_prospective",
            figure_formats,
            cache_mode,
            lambda: render_time_series_figure(
                "prospective",
                model_ids,
                rc,
                model_specs,
                grid_data,
                event_data,
                time_series_filterset,
                model_names,
                figure_dir,
                timeframe_plotting,
                figure_formats,
            ),
        )
    else:
        log_skipped_figure("timeseries_prospective", "missing-prospective-calibrations")
    if has_perspective_models(temporal_results, "prospective", model_ids) and has_perspective_models(temporal_results, "retrospective", model_ids):
        maybe_render_formats(
            figure_dir,
            "csep_temporal_2x4",
            figure_formats,
            cache_mode,
            lambda: plot_likelihood_tests_2x4(
                model_ids,
                "csep_temporal_2x4",
                temporal_results["prospective"],
                temporal_results["retrospective"],
                model_names,
                figure_dir,
                figure_formats,
            ),
        )
    else:
        log_skipped_figure("csep_temporal_2x4", "missing-temporal-assessment-artifacts")
    if has_perspective_models(spatial_results, "retrospective", [model_ids[0], model_ids[1]]):
        maybe_render_formats(
            figure_dir,
            "multi_retrospective",
            figure_formats,
            cache_mode,
            lambda: generate_spatial_performance_plots(
                "multi_retrospective",
                model_ids[0],
                model_ids[1],
                spatial_results["retrospective"],
                model_names,
                groningen_contour,
                figure_dir,
                figure_formats,
                title="retrospective multiresolution spatial test",
            ),
        )
    else:
        log_skipped_figure("multi_retrospective", "missing-retrospective-spatial-artifacts")
    if has_perspective_models(spatial_results, "retrospective", [model_ids[2], model_ids[3]]):
        maybe_render_formats(
            figure_dir,
            "multi_bs_retrospective",
            figure_formats,
            cache_mode,
            lambda: generate_spatial_performance_plots(
                "multi_bs_retrospective",
                model_ids[2],
                model_ids[3],
                spatial_results["retrospective"],
                model_names,
                groningen_contour,
                figure_dir,
                figure_formats,
                title="retrospective multiresolution spatial test",
            ),
        )
    else:
        log_skipped_figure("multi_bs_retrospective", "missing-retrospective-bs-spatial-artifacts")
    if has_perspective_models(spatial_results, "prospective", [model_ids[0], model_ids[1]]):
        maybe_render_formats(
            figure_dir,
            "multi_prospective",
            figure_formats,
            cache_mode,
            lambda: generate_spatial_performance_plots(
                "multi_prospective",
                model_ids[0],
                model_ids[1],
                spatial_results["prospective"],
                model_names,
                groningen_contour,
                figure_dir,
                figure_formats,
                title="prospective multiresolution spatial test",
            ),
        )
    else:
        log_skipped_figure("multi_prospective", "missing-prospective-spatial-artifacts")
    if has_perspective_models(spatial_results, "prospective", [model_ids[2], model_ids[3]]):
        maybe_render_formats(
            figure_dir,
            "multi_bs_prospective",
            figure_formats,
            cache_mode,
            lambda: generate_spatial_performance_plots(
                "multi_bs_prospective",
                model_ids[2],
                model_ids[3],
                spatial_results["prospective"],
                model_names,
                groningen_contour,
                figure_dir,
                figure_formats,
                title="prospective multiresolution spatial test",
            ),
        )
    else:
        log_skipped_figure("multi_bs_prospective", "missing-prospective-bs-spatial-artifacts")
    if has_perspective_models(adaptive_results, "prospective", model_ids):
        maybe_render_formats(
            figure_dir,
            "adaptive_all_prospective",
            figure_formats,
            cache_mode,
            lambda: generate_adaptive_analysis_plots_all_models(
                model_ids,
                "adaptive_all_prospective",
                adaptive_results["prospective"],
                model_names,
                groningen_contour,
                figure_dir,
                figure_formats,
                title="prospective adaptive-resolution spatial test",
            ),
        )
    else:
        log_skipped_figure("adaptive_all_prospective", "missing-prospective-adaptive-artifacts")
    if has_perspective_models(adaptive_results, "retrospective", model_ids):
        maybe_render_formats(
            figure_dir,
            "adaptive_all_retrospective",
            figure_formats,
            cache_mode,
            lambda: generate_adaptive_analysis_plots_all_models(
                model_ids,
                "adaptive_all_retrospective",
                adaptive_results["retrospective"],
                model_names,
                groningen_contour,
                figure_dir,
                figure_formats,
                title="retrospective adaptive-resolution spatial test",
            ),
        )
    else:
        log_skipped_figure("adaptive_all_retrospective", "missing-retrospective-adaptive-artifacts")
    maybe_render_formats(
        figure_dir,
        "loo-cv",
        figure_formats,
        cache_mode,
        lambda: generate_loo_plot(rc, model_names, figure_dir, figure_formats, artifact_dir),
    )
    LOGGER.info("stage=figures status=done figure_dir=%s artifact_dir=%s", figure_dir, artifact_dir)


if __name__ == "__main__":
    main()
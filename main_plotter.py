import argparse
import logging
import sys
from typing import Optional

from src.plots.runner import run_plot_command
from src.plots.settings import PlotSettings, load_plot_settings

LOGGER = logging.getLogger("main_plotter")


def build_parser(settings: PlotSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from aggregated metrics."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="DATASET",
        help="One or more configured datasets.",
    )
    common_parser.add_argument(
        "--title",
        action=argparse.BooleanOptionalAction,
        default=settings.default_title,
        help="Include the automatically generated figure title.",
    )
    common_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    utility_parser = subparsers.add_parser(
        "utility",
        parents=[common_parser],
        help="Generate the downstream utility figure.",
    )
    _add_classifier_argument(utility_parser, settings)

    subparsers.add_parser(
        "privacy",
        parents=[common_parser],
        help="Generate the privacy evaluation figure.",
    )

    tradeoff_parser = subparsers.add_parser(
        "tradeoff",
        parents=[common_parser],
        help="Generate the privacy-utility tradeoff figure.",
    )
    _add_classifier_argument(tradeoff_parser, settings)
    _add_output_ir_argument(tradeoff_parser, settings)

    all_parser = subparsers.add_parser(
        "all",
        parents=[common_parser],
        help="Generate all figures.",
    )
    _add_classifier_argument(all_parser, settings)
    _add_output_ir_argument(all_parser, settings)

    return parser


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        force=True,
    )
    LOGGER.setLevel(logging.DEBUG if verbose else logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def validate_datasets(
    requested: tuple[str, ...],
    available: tuple[str, ...],
) -> tuple[str, ...]:
    datasets = tuple(dict.fromkeys(requested))

    if not datasets:
        raise ValueError("At least one dataset must be selected.")

    unsupported = [dataset for dataset in datasets if dataset not in available]

    if unsupported:
        names = ", ".join(unsupported)
        supported = ", ".join(available)
        raise ValueError(
            f"Unsupported datasets: {names}. Available datasets: {supported}."
        )

    return datasets


def main(argv: Optional[list[str]] = None) -> int:
    try:
        settings = load_plot_settings()
    except (OSError, ValueError, TypeError) as error:
        configure_logging(False)
        LOGGER.error("%s", error)
        return 1

    parser = build_parser(settings)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    requested_datasets = (
        tuple(args.datasets)
        if args.datasets is not None
        else settings.default_datasets
    )

    try:
        datasets = validate_datasets(
            requested_datasets,
            settings.available_datasets,
        )
        classifier = getattr(
            args,
            "classifier",
            settings.default_classifier,
        )
        output_ir = getattr(
            args,
            "output_ir",
            settings.default_output_ir,
        )

        LOGGER.info("Starting command: %s", args.command)
        LOGGER.info("Selected datasets: %s", ", ".join(datasets))
        run_plot_command(
            command=args.command,
            settings=settings,
            datasets=datasets,
            classifier=classifier,
            output_ir=output_ir,
            include_title=args.title,
            logger=LOGGER,
        )
        LOGGER.info("Command completed successfully")
        return 0
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        LOGGER.error("%s", error)
        return 1
    except Exception:
        LOGGER.exception("Unexpected plotting failure")
        return 1


def _add_classifier_argument(
    parser: argparse.ArgumentParser,
    settings: PlotSettings,
) -> None:
    parser.add_argument(
        "--classifier",
        choices=settings.supported_classifiers,
        default=settings.default_classifier,
        help="Classifier used for utility metrics.",
    )


def _add_output_ir_argument(
    parser: argparse.ArgumentParser,
    settings: PlotSettings,
) -> None:
    parser.add_argument(
        "--output-ir",
        type=_positive_float,
        default=settings.default_output_ir,
        help="Single output imbalance ratio used by the tradeoff plot.",
    )


def _positive_float(value: str) -> float:
    parsed = float(value)

    if parsed <= 0:
        raise argparse.ArgumentTypeError("output IR must be greater than zero")

    return parsed


if __name__ == "__main__":
    sys.exit(main())

"""Command-line entry point. Subcommands are added as phases land."""

import argparse

from tractwise import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tractwise", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_subparsers(dest="command")
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

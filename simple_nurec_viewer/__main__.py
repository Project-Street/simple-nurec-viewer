"""
Unified CLI entry point for Simple NuRec Viewer.

This module provides the main entry point for the simple-nurec command,
which supports view and export subcommands using a decorator-based API.
"""

import sys

try:
    from tyro.extras import SubcommandApp
except ImportError:
    print("Error: tyro is not installed. Install with: pip install tyro")
    sys.exit(1)

from .cli.export import export
from .cli.server import server
from .cli.view import view

# Create the SubcommandApp
app = SubcommandApp()

# Register subcommands using decorators
app.command(view)
app.command(export)
app.command(server)


def main() -> int:
    """Main entry point for the unified CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    app.cli()


if __name__ == "__main__":
    main()

"""
Serve the TwinRAG viewer over HTTP.

The viewer is a single self-contained HTML file, so this server is only
needed when you want to access it through a browser or expose it to other
machines on the same network.

Examples:

    python scripts/server_viewer.py
    python scripts/server_viewer.py --rebuild
    python scripts/server_viewer.py --port 9000
    python scripts/server_viewer.py --host 127.0.0.1
    python scripts/server_viewer.py --no-browser

By default, the server binds to 0.0.0.0 so devices on the same Wi-Fi or
LAN can access it. Use --host 127.0.0.1 to restrict access to the current
machine.

The --rebuild option reruns the viewer export and build scripts before
starting the server. This is useful after regenerating simulation,
detection, or evaluation outputs.
"""

from __future__ import annotations

import argparse
import http.server
import socket
import subprocess
import sys
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVE_DIR = PROJECT_ROOT / "data" / "generated"
VIEWER_NAME = "twin_viewer.html"
VIEWER_PATH = SERVE_DIR / VIEWER_NAME

VIEWER_BUILD_SCRIPTS: tuple[str, ...] = (
    "export_graph_view.py",
    "build_viewer.py",
)


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """
    Serve the generated viewer directory over HTTP.

    The root URL is redirected internally to the generated TwinRAG viewer.
    HTML and JSON responses are served using UTF-8 encoding, and browser
    caching is disabled so rebuilt viewer files are always refreshed.
    """

    def do_GET(self) -> None:
        """
        Handle an incoming HTTP GET request.
        """

        if self.path in ("/", "/index.html"):
            self.path = f"/{VIEWER_NAME}"

        favicon_path = SERVE_DIR / "favicon.ico"

        if self.path == "/favicon.ico" and not favicon_path.exists():
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        super().do_GET()

    def guess_type(self, path: str) -> str:
        """
        Return the MIME type for a requested file.

        Args:
            path: Requested file path.

        Returns:
            MIME type string.
        """

        mime_type = super().guess_type(path)

        path_text = str(path).lower()

        if path_text.endswith((".html", ".htm")):
            return "text/html; charset=utf-8"

        if path_text.endswith(".json"):
            return "application/json; charset=utf-8"

        if path_text.endswith(".csv"):
            return "text/csv; charset=utf-8"

        return mime_type

    def end_headers(self) -> None:
        """
        Add response headers before completing the HTTP response.
        """

        self.send_header(
            "Cache-Control",
            "no-store, no-cache, must-revalidate, max-age=0",
        )
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

        super().end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        """
        Write a normalized request log message to stderr.

        Args:
            fmt: Log message format string.
            *args: Values inserted into the format string.
        """

        try:
            message = fmt % args
        except (TypeError, ValueError):
            message = " ".join(str(arg) for arg in args) or str(fmt)

        sys.stderr.write(f"  {message}\n")


def _lan_address() -> str:
    """
    Return the best available local-network IP address.

    The method does not transmit data. It asks the operating system which
    network interface would be used to reach an external address.

    Returns:
        Best-effort IPv4 address for the local machine.
    """

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        probe.connect(("8.8.8.8", 80))
        return str(probe.getsockname()[0])
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"
    finally:
        probe.close()


def _run_build_script(script_name: str) -> None:
    """
    Execute one viewer build script.

    Args:
        script_name: Filename of the script inside the scripts directory.

    Raises:
        SystemExit: If the script does not exist or exits unsuccessfully.
    """

    script_path = PROJECT_ROOT / "scripts" / script_name

    if not script_path.exists():
        raise SystemExit(
            "Required viewer build script was not found:\n"
            f"  {script_path}"
        )

    print(f"Running {script_name}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise SystemExit(
            f"\n{script_name} failed with exit code {result.returncode}.\n\n"
            "Make sure the required project outputs have been generated.\n"
            "Depending on the current project stage, you may need to run:\n"
            "  python scripts/run_baseline_simulation.py\n"
            "  python scripts/run_generated_scenarios.py\n"
            "  python scripts/run_anomaly_detection.py\n"
            "  python scripts/run_event_aggregation.py\n"
            "  python scripts/run_detection_evaluation.py"
        )


def _build_viewer(build_scripts: Sequence[str]) -> None:
    """
    Run all scripts required to generate the viewer.

    Args:
        build_scripts: Ordered script names used to create the viewer.

    Raises:
        SystemExit: If a build step fails or the final viewer is not created.
    """

    SERVE_DIR.mkdir(parents=True, exist_ok=True)

    for script_name in build_scripts:
        _run_build_script(script_name)

    if not VIEWER_PATH.exists():
        raise SystemExit(
            "\nViewer build scripts completed, but the expected file "
            "was not created:\n"
            f"  {VIEWER_PATH}"
        )


def _ensure_viewer_exists(force_rebuild: bool = False) -> None:
    """
    Ensure the generated viewer file is available.

    Args:
        force_rebuild: Rebuild the viewer even if it already exists.
    """

    if VIEWER_PATH.exists() and not force_rebuild:
        return

    if force_rebuild and VIEWER_PATH.exists():
        print("Rebuilding the TwinRAG viewer.\n")
    else:
        print("Viewer not built yet. Generating it now.\n")

    _build_viewer(VIEWER_BUILD_SCRIPTS)

    print("\nViewer build completed.\n")


def _validate_port(port: int) -> int:
    """
    Validate a TCP port number.

    Args:
        port: Requested TCP port.

    Returns:
        The validated port.

    Raises:
        argparse.ArgumentTypeError: If the port is outside the valid range.
    """

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            "Port must be between 1 and 65535."
        )

    return port


def _create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser(
        description="Serve the generated TwinRAG viewer over HTTP."
    )

    parser.add_argument(
        "--port",
        type=_validate_port,
        default=8000,
        help="TCP port to use. Default: 8000.",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help=(
            "Bind address. Use 127.0.0.1 to keep the viewer private to "
            "this machine. Default: 0.0.0.0."
        ),
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the viewer automatically in a browser.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Rebuild the viewer before starting the server, even if the "
            "HTML file already exists."
        ),
    )

    return parser


def _open_browser_later(url: str, delay_s: float = 0.5) -> None:
    """
    Open the viewer URL in the default browser after a short delay.

    Args:
        url: Viewer URL.
        delay_s: Delay before opening the browser.
    """

    timer = threading.Timer(delay_s, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main() -> None:
    """
    Start the TwinRAG viewer HTTP server.
    """

    parser = _create_argument_parser()
    args = parser.parse_args()

    _ensure_viewer_exists(force_rebuild=args.rebuild)

    handler = partial(
        ViewerHandler,
        directory=str(SERVE_DIR),
    )

    try:
        server = http.server.ThreadingHTTPServer(
            (args.host, args.port),
            handler,
        )
    except OSError as error:
        raise SystemExit(
            f"Could not bind to {args.host}:{args.port}.\n"
            f"Reason: {error}\n\n"
            f"Another program may already be using port {args.port}.\n"
            "Try a different port, for example:\n"
            "  python scripts/server_viewer.py --port 8080"
        ) from error

    size_kb = VIEWER_PATH.stat().st_size / 1024
    local_url = f"http://localhost:{args.port}/"

    print(
        f"TwinRAG viewer "
        f"({size_kb:.0f} KB, self-contained)"
    )
    print(f"  This machine   {local_url}")

    if args.host == "0.0.0.0":
        lan_url = f"http://{_lan_address()}:{args.port}/"

        print(f"  Same network   {lan_url}")
        print(
            "\n  Share the 'same network' link with devices connected "
            "to the same Wi-Fi or LAN."
        )
        print(
            "  Windows may ask you to allow the connection through "
            "the firewall."
        )
    else:
        print("\n  Bound to the configured host only.")

    print("\nPress Ctrl+C to stop.\n")

    if not args.no_browser:
        _open_browser_later(local_url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TwinRAG viewer...")
    finally:
        server.shutdown()
        server.server_close()
        print("Stopped.")


if __name__ == "__main__":
    main()
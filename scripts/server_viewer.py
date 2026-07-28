"""
Serve the twin viewer over HTTP.

The viewer is a single self-contained HTML file, so this is only needed
when you want other machines to reach it. On your own machine you can
just open data/generated/twin_viewer.html directly.

    python scripts/server_viewer.py
    python scripts/server_viewer.py --port 9000
    python scripts/server_viewer.py --host 127.0.0.1   # this machine only

By default it binds 0.0.0.0 so anyone on the same network can view it --
Windows will likely raise a firewall prompt the first time. Pass
--host 127.0.0.1 to keep it private to this machine.
"""

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SERVE_DIR = PROJECT_ROOT / "data" / "generated"
VIEWER_NAME = "twin_viewer.html"
VIEWER_PATH = SERVE_DIR / VIEWER_NAME


class ViewerHandler(http.server.SimpleHTTPRequestHandler):
    """
    Serves the generated directory, with the viewer at the root URL.
    """

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/" + VIEWER_NAME

        # Browsers request this unprompted. The page carries an inline
        # icon, so answer "nothing here" rather than raising a 404.
        if self.path == "/favicon.ico" and not (SERVE_DIR / "favicon.ico").exists():
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return

        return super().do_GET()

    def guess_type(self, path):
        # Python's default handler serves .html as bare "text/html" with no
        # charset, which mojibakes the em-dashes, m3 and delta glyphs.
        base = super().guess_type(path)

        if str(path).endswith((".html", ".htm")):
            return "text/html; charset=utf-8"

        if str(path).endswith(".json"):
            return "application/json; charset=utf-8"

        return base

    def end_headers(self):
        # The viewer is rebuilt in place; caching it just confuses people
        # who regenerate and then wonder why nothing changed.
        self.send_header("Cache-Control", "no-store")
        return super().end_headers()

    def log_message(self, fmt, *args):
        # log_error() passes an HTTPStatus enum rather than a string, so
        # never assume these arguments are formattable text.
        try:
            message = fmt % args
        except (TypeError, ValueError):
            message = " ".join(str(arg) for arg in args) or str(fmt)

        sys.stderr.write(f"  {message}\n")


def _lan_address() -> str:
    """
    Best guess at this machine's address on the local network.
    """

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        # No packets are actually sent; this just asks the OS which
        # interface it would use to reach the outside world.
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        probe.close()


def _ensure_viewer_exists() -> None:
    """
    Build the viewer if it has not been generated yet.
    """

    if VIEWER_PATH.exists():
        return

    print("Viewer not built yet - generating it now.\n")

    for script in ("export_graph_view.py", "build_viewer.py"):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            raise SystemExit(
                f"\n{script} failed. Run the simulations first:\n"
                "  python scripts/run_fault_simulation.py\n"
                "  python scripts/run_generated_scenarios.py"
            )

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the TwinRAG viewer over HTTP."
    )

    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address. Use 127.0.0.1 to keep it private to this machine.",
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser window automatically.",
    )

    args = parser.parse_args()

    _ensure_viewer_exists()

    handler = partial(ViewerHandler, directory=str(SERVE_DIR))

    try:
        server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    except OSError as error:
        raise SystemExit(
            f"Could not bind {args.host}:{args.port} - {error}\n"
            f"Another program may already be using port {args.port}; "
            "try --port 8080."
        )

    size_kb = VIEWER_PATH.stat().st_size / 1024

    local_url = f"http://localhost:{args.port}/"

    print(f"TwinRAG viewer  ({size_kb:.0f} KB, self-contained)")
    print(f"  This machine   {local_url}")

    if args.host == "0.0.0.0":
        print(f"  Same network   http://{_lan_address()}:{args.port}/")
        print("\n  Share the 'same network' link with anyone on the same")
        print("  Wi-Fi or LAN. Windows may ask you to allow the connection.")
    else:
        print("\n  Bound to this machine only.")

    print("\nPress Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(local_url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.shutdown()


if __name__ == "__main__":
    main()

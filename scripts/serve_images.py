"""Tiny CORS-enabled static server for the photos, so Label Studio can load them via
plain http:// URLs (avoids the /data/local-files/ serving gotchas).

Run in its own terminal:  .venv/bin/python scripts/serve_images.py
Serves the repo's data/ dir at http://localhost:8081 with Access-Control-Allow-Origin: *.
Images are then at http://localhost:8081/corseacare/<name>.
"""
import http.server
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
PORT = 8081


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def log_message(self, *args):
        pass  # quiet


if __name__ == "__main__":
    socketserver.TCPServer.allow_reuse_address = True
    print(f"Serving {ROOT} at http://localhost:{PORT}  (CORS *) — Ctrl+C to stop")
    with socketserver.TCPServer(("127.0.0.1", PORT), CORSHandler) as httpd:
        httpd.serve_forever()

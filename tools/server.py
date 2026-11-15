from __future__ import annotations
#!/usr/bin/env python3
"""
tools/server.py - Standalone HTTP Tool Server for `dino` Environment

Runs in the `dino` environment (where GroundingDINO and SAM2 reside)
and serves requests from the `think` environment over HTTP.
Zero third-party web dependencies: uses standard library `http.server`.

Endpoints:
  GET  /health
  POST /referdino
  POST /sam2
  POST /kg
"""

import os
import sys
import json
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, List, Dict, Any, Tuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tools.referdino_tool import ReferDINOTool, GroundingDINOTool
from tools.sam2_tool import SAM2Tool
from tools.mock_tools import MockToolServer
from kg.retriever import retrieve


class RealToolServer:
    def __init__(self, media_dir="./REFAVS/media", kg_dir="kg_cache", grounder="gdino"):
        self.media_dir = media_dir
        self.kg_dir = kg_dir
        self.grounder_name = grounder
        if grounder == "referdino":
            self.referdino = ReferDINOTool(use_mock=False, media_dir=media_dir)
        else:
            self.referdino = GroundingDINOTool(use_mock=False, media_dir=media_dir)
        self.sam2 = SAM2Tool(use_mock=False, media_dir=media_dir)

    def referdino_ground(self, video_id: str, text: str) -> List[Dict[str, Any]]:
        candidates = self.referdino.run(video_id, text)
        return [c.to_dict() for c in candidates]

    def sam2_segment(self, video_id: str, candidate_boxes: List[List[float]], keyframe: Optional[int] = None) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
        masks_arr, logits_arr = self.sam2.refine(video_id, candidate_boxes, keyframe=keyframe)
        return masks_arr.tolist(), logits_arr.tolist()

    def kg_retrieve(self, video_id: str, query: str, k_hops: int = 1) -> Dict[str, Any]:
        return retrieve(video_id, query, k_hops=k_hops, kg_dir=self.kg_dir)


class ToolRequestHandler(BaseHTTPRequestHandler):
    tool_server = None

    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "tgs_tool_server"})
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8")
        try:
            req_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body"})
            return

        try:
            if self.path == "/referdino":
                video_id = req_data.get("video_id", "")
                text = req_data.get("text", "")
                candidates = self.tool_server.referdino_ground(video_id, text)
                self._send_json(200, {"candidates": candidates})

            elif self.path == "/sam2":
                video_id = req_data.get("video_id", "")
                boxes = req_data.get("candidate_boxes", [])
                keyframe = req_data.get("keyframe", None)
                masks, logits = self.tool_server.sam2_segment(video_id, boxes, keyframe=keyframe)
                self._send_json(200, {"masks": masks, "logits": logits})

            elif self.path == "/kg":
                video_id = req_data.get("video_id", "")
                query = req_data.get("query", "")
                k_hops = req_data.get("k_hops", 1)
                result = self.tool_server.kg_retrieve(video_id, query, k_hops=k_hops)
                self._send_json(200, result)

            else:
                self._send_json(404, {"error": f"Unknown endpoint {self.path}"})
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def log_message(self, format, *args):
        pass


def run_server(port=8088, host="0.0.0.0", media_dir="./REFAVS/media", kg_dir="kg_cache", grounder="gdino", use_mock=False):
    if use_mock:
        print(f"[Tool Server] Initializing in MOCK mode (--mock passed, grounder={grounder}).")
        tool_server = MockToolServer(media_dir=media_dir, kg_dir=kg_dir)
    else:
        print(f"[Tool Server] Initializing with REAL grounder={grounder} and SAM2 tools.")
        tool_server = RealToolServer(media_dir=media_dir, kg_dir=kg_dir, grounder=grounder)

    ToolRequestHandler.tool_server = tool_server
    server = HTTPServer((host, port), ToolRequestHandler)
    print(f"[Tool Server] Listening on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Tool Server] Shutting down.")
        server.server_close()


def get_parser():
    parser = argparse.ArgumentParser(description="TGS-Agent HTTP Tool Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=8088, help="Port to listen on")
    parser.add_argument("--grounder", type=str, choices=["gdino", "referdino"], default="gdino", help="Grounding tool")
    parser.add_argument("--media_dir", type=str, default="./REFAVS/media", help="Media directory")
    parser.add_argument("--kg_dir", type=str, default="kg_cache", help="KG cache directory")
    parser.add_argument("--mock", action="store_true", help="Run with mock tools for CPU testing")
    return parser


def parse_args(args=None):
    return get_parser().parse_args(args)


def main():
    args = parse_args()
    run_server(port=args.port, host=args.host, media_dir=args.media_dir, kg_dir=args.kg_dir, grounder=args.grounder, use_mock=args.mock)


if __name__ == "__main__":
    main()

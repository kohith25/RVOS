from __future__ import annotations
#!/usr/bin/env python3
"""
tools/client.py - Tool Client with Disk Caching and Explicit Mock Fallback

Communicates with the Tool Server running in the `dino` environment.
Key features:
1. Deterministic Disk Caching: keyed by SHA-256 of (endpoint, video_id, params).
2. Explicit Mock Mode: Mock fallback only occurs if use_mock_fallback=True (--mock).
   Otherwise raises ConnectionError with instructions to start tools/server.py.
3. Strict GT Isolation: tools never accept or access ground truth masks/labels.
"""

import os
import json
import hashlib
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple

from tools.mock_tools import MockToolServer


class ToolClient:
    def __init__(
        self,
        server_url="http://127.0.0.1:8088",
        cache_dir="tool_cache",
        use_cache=True,
        use_mock_fallback=False,
        media_dir="./REFAVS/media",
        kg_dir="kg_cache"
    ):
        self.server_url = server_url.rstrip("/")
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.use_mock_fallback = use_mock_fallback
        self.media_dir = media_dir
        self.kg_dir = kg_dir

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._mock_server = None
        self._server_available = None

    def _get_mock_server(self):
        if self._mock_server is None:
            self._mock_server = MockToolServer(media_dir=self.media_dir, kg_dir=self.kg_dir)
        return self._mock_server

    def _compute_cache_key(self, endpoint, data):
        serialized = json.dumps({"endpoint": endpoint, "data": data}, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _read_cache(self, key):
        if not self.use_cache:
            return None
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def get_info(self) -> Dict[str, Any]:
        """Queries the tool server's /info endpoint to get actual grounder and tool details."""
        if self.use_mock_fallback:
            return {"grounder": "mock", "grounder_class": "MockReferDINOTool", "status": "ok"}
        try:
            req = urllib.request.Request(f"{self.server_url}/info")
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            return {"grounder": "unknown", "grounder_class": "unknown", "status": "offline"}

    def _write_cache(self, key, result):
        if not self.use_cache:
            return
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(result, f)
        except Exception:
            pass

    def _call_mock(self, endpoint, payload, cache_key):
        mock = self._get_mock_server()
        if endpoint == "referdino":
            result = {"candidates": mock.referdino_ground(payload.get("video_id"), payload.get("text"))}
        elif endpoint == "sam2":
            masks, logits = mock.sam2_segment(payload.get("video_id"), payload.get("candidate_boxes"), payload.get("keyframe"))
            result = {"masks": masks, "logits": logits}
        elif endpoint == "kg":
            result = mock.kg_retrieve(payload.get("video_id"), payload.get("query"), payload.get("k_hops", 1))
        else:
            raise ValueError(f"Unknown endpoint {endpoint}")

        self._write_cache(cache_key, result)
        return result

    def _post(self, endpoint, payload):
        cache_key = self._compute_cache_key(endpoint, payload)
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        if self._server_available is False:
            if not self.use_mock_fallback:
                raise ConnectionError(
                    f"Tool server not reachable at {self.server_url}. "
                    "Start it with: python tools/server.py --port 8088 or pass --mock to run with mocks."
                )
            return self._call_mock(endpoint, payload, cache_key)

        # Attempt HTTP call to tool server
        url = f"{self.server_url}/{endpoint.lstrip('/')}"
        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_data,
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                resp_bytes = resp.read()
                result = json.loads(resp_bytes.decode("utf-8"))
                self._server_available = True
                self._write_cache(cache_key, result)
                return result
        except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
            self._server_available = False
            if not self.use_mock_fallback:
                raise ConnectionError(
                    f"Tool server not reachable at {self.server_url}: {e}. "
                    "Start it with: python tools/server.py --port 8088 or pass --mock to run with mocks."
                )

            return self._call_mock(endpoint, payload, cache_key)

    def referdino_ground(self, video_id: str, text: str) -> List[Dict[str, Any]]:
        resp = self._post("referdino", {"video_id": video_id, "text": text})
        return resp.get("candidates", [])

    def sam2_segment(self, video_id: str, candidate_boxes: List[List[float]], keyframe: Optional[int] = None) -> Tuple[List[List[List[int]]], List[List[List[float]]]]:
        resp = self._post("sam2", {"video_id": video_id, "candidate_boxes": candidate_boxes, "keyframe": keyframe})
        return resp.get("masks", []), resp.get("logits", [])

    def kg_retrieve(self, video_id: str, query: str, k_hops: int = 1) -> Dict[str, Any]:
        return self._post("kg", {"video_id": video_id, "query": query, "k_hops": k_hops})

    def retrieve_evidence(self, video_id: str, query: str, k_hops: int = 1) -> Dict[str, Any]:
        return self.kg_retrieve(video_id, query, k_hops=k_hops)


_default_tool_client = None


def get_tool_client(server_url="http://127.0.0.1:8088", use_mock_fallback=False, **kwargs):
    global _default_tool_client
    if _default_tool_client is None:
        _default_tool_client = ToolClient(
            server_url=server_url,
            use_mock_fallback=use_mock_fallback,
            **kwargs
        )
    return _default_tool_client

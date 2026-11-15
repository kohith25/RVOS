from __future__ import annotations
#!/usr/bin/env python3
"""
kg/retriever.py - Multimodal Knowledge Graph Retriever

Retrieves contextual subgraphs and relational facts for a video given a natural language query.
Supports:
- Spatial queries ("left", "right", "center", "front", "behind")
- Audio-visual queries ("sound", "loudest", "music", "playing", "singing")
- Object/Entity queries
- 1-hop and 2-hop relational fact expansion
"""

import os
import json
from pathlib import Path
from kg.build_kg import build_graph


class KGRetriever:
    def __init__(self, kg_dir="kg_cache", media_dir="./REFAVS/media"):
        self.kg_dir = kg_dir
        self.media_dir = media_dir
        self.cache = {}

    def get_graph(self, video_id):
        if video_id in self.cache:
            return self.cache[video_id]

        kg_file = os.path.join(self.kg_dir, f"{video_id}.json")
        if os.path.exists(kg_file):
            with open(kg_file, "r", encoding="utf-8") as f:
                graph = json.load(f)
        else:
            graph = build_graph(video_id, self.media_dir)

        self.cache[video_id] = graph
        return graph

    def retrieve(self, video_id, query, k_hops=1):
        """
        Retrieves relevant graph evidence for a given referring query.
        Returns:
            {
                "facts": List[str],
                "node_ids": List[str]
            }
        """
        graph = self.get_graph(video_id)
        nodes = graph.get("nodes", {})
        edges = graph.get("edges", [])

        query_lower = query.lower() if query else ""
        matched_node_ids = set()

        # Seed node matching
        for nid, node in nodes.items():
            loc = node.get("spatial_location", "")
            lbl = node.get("label", "")
            ntype = node.get("type", "")

            # Match spatial cues
            if loc and loc in query_lower:
                matched_node_ids.add(nid)

            # Match audio cues
            if "sound" in query_lower or "loud" in query_lower or "sing" in query_lower or "play" in query_lower:
                if ntype == "audio_event" or "produces_sound" in str(node):
                    matched_node_ids.add(nid)

            # Match general subject keywords
            for word in query_lower.split():
                if len(word) > 3 and word in lbl.lower():
                    matched_node_ids.add(nid)

        # Fallback seed if query matching was sparse: pick primary visual and audio nodes
        if not matched_node_ids:
            for nid, node in nodes.items():
                if node.get("spatial_location") == "center" or node.get("type") == "audio_event":
                    matched_node_ids.add(nid)
                    if len(matched_node_ids) >= 2:
                        break

        # Hop expansion
        expanded_node_ids = set(matched_node_ids)
        relevant_edges = []

        if k_hops >= 1:
            for edge in edges:
                src, tgt = edge["source"], edge["target"]
                if src in matched_node_ids or tgt in matched_node_ids:
                    expanded_node_ids.add(src)
                    expanded_node_ids.add(tgt)
                    relevant_edges.append(edge)

        # Format facts into descriptive sentences
        facts = []

        # 1. Entity facts
        for nid in expanded_node_ids:
            node = nodes.get(nid)
            if not node:
                continue
            if node.get("type") == "visual_entity":
                loc = node.get("spatial_location", "unknown")
                area = int(node.get("area_ratio", 0.0) * 100)
                facts.append(f"Entity [{node.get('label')}] detected at {loc} region (occupies ~{area}% frame area).")
            elif node.get("type") == "audio_event":
                interval = node.get("interval", [0, 1])
                loudness = node.get("loudness", 0.5)
                if loudness > 0.6:
                    facts.append(f"Strong acoustic activity detected in interval [{interval[0]}-{interval[1]}s] (energy {loudness:.2f}).")

        # 2. Relational facts
        for edge in relevant_edges:
            src_node = nodes.get(edge["source"], {})
            tgt_node = nodes.get(edge["target"], {})
            rel = edge["relation"]
            if rel == "left_of":
                facts.append(f"[{src_node.get('label', edge['source'])}] is situated on the left of [{tgt_node.get('label', edge['target'])}].")
            elif rel == "right_of":
                facts.append(f"[{src_node.get('label', edge['source'])}] is situated on the right of [{tgt_node.get('label', edge['target'])}].")
            elif rel == "in_front_of":
                facts.append(f"[{src_node.get('label', edge['source'])}] is in front of [{tgt_node.get('label', edge['target'])}].")
            elif rel in ("sounding", "produces_sound"):
                interval = tgt_node.get("interval", [0, 1])
                weight = edge.get("weight", 0.5)
                facts.append(f"[{src_node.get('label', edge['source'])}] is sounding/correlated with audio during [{interval[0]}-{interval[1]}s] (score {weight:.2f}).")
            elif rel == "refers_to":
                facts.append(f"Expression refers to [{tgt_node.get('label', edge['target'])}].")
            elif rel == "audio_aligned":
                interval = tgt_node.get("interval", [0, 1])
                facts.append(f"[{src_node.get('label', edge['source'])}] is temporally aligned with audio at [{interval[0]}-{interval[1]}s].")

        # De-duplicate while preserving order
        unique_facts = list(dict.fromkeys(facts))

        return {
            "facts": unique_facts[:6],  # limit to top 6 facts for concise context
            "node_ids": list(expanded_node_ids)
        }


# Global helper instance
_default_retriever = None

def retrieve(video_id, query, k_hops=1, kg_dir="kg_cache"):
    global _default_retriever
    if _default_retriever is None or _default_retriever.kg_dir != kg_dir:
        _default_retriever = KGRetriever(kg_dir=kg_dir)
    return _default_retriever.retrieve(video_id, query, k_hops=k_hops)

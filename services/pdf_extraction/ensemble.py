# services/pdf_extraction/ensemble.py
"""Ensemble voting and consensus logic for table detection and extraction."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from .detectors import DetectedTable
from .utils import BBox, bbox_iou

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3


@dataclass
class ConsensusTable:
    """A table confirmed by ensemble consensus."""
    bbox: BBox
    detections: List[DetectedTable]
    consensus_score: float
    best_detection: DetectedTable
    column_count: Optional[int] = None


def cluster_detections(all_detections: List[DetectedTable],
                       iou_threshold: float = IOU_THRESHOLD) -> List[List[DetectedTable]]:
    """Cluster detections that refer to the same physical table via bbox overlap."""
    if not all_detections:
        return []

    clusters: List[List[DetectedTable]] = []
    used = set()

    sorted_dets = sorted(all_detections, key=lambda d: d.confidence, reverse=True)

    for i, det in enumerate(sorted_dets):
        if i in used:
            continue
        cluster = [det]
        used.add(i)

        for j, other in enumerate(sorted_dets):
            if j in used:
                continue
            for member in cluster:
                if bbox_iou(member.bbox, other.bbox) >= iou_threshold:
                    cluster.append(other)
                    used.add(j)
                    break

        clusters.append(cluster)

    return clusters


def build_consensus(all_detections: List[DetectedTable],
                    total_detectors: int,
                    min_votes: int = 2) -> List[ConsensusTable]:
    """Build consensus tables from clustered detections."""
    clusters = cluster_detections(all_detections)
    consensus_tables = []

    for cluster in clusters:
        unique_detectors = set(d.detector_name for d in cluster)
        num_votes = len(unique_detectors)
        best = max(cluster, key=lambda d: d.confidence)

        # Require min_votes, but allow high-confidence PyMuPDF alone
        if num_votes < min_votes:
            if not (best.confidence >= 0.8 and best.detector_name == "pymupdf"):
                continue

        # Column count by majority vote
        col_counts = [d.column_count for d in cluster if d.column_count is not None]
        consensus_cols = None
        if col_counts:
            consensus_cols = Counter(col_counts).most_common(1)[0][0]

        consensus_tables.append(ConsensusTable(
            bbox=best.bbox,
            detections=cluster,
            consensus_score=num_votes / max(total_detectors, 1),
            best_detection=best,
            column_count=consensus_cols,
        ))

    # Sort by y-position (top to bottom)
    consensus_tables.sort(key=lambda ct: ct.bbox[1])

    return consensus_tables


def score_extraction(extraction_rows: List[List[str]],
                     consensus: ConsensusTable) -> float:
    """Score extraction quality against consensus expectations. Returns 0.0-1.0."""
    if not extraction_rows:
        return 0.0

    score = 0.5  # Base score

    # Column count match
    if consensus.column_count is not None:
        actual_cols = max(len(row) for row in extraction_rows)
        if actual_cols == consensus.column_count:
            score += 0.25
        elif abs(actual_cols - consensus.column_count) <= 1:
            score += 0.1

    # Row count reasonableness
    expected_rows = consensus.best_detection.row_count
    if expected_rows:
        ratio = len(extraction_rows) / expected_rows
        if 0.8 <= ratio <= 1.2:
            score += 0.15
        elif 0.5 <= ratio <= 1.5:
            score += 0.05

    # Content completeness
    total_cells = sum(len(row) for row in extraction_rows)
    non_empty = sum(1 for row in extraction_rows for cell in row if cell.strip())
    if total_cells > 0:
        score += (non_empty / total_cells) * 0.1

    return min(1.0, score)

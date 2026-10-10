"""Score the existing EPFL pipeline against its own ground truth.

This does not train anything and it does not write the benchmark mp4, gif, or
``docs/artifacts/epfl_demo.json``. Ground-truth positions are read only after a
frame has already been tracked, and only to score that frame.

Two protocols, kept apart:

* benchmark — the published 60 annotated frames, one second apart, per-view
  ``n_init=1``, position cap 4.5 cells. Same constants as ``mcreid-public-demo epfl``.
* live — every video frame from the first of those labels to the last, per-view
  ``n_init=2``, position cap 8.5 cells. Accuracy is computed only on the labelled
  timestamps.

The ground truth is a grid-cell id, not a box. TrackEval's HOTA is not computed.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mcreid.calib.epfl import (
    build_rig,
    grid_id_to_world_m,
    parse_calibration,
    parse_ground_truth,
)
from mcreid.cli.epfl_live import LIVE_POSITION_SIGMA_CELLS, live_fusion_config
from mcreid.cli.public_demo import (
    EPFL_CELL_UNIT,
    EPFL_GT_STRIDE,
    EPFL_IMAGE_SIZE,
    EPFL_MAX_POSITION_SIGMA_CELLS,
    epfl_fusion_config,
)
from mcreid.eval.id_metrics import evaluate_id_consistency
from mcreid.fusion.global_id import GlobalIDManager
from mcreid.fusion.types import GlobalTrackSnapshot
from mcreid.track.gpu_view import GpuPerViewBackend, GpuViewConfig
from mcreid.track.per_view import Detection, PerViewConfig
from mcreid.track.reid_models import build_embedder
from mcreid.utils.device import resolve_device
from mcreid.utils.seed import DEFAULT_SEED, seed_everything

REPO = Path(__file__).resolve().parents[1]
# Same radius the published EPFL JSON is scored with: the birth-cluster radius.
MATCH_RADIUS_CELLS = 3.0
PUBLISHED_N_FRAMES = 60
OUT_DIR = REPO / "reports" / "epfl_evaluation"


@dataclass
class Clock:
    """Wall times for one processed frame. Seconds."""

    decode_s: float
    yolo_s: float
    osnet_s: float
    local_track_s: float
    fusion_s: float

    @property
    def pipeline_s(self) -> float:
        return self.yolo_s + self.osnet_s + self.local_track_s + self.fusion_s

    @property
    def end_to_end_s(self) -> float:
        return self.decode_s + self.pipeline_s


def _sync(kind: str) -> None:
    if kind == "mps":
        torch.mps.synchronize()
    elif kind == "cuda":
        torch.cuda.synchronize()


def _self_check() -> None:
    """The metric formulas, on a toy matching, before any video is opened."""
    perfect = [{1: 10}, {1: 10}]
    perfect_counts = _identity_scores(perfect, n_gt=2, n_pred=2)
    if perfect_counts["idf1"] != 1.0 or perfect_counts["idtp"] != 2:
        raise RuntimeError(f"IDF1 self-check failed on a perfect match: {perfect_counts}")

    switched = [{1: 10}, {1: 11}]
    switched_counts = _identity_scores(switched, n_gt=2, n_pred=2)
    if switched_counts["idtp"] != 1 or abs(switched_counts["idf1"] - 0.5) > 1e-9:
        raise RuntimeError(f"IDF1 self-check failed on a switch: {switched_counts}")

    gap = [{1: 7}, {}, {1: 7}]
    present = [True, True, True]
    frag = _fragmentations(gap, [1], present)
    if frag["total"] != 1:
        raise RuntimeError(f"fragmentation self-check failed: {frag}")
    mota = _mota(n_gt=3, n_pred=2, n_matches=2, id_switches=0)
    if abs(mota - (1.0 - 1.0 / 3.0)) > 1e-9:
        raise RuntimeError(f"MOTA self-check failed: {mota}")


def _identity_scores(
    matches: list[dict[int, int]],
    n_gt: int,
    n_pred: int,
) -> dict[str, float]:
    """IDF1 from Ristani et al., given an already-decided per-frame matching.

    IDTP is the maximum pairing of ground-truth ids with predicted ids, weighted
    by how many frames that pair was the matched pair. It is not image-plane
    TrackEval IDF1: the frame matching that feeds it is a ground-plane distance.
    """
    counts: Counter[tuple[int, int]] = Counter()
    for assignment in matches:
        for agent, gid in assignment.items():
            counts[(agent, gid)] += 1
    agents = sorted({agent for agent, _gid in counts})
    gids = sorted({gid for _agent, gid in counts})
    idtp = 0.0
    if agents and gids:
        weight = np.zeros((len(agents), len(gids)), dtype=np.float64)
        for i, agent in enumerate(agents):
            for j, gid in enumerate(gids):
                weight[i, j] = counts.get((agent, gid), 0)
        rows, cols = linear_sum_assignment(-weight)
        idtp = float(weight[rows, cols].sum())
    idfp = float(n_pred) - idtp
    idfn = float(n_gt) - idtp
    denom = (2.0 * idtp) + idfp + idfn
    return {
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idp": (idtp / n_pred) if n_pred else float("nan"),
        "idr": (idtp / n_gt) if n_gt else float("nan"),
        "idf1": (2.0 * idtp / denom) if denom else float("nan"),
    }


def _fragmentations(
    matches: list[dict[int, int]],
    agents: list[int],
    present: list[bool],
) -> dict[str, Any]:
    """Extra pieces of a ground-truth trajectory.

    Only frames where that person is in the room are visited. A fragment is a
    maximal run of those frames in which they are matched. The count is
    ``max(0, n_runs - 1)``: the first run is not an interruption. Leaving the
    room does not itself split a track. An identity change inside one run is a
    switch, not a fragment.
    """
    per_agent: dict[str, int] = {}
    total = 0
    for agent in agents:
        runs = 0
        in_run = False
        for frame, assignment in enumerate(matches):
            if not present[frame]:
                continue
            if agent not in assignment:
                in_run = False
                continue
            if not in_run:
                runs += 1
                in_run = True
        extra = max(0, runs - 1)
        per_agent[str(agent)] = extra
        total += extra
    return {"total": total, "per_gt_id": per_agent}


def _mota(n_gt: int, n_pred: int, n_matches: int, id_switches: int) -> float:
    """CLEAR MOTA on this matching. Not MOTChallenge MOTA: there are no boxes."""
    if n_gt <= 0:
        return float("nan")
    fn = n_gt - n_matches
    fp = n_pred - n_matches
    return 1.0 - (fn + fp + id_switches) / n_gt


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None, "rmse": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "rmse": float(np.sqrt(np.mean(arr**2))),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _score(
    gt_world: dict[int, np.ndarray],
    gt_visible: dict[int, np.ndarray],
    results: list[list[GlobalTrackSnapshot]],
    n_ids_issued: int,
) -> dict[str, Any]:
    report = evaluate_id_consistency(
        gt_world=gt_world,
        gt_visible=gt_visible,
        results=results,
        n_ids_issued=n_ids_issued,
        match_radius_m=MATCH_RADIUS_CELLS,
    )
    agents = sorted(gt_world)
    matches = report.per_frame_matches
    distances: list[float] = []
    nearest: list[float] = []
    support: list[int] = []
    multi_camera_matches = 0
    n_matches = 0
    for frame, assignment in enumerate(matches):
        by_gid = {snap.global_id: snap for snap in results[frame]}
        pred = (
            np.stack([snap.world_xy for snap in results[frame]])
            if results[frame]
            else np.zeros((0, 2), dtype=np.float64)
        )
        for agent in agents:
            truth = gt_world[agent][frame]
            if not np.isfinite(truth).all():
                continue
            if pred.shape[0]:
                nearest.append(float(np.linalg.norm(pred - truth, axis=1).min()))
            gid = assignment.get(agent)
            if gid is None:
                continue
            n_matches += 1
            snap = by_gid[gid]
            distances.append(float(np.linalg.norm(truth - snap.world_xy)))
            n_cam = len(snap.supporting_cameras)
            support.append(n_cam)
            if n_cam >= 2:
                multi_camera_matches += 1

    n_gt = sum(
        1
        for frame in range(len(results))
        for agent in agents
        if np.isfinite(gt_world[agent][frame]).all()
    )
    n_pred = sum(len(frame) for frame in results)
    identity = _identity_scores(matches, n_gt=n_gt, n_pred=n_pred)
    # Presence is per agent. A frame where that agent is absent must end a run.
    per_agent_present = {
        agent: [bool(np.isfinite(gt_world[agent][frame]).all()) for frame in range(len(results))]
        for agent in agents
    }
    frag_total = 0
    frag_per: dict[str, int] = {}
    for agent in agents:
        one = _fragmentations(matches, [agent], per_agent_present[agent])
        frag_per[str(agent)] = int(one["per_gt_id"][str(agent)])
        frag_total += frag_per[str(agent)]

    votes: dict[int, Counter[int]] = {}
    for assignment in matches:
        for agent, gid in assignment.items():
            votes.setdefault(gid, Counter())[agent] += 1
    pure = sum(1 for hist in votes.values() if len(hist) == 1)
    collisions = sum(1 for hist in votes.values() if len(hist) > 1)
    shown = sorted({snap.global_id for frame in results for snap in frame})
    position = _percentiles(distances)
    # The existing scorer's mean is the number already published. Keep it, and
    # fail the run if the two means disagree — that would mean two matchers.
    published_mean = report.mean_position_error_m
    if position["n"] and published_mean is not None and np.isfinite(published_mean):
        if abs(float(position["mean"]) - float(published_mean)) > 1e-6:
            raise RuntimeError(
                f"position mean {position['mean']} != evaluate_id_consistency "
                f"{published_mean}"
            )

    return {
        "match_radius_cells": MATCH_RADIUS_CELLS,
        "match_rule": (
            "Per labelled frame, Hungarian assignment between ground-truth cell "
            "positions and fused global-track positions. A pair is kept only when "
            "its Euclidean distance is <= 3.0 grid cells, which is "
            "EPFL_BIRTH_CLUSTER_CELLS, the radius already used to produce "
            "docs/artifacts/epfl_demo.json. One-to-one. Coasting tracks may match. "
            "Ground truth is room presence, so every camera is marked visible and "
            "the hidden-agent rule in evaluate_id_consistency does not fire."
        ),
        "n_gt_person_frames": n_gt,
        "n_predicted_tracks_frames": n_pred,
        "true_positives": n_matches,
        "false_negatives": n_gt - n_matches,
        "false_positives": n_pred - n_matches,
        "id_switches": report.total_id_switches,
        "id_switches_per_gt": {str(k): int(v) for k, v in report.id_switches.items()},
        "ids_per_gt": {str(k): [int(i) for i in v] for k, v in report.ids_per_agent.items()},
        "coverage_of_present_frames": {str(k): float(v) for k, v in report.coverage.items()},
        "fragmentations": frag_total,
        "fragmentations_per_gt": frag_per,
        "false_positive_tracks": int(report.false_positive_tracks),
        "ids_minted_including_tentative": int(report.n_ids_issued),
        "ids_shown": len(shown),
        "ids_shown_list": shown,
        "gt_identities": len(agents),
        "mota_ground_plane": _mota(n_gt, n_pred, n_matches, report.total_id_switches),
        "identity_f1_ground_plane": identity,
        "position_error_cells_on_matches": position,
        "nearest_prediction_distance_cells": _percentiles(nearest),
        "nearest_note": (
            "Min distance from each present ground-truth person to any prediction "
            "in that frame. Not one-to-one. Diagnostic only; the position-error "
            "block above is the matched one-to-one set."
        ),
        "cross_camera": {
            "matched_snapshots": n_matches,
            "matched_with_at_least_two_cameras": multi_camera_matches,
            "fraction_of_matches_with_at_least_two_cameras": (
                (multi_camera_matches / n_matches) if n_matches else None
            ),
            "supporting_cameras_on_matches": _percentiles([float(n) for n in support]),
            "predicted_ids_matched_to_exactly_one_gt": pure,
            "predicted_ids_matched_to_multiple_gt": collisions,
            "gt_ids_tracked_under_exactly_one_global_id": sum(
                1 for ids in report.ids_per_agent.values() if len(ids) == 1
            ),
        },
        "hota": None,
        "hota_why_not": (
            "Not computed. HOTA in TrackEval integrates detection and association "
            "over bounding-box IoU thresholds from 0.05 to 0.95. gt_lab_6p.txt has "
            "no boxes, only a grid-cell id per person. A distance-threshold average "
            "would be a different number with the same name."
        ),
    }


def _timing_summary(clocks: list[Clock], warmup_s: float) -> dict[str, Any]:
    if not clocks:
        raise RuntimeError("no timed frames")
    pipeline = np.asarray([c.pipeline_s for c in clocks], dtype=np.float64)
    e2e = np.asarray([c.end_to_end_s for c in clocks], dtype=np.float64)
    yolo = np.asarray([c.yolo_s for c in clocks], dtype=np.float64)
    osnet = np.asarray([c.osnet_s for c in clocks], dtype=np.float64)
    fusion = np.asarray([c.fusion_s for c in clocks], dtype=np.float64)
    decode = np.asarray([c.decode_s for c in clocks], dtype=np.float64)
    after = pipeline[1:] if pipeline.size > 1 else pipeline

    def _ms(arr: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(arr.mean() * 1000.0),
            "p95": float(np.percentile(arr, 95) * 1000.0),
        }

    return {
        "warmup_s": warmup_s,
        "warmup_excluded_from_fps": True,
        "timed_frames": int(pipeline.size),
        "processing_fps": float(pipeline.size / pipeline.sum()),
        "processing_fps_excluding_first_timed_frame": float(after.size / after.sum()),
        "end_to_end_fps_including_decode": float(e2e.size / e2e.sum()),
        "pipeline_latency_ms": _ms(pipeline),
        "end_to_end_latency_ms": _ms(e2e),
        "yolo_ms_sum_of_four_cameras": _ms(yolo),
        "osnet_ms_sum_of_four_cameras": _ms(osnet),
        "fusion_ms": _ms(fusion),
        "decode_ms": _ms(decode),
        "pipeline_s_total": float(pipeline.sum()),
        "decode_s_total": float(decode.sum()),
        "end_to_end_s_total_excluding_warmup": float(e2e.sum()),
    }


def _load_models(
    weights: Path,
    camera_ids: list[str],
    per_view: PerViewConfig,
) -> tuple[Any, dict[str, GpuPerViewBackend], str]:
    spec = resolve_device("auto")
    if spec.kind == "cpu":
        raise RuntimeError(f"evaluation resolved to CPU ({spec}); MPS or CUDA is required")
    probe = torch.zeros(1, device=spec.torch_device)
    if probe.device.type != spec.kind:
        raise RuntimeError(f"tensor landed on {probe.device}, expected {spec.kind}")
    view = GpuViewConfig(
        weights=weights,
        imgsz=640,
        conf_threshold=0.25,
        embedder="osnet_x1_0_msmt17",
        device="auto",
    )
    from ultralytics import YOLO

    detector = YOLO(str(weights))
    embedder = build_embedder(
        view.embedder,
        device=spec,
        batch_size=view.embed_batch,
        weights_dir=view.weights_dir,
    )
    backends = {
        camera_id: GpuPerViewBackend(
            camera_id,
            view,
            per_view,
            detector=detector,
            embedder=embedder,
        )
        for camera_id in camera_ids
    }
    return detector, backends, spec.kind


def _warmup(backend: GpuPerViewBackend, image: np.ndarray, kind: str) -> float:
    """One detect + embed. The tracker is not updated, so frame 0 is still fresh."""
    _sync(kind)
    started = time.perf_counter()
    boxes, _scores = backend.detect(image)
    _ = backend.embedder(image, boxes)
    _sync(kind)
    return time.perf_counter() - started


def _view_step(
    backend: GpuPerViewBackend,
    image: np.ndarray,
    frame: int,
    kind: str,
) -> tuple[list[Any], float, float, float]:
    """Same body as ``GpuPerViewBackend.step``, with the three stages timed."""
    _sync(kind)
    t0 = time.perf_counter()
    boxes, scores = backend.detect(image)
    _sync(kind)
    t1 = time.perf_counter()
    embeddings = backend.embedder(image, boxes)
    _sync(kind)
    t2 = time.perf_counter()
    detections = [
        Detection(
            bbox_xyxy=boxes[i],
            score=float(np.clip(scores[i], 0.0, 1.0)),
            embedding=embeddings[i],
        )
        for i in range(boxes.shape[0])
    ]
    observations = backend.tracker.update(detections, frame)
    _sync(kind)
    t3 = time.perf_counter()
    return observations, t1 - t0, t2 - t1, t3 - t2


def _fuse(
    manager: GlobalIDManager,
    views: list[Any],
    frame: int,
    dt: float,
    kind: str,
) -> tuple[list[GlobalTrackSnapshot], float]:
    _sync(kind)
    started = time.perf_counter()
    snapshots = manager.step(views, frame, dt)
    _sync(kind)
    return list(snapshots), time.perf_counter() - started


def _read_quartet(captures: list[Any]) -> tuple[list[np.ndarray], int] | None:
    images: list[np.ndarray] = []
    reported: list[int] = []
    for capture in captures:
        ok, frame = capture.read()
        if not ok or frame is None:
            return None
        image = np.asarray(frame, dtype=np.uint8)
        width, height = EPFL_IMAGE_SIZE
        if image.shape[1] != width or image.shape[0] != height:
            raise RuntimeError(
                f"frame is {image.shape[1]}x{image.shape[0]}, calibration expects {width}x{height}"
            )
        images.append(image.copy())
        reported.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    if len(set(reported)) != 1:
        raise RuntimeError(f"cameras are not on the same frame: {reported}")
    return images, reported[0]


def _open_captures(root: Path, sequence: str, n_cameras: int) -> list[Any]:
    captures = []
    for index in range(n_cameras):
        path = root / f"{sequence}-c{index}.avi"
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            for opened in captures:
                opened.release()
            raise RuntimeError(f"could not open {path}")
        captures.append(capture)
    return captures


def _run_benchmark(
    root: Path,
    sequence: str,
    camera_ids: list[str],
    rig: Any,
    fusion: Any,
    weights: Path,
    labelled_frames: list[int],
    dt: float,
) -> dict[str, Any]:
    seed_everything(DEFAULT_SEED)
    per_view = PerViewConfig(n_init=1)
    detector, backends, kind = _load_models(weights, camera_ids, per_view)
    captures = _open_captures(root, sequence, len(camera_ids))
    decode_started = time.perf_counter()
    frames: list[list[np.ndarray]] = []
    try:
        for index in labelled_frames:
            for capture in captures:
                capture.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            quartet = _read_quartet(captures)
            if quartet is None:
                raise RuntimeError(f"could not read annotated frame {index}")
            images, reported = quartet
            if reported != index:
                raise RuntimeError(f"seeked to {index} but cameras reported {reported}")
            frames.append(images)
    finally:
        for capture in captures:
            capture.release()
    decode_s = time.perf_counter() - decode_started
    warmup_s = _warmup(backends[camera_ids[0]], frames[0][0], kind)
    manager = GlobalIDManager(rig, fusion)
    clocks: list[Clock] = []
    results: list[list[GlobalTrackSnapshot]] = []
    per_frame_decode = decode_s / len(frames)
    for images, index in zip(frames, labelled_frames, strict=True):
        views: list[Any] = []
        yolo_s = osnet_s = local_s = 0.0
        for camera_id, image in zip(camera_ids, images, strict=True):
            observed, yolo, osnet, local = _view_step(backends[camera_id], image, index, kind)
            views.extend(observed)
            yolo_s += yolo
            osnet_s += osnet
            local_s += local
        snapshots, fusion_s = _fuse(manager, views, index, dt, kind)
        results.append(snapshots)
        clocks.append(
            Clock(
                decode_s=per_frame_decode,
                yolo_s=yolo_s,
                osnet_s=osnet_s,
                local_track_s=local_s,
                fusion_s=fusion_s,
            )
        )
    n_issued = manager.n_ids_issued
    del manager, backends, detector
    gc.collect()
    if kind == "mps":
        torch.mps.empty_cache()
    return {
        "results": results,
        "clocks": clocks,
        "warmup_s": warmup_s,
        "n_ids_issued": n_issued,
        "per_view_n_init": per_view.n_init,
        "position_sigma_cap_cells": fusion.max_position_sigma_m,
        "dt_s": dt,
        "frames_fed_to_tracker": len(labelled_frames),
    }


def _run_live(
    root: Path,
    sequence: str,
    camera_ids: list[str],
    rig: Any,
    fusion: Any,
    weights: Path,
    score_frames: set[int],
    start: int,
    end: int,
) -> dict[str, Any]:
    seed_everything(DEFAULT_SEED)
    per_view = PerViewConfig()
    detector, backends, kind = _load_models(weights, camera_ids, per_view)
    captures = _open_captures(root, sequence, len(camera_ids))
    fps = float(captures[0].get(cv2.CAP_PROP_FPS))
    if not fps > 1.0:
        raise RuntimeError(f"video fps is not usable: {fps}")
    dt = 1.0 / fps
    for capture in captures:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    first = _read_quartet(captures)
    if first is None:
        raise RuntimeError(f"could not read live start frame {start}")
    images, reported = first
    if reported != start:
        raise RuntimeError(f"live start seeked to {start} but cameras reported {reported}")
    warmup_s = _warmup(backends[camera_ids[0]], images[0], kind)
    manager = GlobalIDManager(rig, fusion)
    clocks: list[Clock] = []
    scored: dict[int, list[GlobalTrackSnapshot]] = {}
    index = reported
    processed = 0
    started = time.perf_counter()
    try:
        while index <= end:
            decode_s = 0.0
            if processed > 0:
                decode_started = time.perf_counter()
                quartet = _read_quartet(captures)
                decode_s = time.perf_counter() - decode_started
                if quartet is None:
                    raise RuntimeError(f"video ended at frame {index} before {end}")
                images, index = quartet
            views: list[Any] = []
            yolo_s = osnet_s = local_s = 0.0
            for camera_id, image in zip(camera_ids, images, strict=True):
                observed, yolo, osnet, local = _view_step(backends[camera_id], image, index, kind)
                views.extend(observed)
                yolo_s += yolo
                osnet_s += osnet
                local_s += local
            snapshots, fusion_s = _fuse(manager, views, index, dt, kind)
            if index in score_frames:
                scored[index] = snapshots
            clocks.append(
                Clock(
                    decode_s=decode_s,
                    yolo_s=yolo_s,
                    osnet_s=osnet_s,
                    local_track_s=local_s,
                    fusion_s=fusion_s,
                )
            )
            processed += 1
            if processed % 100 == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"  live {weights.name}: frame {index}  {processed} steps  "
                    f"{processed / elapsed:.2f} fps",
                    flush=True,
                )
            if index >= end:
                break
    finally:
        for capture in captures:
            capture.release()
    missing = sorted(score_frames - set(scored))
    if missing:
        raise RuntimeError(f"live run missed labelled frames: {missing[:8]}")
    order = sorted(score_frames)
    results = [scored[frame] for frame in order]
    n_issued = manager.n_ids_issued
    del manager, backends, detector
    gc.collect()
    if kind == "mps":
        torch.mps.empty_cache()
    return {
        "results": results,
        "clocks": clocks,
        "warmup_s": warmup_s,
        "n_ids_issued": n_issued,
        "per_view_n_init": per_view.n_init,
        "position_sigma_cap_cells": fusion.max_position_sigma_m,
        "dt_s": dt,
        "video_fps": fps,
        "frames_fed_to_tracker": processed,
    }


def _gt_arrays(
    positions: dict[int, dict[int, int]],
    frames: list[int],
    grid_w: int,
    n_cameras: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[int]]:
    people = sorted({person for frame in frames for person in positions[frame]})
    gt_world = {person: np.full((len(frames), 2), np.nan, dtype=np.float64) for person in people}
    gt_visible = {
        person: np.zeros((len(frames), n_cameras), dtype=bool) for person in people
    }
    for slot, frame in enumerate(frames):
        for person, position_id in positions[frame].items():
            gt_world[person][slot] = grid_id_to_world_m(position_id, grid_w, EPFL_CELL_UNIT)
            gt_visible[person][slot, :] = True
    return gt_world, gt_visible, people


def _describe_gt(
    positions: dict[int, dict[int, int]],
    header: dict[str, int],
    frames: list[int],
) -> dict[str, Any]:
    keys = sorted(positions)
    gaps = sorted({b - a for a, b in zip(keys, keys[1:], strict=False)})
    occupancy = [len(positions[frame]) for frame in frames]
    inside = [frame for frame in keys if frames[0] <= frame <= frames[-1]]
    return {
        "file": "data/epfl_lab/gt_lab_6p.txt",
        "format": (
            "First line is a version integer. The next line is "
            "n_frames n_people grid_w grid_h fps, then extra trailing integers "
            "that the parser ignores. Each later line is one video frame, one "
            "column per person. A value >= 0 is a row-major grid-cell id. A "
            "negative value means that person is not in the room and is dropped."
        ),
        "header": header,
        "grid_cells": [header["grid_w"], header["grid_h"]],
        "annotated_frames_in_file": len(keys),
        "first_annotated_frame": keys[0],
        "last_annotated_frame": keys[-1],
        "gaps_between_annotated_frames": gaps,
        "people_columns": header["n_people"],
        "people_present_anywhere": sorted({p for row in positions.values() for p in row}),
        "benchmark_frames": frames,
        "benchmark_span_inclusive": [frames[0], frames[-1]],
        "video_frames_in_benchmark_span": frames[-1] - frames[0] + 1,
        "annotated_frames_inside_benchmark_span": inside,
        "span_contains_only_the_benchmark_labels": inside == frames,
        "occupancy_on_benchmark_labels": {
            "mean": float(np.mean(occupancy)),
            "min": int(np.min(occupancy)),
            "max": int(np.max(occupancy)),
        },
        "world_coordinate": (
            "grid_id_to_world_m maps a cell id to (col, row) in grid cells, the "
            "cell origin, not the cell centre. Predictions are continuous. This is "
            "the same mapping the published JSON uses. Distances are grid cells, "
            "never metres."
        ),
        "fed_to_tracker": False,
    }


def _pack_run(
    protocol: str,
    weights: Path,
    raw: dict[str, Any],
    score: dict[str, Any],
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "detector": weights.name,
        "weights": str(weights.relative_to(REPO)),
        "imgsz": 640,
        "conf": 0.25,
        "embedder": "osnet_x1_0_msmt17",
        "seed": DEFAULT_SEED,
        "per_view_n_init": raw["per_view_n_init"],
        "fusion_n_init": 5,
        "position_sigma_cap_cells": raw["position_sigma_cap_cells"],
        "benchmark_position_sigma_cap_cells": EPFL_MAX_POSITION_SIGMA_CELLS,
        "live_position_sigma_cap_cells": LIVE_POSITION_SIGMA_CELLS,
        "dt_s": raw["dt_s"],
        "video_fps": raw.get("video_fps"),
        "frames_fed_to_tracker": raw["frames_fed_to_tracker"],
        "labelled_frames_scored": PUBLISHED_N_FRAMES,
        "score": score,
        "timing": _timing_summary(raw["clocks"], raw["warmup_s"]),
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def _render_report(doc: dict[str, Any]) -> str:
    runs = { (r["protocol"], r["detector"]): r for r in doc["runs"] }
    bx = runs[("benchmark", "yolo11x.pt")]
    bs = runs[("benchmark", "yolo11s.pt")]
    lx = runs[("live", "yolo11x.pt")]
    ls = runs[("live", "yolo11s.pt")]
    gt = doc["ground_truth"]
    published = doc["published_artifact"]

    def row(run: dict[str, Any]) -> str:
        score = run["score"]
        pos = score["position_error_cells_on_matches"]
        ident = score["identity_f1_ground_plane"]
        timing = run["timing"]
        return (
            f"| {run['protocol']} / {run['detector']} "
            f"| {_fmt(ident['idf1'])} "
            f"| {_fmt(ident['idp'])} "
            f"| {_fmt(ident['idr'])} "
            f"| {_fmt(score['mota_ground_plane'])} "
            f"| {score['id_switches']} "
            f"| {score['fragmentations']} "
            f"| {score['false_positives']} "
            f"| {score['false_negatives']} "
            f"| {score['gt_identities']} / {score['ids_shown']} "
            f"| {_fmt(pos['mean'])} "
            f"| {_fmt(pos['median'])} "
            f"| {_fmt(pos['p95'])} "
            f"| {_fmt(pos['max'])} "
            f"| {_fmt(timing['processing_fps'], 2)} "
            f"| {_fmt(timing['pipeline_latency_ms']['mean'], 1)} "
            f"| {_fmt(timing['pipeline_latency_ms']['p95'], 1)} |"
        )

    def problems(primary: dict[str, Any]) -> list[str]:
        score = primary["score"]
        fn = score["false_negatives"]
        fp = score["false_positives"]
        switches = score["id_switches"]
        terms = [("kimlik değişimi (ID switch)", switches), ("false positive", fp), ("false negative", fn)]
        terms.sort(key=lambda item: item[1], reverse=True)
        lines = []
        for name, count in terms:
            lines.append(f"{name}: {count} olay, MOTA paydasındaki {score['n_gt_person_frames']} GT kişi-karesine karşı")
        return lines

    lines = [
        "# EPFL Laboratory — doğruluk ve takip değerlendirmesi",
        "",
        f"Ölçüm zamanı: {doc['measured_at']}. Cihaz: {doc['device']['spec']}. "
        f"Tensor cihazı: `{doc['device']['tensor_device']}`. half={doc['device']['half']}.",
        "",
        "Bu rapor yeni bir model değildir. Mevcut YOLO + OSNet + per-view tracker + "
        "global fusion çıktısı, `gt_lab_6p.txt` içindeki zemin hücrelerine karşı skorlanmıştır. "
        "Ground truth tracker'a verilmemiştir.",
        "",
        "## Protokol",
        "",
        f"- GT dosyası: {gt['annotated_frames_in_file']} etiketli kare "
        f"(video karesi {gt['first_annotated_frame']} … {gt['last_annotated_frame']}), "
        f"aralıklar {gt['gaps_between_annotated_frames']}. "
        f"Izgara {gt['grid_cells'][0]}×{gt['grid_cells'][1]}. Birim grid cell.",
        f"- Yayınlanmış benchmark penceresi: ilk {PUBLISHED_N_FRAMES} etiketli kare, "
        f"video {gt['benchmark_span_inclusive'][0]} … {gt['benchmark_span_inclusive'][1]} "
        f"({gt['video_frames_in_benchmark_span']} video karesi). "
        f"Bu aralıkta başka etiket yok: {gt['span_contains_only_the_benchmark_labels']}.",
        f"- Eşleştirme: her etiketli karede Hungarian, bire bir, mesafe eşiği "
        f"**{MATCH_RADIUS_CELLS:g} grid cell** (fusion doğum yarıçapı, `epfl_demo.json` ile aynı).",
        "- Konum: `grid_id_to_world_m` hücre köşesidir, hücre merkezi değil. "
        "Hata yalnız eşiği geçen eşleşmeler üzerindedir.",
        "- **HOTA, DetA, AssA hesaplanmadı.** TrackEval bu sayıları kutu IoU'su üzerinden üretir. "
        "Bu GT'de kutu yok. Sayı uydurulmadı.",
        "- IDF1, aşağıdaki ground-plane eşleşmesinin üstünde Ristani formülüdür. "
        "TrackEval'in görüntü-düzlemi IDF1'si değildir.",
        "- MOTA, aynı eşleşmede CLEAR formülüdür: `1 - (FN + FP + IDSW) / N_gt`. "
        "MOTChallenge MOTA'sı değildir.",
        "- ID switch, `evaluate_id_consistency` tanımıdır: bir GT kişisi A kimliğinden "
        "sonra B kimliğine eşlenirse sayılır. Eşleşmeyen kare tek başına switch değildir.",
        "- Fragmentation: bir GT kişisinin etiketli karelerdeki eşleşme koşuları. "
        "İlki hariç her yeni koşu bir parçalanmadır. Switch, koşu içindeki kimlik değişimidir.",
        "",
        "### Benchmark ve live ayrımı",
        "",
        "| | benchmark | live |",
        "|---|---|---|",
        "| Tracker'a giren kareler | yalnız 60 etiket, 1 s arayla | aralıktaki her video karesi |",
        "| Skorlanan kareler | o 60 etiket | yalnız o 60 etiket |",
        "| per-view n_init | 1 | 2 (tracker varsayılanı) |",
        "| fusion n_init | 5 | 5 |",
        f"| konum sigma tavanı | {EPFL_MAX_POSITION_SIGMA_CELLS:g} cell | {LIVE_POSITION_SIGMA_CELLS:g} cell |",
        "| dt | 1.0 s | video fps'ten, 1/fps |",
        "",
        "Live tavanı benchmark fonksiyonunu değiştirmez. `epfl_fusion_config()` 4.5 cell kalır; "
        "live kopyası yalnız `max_position_sigma_m` alanını 8.5 cell yapar.",
        "",
        "## Metrikler",
        "",
        "Birim, konum sütunlarında grid cell. FPS, warmup hariç işlem hızıdır "
        "(dört kameranın YOLO + OSNet + yerel takip + fusion toplamı).",
        "",
        "| protokol / model | IDF1 | IDP | IDR | MOTA | IDSW | Frag | FP | FN | GT kişi / gösterilen ID | ort. hata | medyan | P95 | max | FPS | ort. ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        row(bx),
        row(bs),
        row(lx),
        row(ls),
        "",
        "### Yayınlanmış JSON ile YOLO11x benchmark",
        "",
        f"`docs/artifacts/epfl_demo.json` calibrated kolu, bu dosyaya dokunulmadan okundu: "
        f"ID switch {published['id_switches']}, ortalama konum hatası "
        f"{published['mean_position_error_cells']:.3f} cell, gösterilen ID "
        f"{published['ids_shown']}, GT kimlik {published['gt_identities']}, "
        f"false-positive track {published['false_positive_tracks']}.",
        "",
        f"Bu çalıştırmada aynı protokol: ID switch {bx['score']['id_switches']}, "
        f"ortalama hata {_fmt(bx['score']['position_error_cells_on_matches']['mean'])} cell, "
        f"gösterilen ID {bx['score']['ids_shown']}, "
        f"false-positive track {bx['score']['false_positive_tracks']}.",
        "",
        "### Zamanlama",
        "",
        "| protokol / model | warmup s | işlem FPS | ilk kare hariç FPS | e2e FPS (decode dahil) | YOLO ms (4 kamera) | OSNet ms | fusion ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in (bx, bs, lx, ls):
        timing = run["timing"]
        lines.append(
            f"| {run['protocol']} / {run['detector']} "
            f"| {_fmt(timing['warmup_s'], 2)} "
            f"| {_fmt(timing['processing_fps'], 2)} "
            f"| {_fmt(timing['processing_fps_excluding_first_timed_frame'], 2)} "
            f"| {_fmt(timing['end_to_end_fps_including_decode'], 2)} "
            f"| {_fmt(timing['yolo_ms_sum_of_four_cameras']['mean'], 1)} "
            f"| {_fmt(timing['osnet_ms_sum_of_four_cameras']['mean'], 1)} "
            f"| {_fmt(timing['fusion_ms']['mean'], 1)} |"
        )
    lines.extend(
        [
            "",
            "Decode, benchmark'ta 60 karenin diske seek edilmesidir ve kare başına eşit bölünmüştür. "
            "Live'da ilk karenin decode'u seek'in içindedir; sonraki kareler ardışık `read` süresidir.",
            "",
            "## Ne işe yarıyor, ne yaramıyor",
            "",
        ]
    )
    lines.extend(_judgement(bx, bs, lx, ls))
    lines.extend(["", "## En büyük üç teknik problem", ""])
    for index, text in enumerate(problems(bx), start=1):
        lines.append(f"{index}. Benchmark YOLO11x MOTA terimleri, büyükten küçüğe: {text}.")
    lines.extend(["", "## Sonraki en değerli iyileştirme", "", _next_step(bx, lx), ""])
    lines.append(
        "Ham sayılar `reports/epfl_evaluation/metrics.json` içindedir. "
        "GT koordinatları tracker fonksiyonlarına geçmez; skor, `step` döndükten sonra hesaplanır."
    )
    lines.append("")
    return "\n".join(lines)


def _judgement(
    bx: dict[str, Any],
    bs: dict[str, Any],
    lx: dict[str, Any],
    ls: dict[str, Any],
) -> list[str]:
    """Short reading of the four runs. Numbers only, no model ranking by speed."""
    out = []
    for run in (bx, bs, lx, ls):
        score = run["score"]
        pos = score["position_error_cells_on_matches"]
        cross = score["cross_camera"]
        out.append(
            f"- **{run['protocol']} / {run['detector']}.** "
            f"Eşleşen noktaların medyan hatası {_fmt(pos['median'])} cell, "
            f"P95 {_fmt(pos['p95'])} cell (eşik 3.0). "
            f"GT kişi {score['gt_identities']}, gösterilen global ID {score['ids_shown']}, "
            f"mint edilen (tentative dahil) {score['ids_minted_including_tentative']}. "
            f"IDF1 {_fmt(score['identity_f1_ground_plane']['idf1'])}, "
            f"MOTA {_fmt(score['mota_ground_plane'])}, "
            f"switch {score['id_switches']}, frag {score['fragmentations']}, "
            f"FP {score['false_positives']}, FN {score['false_negatives']}. "
            f"Eşleşmelerin {_fmt(100 * (cross['fraction_of_matches_with_at_least_two_cameras'] or 0), 1)}% "
            f"ikisi en az iki kameradan geliyor. "
            f"Birden fazla GT'ye yapışan global ID: {cross['predicted_ids_matched_to_multiple_gt']}. "
            f"Tek global ID altında kalan GT kişi: "
            f"{cross['gt_ids_tracked_under_exactly_one_global_id']} / {score['gt_identities']}."
        )
    bxf = bx["timing"]["processing_fps"]
    bsf = bs["timing"]["processing_fps"]
    out.append(
        f"- Hız bir doğruluk ölçüsü değildir. Benchmark'ta yolo11s {_fmt(bsf, 2)} FPS, "
        f"yolo11x {_fmt(bxf, 2)} FPS. IDF1 {_fmt(bs['score']['identity_f1_ground_plane']['idf1'])} "
        f"karşı {_fmt(bx['score']['identity_f1_ground_plane']['idf1'])}. "
        "Hangisinin daha iyi olduğu IDF1, switch ve konum hatasına bakılarak söylenir."
    )
    return out


def _next_step(benchmark_x: dict[str, Any], live_x: dict[str, Any]) -> str:
    score = benchmark_x["score"]
    fn = score["false_negatives"]
    fp = score["false_positives"]
    switches = score["id_switches"]
    largest = max((switches, "switch"), (fp, "fp"), (fn, "fn"))
    if largest[1] == "switch":
        what = (
            "Benchmark MOTA'yı en çok ID switch şişiriyor. Aynı kişinin saniyede bir örneklenen "
            "konumu, kutular örtüşmediği için yerel IoU takibine değil fusion'a kalıyor. "
            "En değerli sonraki iş yeni bir dedektör değil, 1 saniyelik boşlukta kimliği "
            "tutan association. Konum hatası eşik içindeyse homografiyi yeniden yazmak bu sayıyı düzeltmez."
        )
    elif largest[1] == "fn":
        what = (
            "En büyük terim false negative: etiketli kişinin 3 cell içinde bir fused track'i yok. "
            "Önce kaçırmanın dedektörden mi, `n_init` onayından mı, yoksa sigma tavanının "
            "ölçümü atmasından mı geldiğini ayırmak gerekir."
        )
    else:
        what = (
            "En büyük terim false positive: haritada GT'ye bağlanmayan fused track var. "
            "Yeni bir kimlik modeli değil, sahte doğumları kesmek (güven ve çok-kamera onayı) "
            "bu protokolde daha çok MOTA kazandırır."
        )
    live_fn = live_x["score"]["false_negatives"]
    return (
        f"{what} Live YOLO11x, aynı 60 etikette FN={live_fn} "
        f"(benchmark FN={fn}), sigma tavanı 8.5 cell ile. "
        "Bu iki FN'yi birbirine eşitleyip 'live daha iyi' deme: live her video karesini gördü, "
        "benchmark yalnız 1 Hz örnek gördü."
    )


def main() -> int:
    _self_check()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "data" / "epfl_lab")
    parser.add_argument("--sequence", default="6p")
    args = parser.parse_args()

    positions, header = parse_ground_truth(args.root / f"gt_lab_{args.sequence}.txt")
    labelled = sorted(positions)[:PUBLISHED_N_FRAMES]
    if len(labelled) < PUBLISHED_N_FRAMES:
        raise RuntimeError(f"only {len(labelled)} annotated frames, need {PUBLISHED_N_FRAMES}")
    calibs = parse_calibration(args.root / f"calibration-{args.sequence}.txt")
    grid = (header["grid_w"], header["grid_h"])
    rig = build_rig(calibs, EPFL_CELL_UNIT, EPFL_IMAGE_SIZE, grid)
    camera_ids = [camera.camera_id for camera in rig.cameras]
    gt_world, gt_visible, _people = _gt_arrays(positions, labelled, grid[0], len(camera_ids))
    spec = resolve_device("auto")
    published_path = REPO / "docs" / "artifacts" / "epfl_demo.json"
    published_raw = json.loads(published_path.read_text(encoding="utf-8"))
    published_cal = published_raw["arms"]["calibrated"]

    print(f"device {spec}", flush=True)
    print(
        f"benchmark labels {labelled[0]}..{labelled[-1]}  n={len(labelled)}  "
        f"people={sorted(gt_world)}",
        flush=True,
    )

    jobs = (
        ("benchmark", REPO / "weights" / "yolo11x.pt"),
        ("benchmark", REPO / "weights" / "yolo11s.pt"),
        ("live", REPO / "weights" / "yolo11x.pt"),
        ("live", REPO / "weights" / "yolo11s.pt"),
    )
    packed: list[dict[str, Any]] = []
    for protocol, weights in jobs:
        if not weights.is_file():
            raise RuntimeError(f"missing weights: {weights}")
        print(f"running {protocol} {weights.name}", flush=True)
        if protocol == "benchmark":
            raw = _run_benchmark(
                args.root,
                args.sequence,
                camera_ids,
                rig,
                epfl_fusion_config(),
                weights,
                labelled,
                dt=EPFL_GT_STRIDE / float(header["fps"]),
            )
        else:
            raw = _run_live(
                args.root,
                args.sequence,
                camera_ids,
                rig,
                live_fusion_config(epfl_fusion_config()),
                weights,
                set(labelled),
                labelled[0],
                labelled[-1],
            )
        score = _score(gt_world, gt_visible, raw["results"], raw["n_ids_issued"])
        del raw["results"]
        packed.append(_pack_run(protocol, weights, raw, score))
        ident = score["identity_f1_ground_plane"]
        print(
            f"  IDF1 {ident['idf1']:.3f}  MOTA {score['mota_ground_plane']:.3f}  "
            f"IDSW {score['id_switches']}  "
            f"mean {score['position_error_cells_on_matches']['mean']:.3f} cells  "
            f"FPS {packed[-1]['timing']['processing_fps']:.2f}",
            flush=True,
        )

    tensor = torch.zeros(1, device=spec.torch_device)
    doc = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "units": "grid cells",
        "ground_truth_fed_to_tracker": False,
        "device": {
            "spec": str(spec),
            "kind": spec.kind,
            "half": spec.use_half,
            "tensor_device": str(tensor.device),
            "torch": torch.__version__,
        },
        "ground_truth": _describe_gt(positions, header, labelled),
        "published_artifact": {
            "path": "docs/artifacts/epfl_demo.json",
            "not_rewritten": True,
            "id_switches": published_cal["id_switches"],
            "mean_position_error_cells": published_cal["mean_position_error_cells"],
            "ids_shown": published_cal["ids_shown"],
            "gt_identities": published_raw["gt_identities_in_segment"],
            "false_positive_tracks": published_cal["false_positive_tracks"],
            "mean_live_ids_per_frame": published_cal["mean_live_ids_per_frame"],
        },
        "runs": packed,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT_DIR / "metrics.json"
    report_path = OUT_DIR / "report.md"
    metrics_path.write_text(json.dumps(_jsonable(doc), indent=2), encoding="utf-8")
    report_path.write_text(_render_report(doc), encoding="utf-8")
    print(f"wrote {metrics_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

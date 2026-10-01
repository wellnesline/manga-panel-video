"""slide_animator — diagram_blob.blob_renderer の wrapper

各 slide PNG について:
1. blob_split_v2 が出力した labels.npz + per-blob PNG を読む
2. 自動 order JSON を生成(auto_stagger=True 用)
3. blob_renderer.render_diagram_anim() で 3秒で全 blob 描写する mp4 を出力
"""
from __future__ import annotations

import os
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(os.environ.get("WORKSPACE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT))

from src.diagram_blob.blob_renderer import render_diagram_anim  # type: ignore
from src.diagram_blob.blob_animate import resolve_blobs  # type: ignore


@dataclass
class SlideAnimConfig:
    slide_png: Path           # 完成形 slide PNG
    labels_npz: Path          # blob_split_v2 出力
    n_blobs: int              # 検出 blob 数
    n_groups: int = 10        # group 数
    anim_type: str = "fade_in"  # blob_renderer 対応の anim type
    duration: float = 3.0     # 全体尺(=target_finish_time)


def make_auto_order(blobs_v2_json: Path, n_groups: int, anim_type: str) -> dict:
    """blobs_v2.json から bbox 情報を取得し、blob_renderer 形式の order を作る
    各 blob には {"id", "x0", "y0", "x1", "y1"} が必要
    """
    with open(blobs_v2_json) as f:
        data = json.load(f)
    blobs_info = data["blobs"]
    n_blobs = len(blobs_info)
    group_size = max(1, (n_blobs + n_groups - 1) // n_groups)
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = min((g + 1) * group_size, n_blobs)
        if start >= end:
            break
        members = []
        for bi in range(start, end):
            bbox = blobs_info[bi]["bbox"]  # [x0, y0, x1, y1]
            members.append({
                "id": bi,
                "x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3],
            })
        groups.append({
            "resolved_blobs": [members],   # list of sub-list of dicts
            "anim": anim_type,
        })
    return {"groups": groups}


def _ensure_resolved(order: dict, blobs_v2_json: Path) -> dict:
    """orderer agent の `blob_filter` 形式を diagram_blob 公式 resolver で
    `resolved_blobs` (list of list of {id,x0,y0,x1,y1}) に変換する。
    既に resolved 済なら idempotent に素通し。

    Why: orderer の標準出力は `blob_filter: "manual:..."`、
    blob_renderer は `resolved_blobs` を要求 → 呼び出し側で公式 resolver
    (`src/diagram_blob/blob_animate.resolve_blobs`) を毎回挟む必要あり
    (過去事故の回避。 旧変換スクリプトは本ロジックに統合済)。
    """
    groups = order.get("groups", [])
    if groups:
        sample = groups[0].get("resolved_blobs")
        if isinstance(sample, list) and sample and isinstance(sample[0], list) \
                and sample[0] and isinstance(sample[0][0], dict) and "x0" in sample[0][0]:
            return order  # idempotent
    if not blobs_v2_json.exists():
        # 空 groups + blobs_v2 不在の保険(idempotent fast-path 漏れケース)
        return order
    # blobs_v2.json を all_blobs 形式に詰めて公式 resolver へ
    with open(blobs_v2_json) as f:
        v2 = json.load(f)
    diag_id = order.get("diagram_id") or blobs_v2_json.parent.name
    all_blobs = {diag_id: v2["blobs"]}
    order.setdefault("diagram_id", diag_id)
    # sort_z_order=False: manga の orderer agent は appear_t を意味順で手動指定する
    # ため Z字 再ソートで意図破壊を避ける
    return resolve_blobs(order, all_blobs, sort_z_order=False)


def render_slide_animation(
    slide_png: Path,
    labels_npz: Path,
    blobs_v2_json: Path,
    out_mp4: Path,
    target_finish_time: float = 3.0,
    n_groups: int = 10,
    anim_type: str = "fade_in",
    fps: int = 30,
    canvas_size: tuple[int, int] | None = None,
    order_json: Path | None = None,
) -> Path:
    """diagram_blob の blob_renderer で 3秒 progressive reveal アニメ mp4 を作る
    order_json があれば優先(orderer agent 出力)、無ければ auto chunk。
    orderer 出力が blob_filter 形式の場合は読込時に公式 resolver で resolved_blobs に変換。"""
    if order_json is not None and order_json.exists():
        with open(order_json) as f:
            order = json.load(f)
        order = _ensure_resolved(order, blobs_v2_json)
    else:
        order = make_auto_order(blobs_v2_json, n_groups, anim_type)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    render_diagram_anim(
        base_img=slide_png,
        labels_npz=labels_npz,
        order=order,
        duration=target_finish_time + 0.3,  # 末尾少しbuffer
        out_mp4=out_mp4,
        line_timings=None,
        canvas_size=canvas_size,
        fps=fps,
        bg_color=(255, 255, 255),
        auto_stagger=True,
        stagger_gap=target_finish_time / n_groups,
        initial_delay=0.0,
        target_finish_time=target_finish_time,
        max_reveal_cap=0.4,
    )
    return out_mp4


def get_slide_blobs_dir(slide_filename: str, blobs_root: Path) -> tuple[Path, Path, Path, int]:
    """blob_split_v2 が出力した <blobs_root>/<slug>/<slug>/ ディレクトリを返す
    Returns (dir, labels.npz, blobs_v2.json, n_blobs)"""
    slug = slide_filename.replace(".png", "")
    sub = blobs_root / slug / slug
    labels = sub / "labels.npz"
    blobs_v2 = sub / "blobs_v2.json"
    blobs = sorted(sub.glob("blob_*.png"))
    return sub, labels, blobs_v2, len(blobs)

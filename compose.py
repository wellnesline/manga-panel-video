"""compose — 漫画系 共通の動画ビルダー(音声first設計)

責務:
- シーン定義(scenes.py)を読み込み、各シーンの audio timing を実測
- silent動画を line-driven で frame-by-frame レンダリング
  - 漫画シーン: panel reveal は trigger_line ベース
  - 図解シーン: 最初3秒は diagram_blob の blob_renderer の slide animation mp4 を embed、その後静止
  - 字幕: 各 line の (start, end) で表示切替
  - 小窓: line ごと left_expr/right_expr で表情切替(cross-fade 0.4s)
- 最後に音声を mux → final.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("WORKSPACE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
from audio_timing import compute_line_timings, scene_total_duration, get_duration  # type: ignore
from slide_animator import (  # type: ignore
    render_slide_animation, get_slide_blobs_dir,
)
from audio_master import pre_bake_loudnorm, ALIMITER_SUFFIX  # type: ignore  # noqa: E402

W, H = 1920, 1080
FPS = 30
FONT_PATH = os.environ.get("SUBTITLE_FONT_PATH", "")  # 字幕用フォントのパス。環境変数で渡す(既定値なし)


def _load_font(size: int):
    if not FONT_PATH or not Path(FONT_PATH).exists():
        raise RuntimeError("字幕フォントが見つかりません。環境変数 SUBTITLE_FONT_PATH に .ttf/.ttc/.otf のパスを設定してください")
    return ImageFont.truetype(FONT_PATH, size)
FONT_SUBTITLE = _load_font(54)  # 字幕原設定
WB_SLIDE_RECT = (141, 24, 1778, 897)               # v8 wb_template の枠実測(上シフト 75px後)
WIPE_SIZE = 260   # 旧 300 → 260 にやや縮小 (slide content 遮蔽軽減)
WIPE_MARGIN = 15  # 旧 30 → 15 で枠ぎりぎりに寄せる(slide area の外角に近づける)
WIPE_POP_DUR = 0.35
WIPE_CROSSFADE_DUR = 0.4
SLIDE_ANIM_DUR = 3.0
SCENE_END_BUFFER = 1.5   # 各シーン末尾に 1.5秒 buffer (旧 0.5 では切替が早すぎた)
SUBTITLE_MAX_W_NARROW = W - 2 * (WIPE_MARGIN + WIPE_SIZE) - 80  # 2行以上=ワイプ完全回避幅(1180px)
SUBTITLE_MAX_W = W - 200                            # 1行=広い幅
SUBTITLE_Y_FROM_BOTTOM_1LINE = 50                   # 1行: 画面下から50px (調整で -10px 下げた)
SUBTITLE_Y_FROM_BOTTOM_MULTI = 15                   # 2行以上: 画面下に貼り付け、 -10px 下に

# 各種 cache
_img_cache: dict = {}
_page_cache: dict = {}
_sub_cache: dict = {}
_wb_static_cache: dict = {}
_wipe_cache: dict = {}


def load(path: Path) -> Image.Image:
    key = str(path)
    if key not in _img_cache:
        _img_cache[key] = Image.open(path).convert("RGBA")
    return _img_cache[key]


# ─────────────── auto crop + slide static ───────────────


def compute_crop_bbox(img: Image.Image, threshold: int = 700, m: int = 10):
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    nw = arr.astype(int).sum(axis=2) < threshold
    rows = np.where(nw.any(axis=1))[0]
    cols = np.where(nw.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return (0, 0, w, h)
    return (max(0, cols[0] - m), max(0, rows[0] - m),
            min(w, cols[-1] + m + 1), min(h, rows[-1] + m + 1))


def auto_crop_white(img: Image.Image) -> Image.Image:
    bbox = compute_crop_bbox(img)
    return img.crop(bbox)


def whiteboard_static(wb_template_path: Path, slide_path: Path) -> Image.Image:
    key = f"wb_static_{slide_path.name}"
    if key in _wb_static_cache:
        return _wb_static_cache[key]
    wb = load(wb_template_path).resize((W, H), Image.LANCZOS).copy()
    slide = load(slide_path)
    cropped = auto_crop_white(slide)
    x1, y1, x2, y2 = WB_SLIDE_RECT
    rw, rh = x2 - x1, y2 - y1
    sw, sh = cropped.size
    s = min(rw / sw, rh / sh)
    nw_, nh_ = int(sw * s), int(sh * s)
    resized = cropped.resize((nw_, nh_), Image.LANCZOS)
    wb.paste(resized, (x1 + (rw - nw_) // 2, y1 + (rh - nh_) // 2), resized)
    _wb_static_cache[key] = wb
    return wb


# ─────────────── Slide Animation via diagram_blob の blob_renderer ───────────────


def ensure_slide_animation(slide_path: Path, blobs_root: Path, work_dir: Path, fps: int = 30) -> Path:
    """slide の diagram_blob の blob_renderer アニメ mp4 を生成(キャッシュ前提)
    出力 mp4 size = native slide size (1792x1024)。後で whiteboard と合成時に縮小して貼る。

    cache invalidation (★ v4 事故由来):
      - order.json が anim.mp4 より新しい場合は cache を無効化して regenerate
      - 旧 v2 build で order.json 未生成のまま anim.mp4 だけ作られ、 v3 で order.json 後付けしても
        cache が hit して旧 anim.mp4 (auto fallback) が継続使用される事故を防ぐ
    """
    slide_name = slide_path.name
    out_mp4 = work_dir / f"{slide_name.replace('.png', '')}_anim.mp4"
    sub_dir, labels, blobs_v2, n_blobs = get_slide_blobs_dir(slide_name, blobs_root)
    if not labels.exists():
        raise FileNotFoundError(f"labels.npz not found: {labels}")
    # orderer agent 出力 (slide_blobs/<slug>/order.json) を優先利用
    order_json = sub_dir.parent / "order.json"

    if out_mp4.exists() and out_mp4.stat().st_size > 1000:
        anim_mtime = out_mp4.stat().st_mtime
        # order.json が anim.mp4 より新しいなら invalidate
        if order_json.exists() and order_json.stat().st_mtime > anim_mtime:
            print(f"  [slide-anim cache invalidate] {slide_name}: order.json newer than anim.mp4 → regenerate")
            out_mp4.unlink()
            # frames cache も invalidate
            frames_dir = work_dir / f"{slide_name.replace('.png', '')}_frames"
            if frames_dir.exists():
                for f in frames_dir.glob("frame_*.png"):
                    f.unlink()
        else:
            return out_mp4

    print(f"  [slide-anim] {slide_name}: {n_blobs} blobs → {out_mp4.name}"
          f" (order_json={'YES' if order_json.exists() else 'auto'})")
    render_slide_animation(
        slide_png=slide_path, labels_npz=labels, blobs_v2_json=blobs_v2,
        out_mp4=out_mp4, target_finish_time=SLIDE_ANIM_DUR,
        n_groups=10, anim_type="fade_in", fps=fps,
        order_json=order_json if order_json.exists() else None,
    )
    return out_mp4


def pre_extract_slide_anim_frames(slide_anim_mp4: Path, frames_dir: Path):
    """slide animation mp4 を frame_NNNN.png 群に pre-extract(idempotent)"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    existing = list(frames_dir.glob("frame_*.png"))
    if len(existing) > 10:
        return  # 既に extract 済
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(slide_anim_mp4),
        "-vf", f"fps={FPS}",
        str(frames_dir / "frame_%04d.png"),
    ], check=True)


def whiteboard_anim_frame(wb_template_path: Path, slide_path: Path,
                          slide_anim_frames_dir: Path, t_in_scene: float) -> Image.Image:
    """t_in_scene < 3秒: pre-extracted frame_NNNN.png を読んで wb_template に合成
    t_in_scene >= 3秒: whiteboard_static を返す
    """
    if t_in_scene >= SLIDE_ANIM_DUR:
        return whiteboard_static(wb_template_path, slide_path)
    frame_num = int(t_in_scene * FPS) + 1  # ffmpeg fps filter は 1-index
    cache_key = f"wb_anim_{slide_path.name}_{frame_num}"
    if cache_key in _wb_static_cache:
        return _wb_static_cache[cache_key]

    frame_path = slide_anim_frames_dir / f"frame_{frame_num:04d}.png"
    if not frame_path.exists():
        # ない場合は最終frame使用 or static
        frames = sorted(slide_anim_frames_dir.glob("frame_*.png"))
        if not frames:
            return whiteboard_static(wb_template_path, slide_path)
        frame_path = frames[-1]
    slide_frame = Image.open(frame_path).convert("RGBA")

    # auto_crop は完成形 slide で固定(位置不動)
    fixed_bbox = compute_crop_bbox(load(slide_path))
    cropped = slide_frame.crop(fixed_bbox)

    wb = load(wb_template_path).resize((W, H), Image.LANCZOS).copy()
    x1, y1, x2, y2 = WB_SLIDE_RECT
    rw, rh = x2 - x1, y2 - y1
    sw, sh = cropped.size
    s = min(rw / sw, rh / sh)
    nw_, nh_ = int(sw * s), int(sh * s)
    resized = cropped.resize((nw_, nh_), Image.LANCZOS)
    wb.paste(resized, (x1 + (rw - nw_) // 2, y1 + (rh - nh_) // 2), resized)
    _wb_static_cache[cache_key] = wb
    return wb


# ─────────────── Full-frame panel (assets_per_line) ───────────────


_full_panel_cache: dict[str, Image.Image] = {}


def render_full_panel(panel_path: Path) -> Image.Image:
    """1 line = 1 panel PNG をキャンバス W×H に letterbox 配置 (RGBA)。
    cached per-path."""
    key = str(panel_path)
    if key in _full_panel_cache:
        return _full_panel_cache[key]
    src = load(panel_path)
    pw, ph = src.size
    scale = min(W / pw, H / ph)
    nw = int(pw * scale)
    nh = int(ph * scale)
    resized = src.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
    canvas.paste(resized, ((W - nw) // 2, (H - nh) // 2), resized)
    _full_panel_cache[key] = canvas
    return canvas


_slot_panel_cache: dict[str, Image.Image] = {}


def fit_to_slot(panel_path: Path, slot_w: int, slot_h: int) -> Image.Image:
    """panel PNG を slot サイズに letterbox 配置 (RGBA、 黒枠付き)。 cached per (path, slot)。"""
    key = f"{panel_path}_{slot_w}x{slot_h}"
    if key in _slot_panel_cache:
        return _slot_panel_cache[key]
    src = load(panel_path)
    pw, ph = src.size
    inner_w = slot_w - 8  # 黒枠 4px×2
    inner_h = slot_h - 8
    scale = min(inner_w / pw, inner_h / ph)
    nw = int(pw * scale)
    nh = int(ph * scale)
    resized = src.resize((nw, nh), Image.LANCZOS)
    slot = Image.new("RGBA", (slot_w, slot_h), (252, 250, 245, 255))
    slot.paste(resized, ((slot_w - nw) // 2, (slot_h - nh) // 2), resized)
    # 黒枠
    d = ImageDraw.Draw(slot)
    d.rectangle((0, 0, slot_w - 1, slot_h - 1), outline=(20, 20, 20, 255), width=6)
    _slot_panel_cache[key] = slot
    return slot


def compute_page_slots(per_page: int) -> list[tuple[int, int, int, int]]:
    """1 page (W×H) を per_page 個の slot に分割。 戻り値: [(x, y, w, h), ...]
    margin 30px、 gap 20px。"""
    margin = 30
    gap = 20
    available_h = H - 2 * margin
    if per_page == 1:
        return [(margin, margin, W - 2 * margin, available_h)]
    if per_page == 2:
        slot_w = (W - 2 * margin - gap) // 2
        return [
            (margin, margin, slot_w, available_h),
            (margin + slot_w + gap, margin, slot_w, available_h),
        ]
    if per_page == 3:
        slot_w = (W - 2 * margin - 2 * gap) // 3
        return [
            (margin, margin, slot_w, available_h),
            (margin + slot_w + gap, margin, slot_w, available_h),
            (margin + 2 * (slot_w + gap), margin, slot_w, available_h),
        ]
    raise ValueError(f"unsupported per_page={per_page}")


def render_panel_layout(canvas: Image.Image, layout: str, apl: list[str],
                        line_timings: list, cur_idx: int, t_in: float,
                        refs_dir: Path) -> None:
    """layout に応じて canvas に panel を描画。
    - 1_per_page: 既存挙動 (full-frame + crossfade)
    - N_per_page: 1 page = N panel、 各 panel に 1 line。 line 切替で次 panel pop-in、
                  N line 経過で page 切替 (crossfade)。
    """
    if layout == "1_per_page":
        cur_path = refs_dir / apl[min(cur_idx, len(apl) - 1)]
        cur_img = render_full_panel(cur_path)
        cur_lt = line_timings[cur_idx]
        t_since = t_in - cur_lt.start
        crossfade = 0.25
        if cur_idx > 0 and t_since < crossfade:
            prev_path = refs_dir / apl[min(cur_idx - 1, len(apl) - 1)]
            prev_img = render_full_panel(prev_path)
            p = t_since / crossfade
            canvas.alpha_composite(alpha_modulate(prev_img, 1 - p))
            canvas.alpha_composite(alpha_modulate(cur_img, p))
        else:
            canvas.alpha_composite(cur_img)
        return

    if layout == "2_per_page":
        per_page = 2
    elif layout == "3_per_page":
        per_page = 3
    else:
        raise ValueError(f"unknown layout: {layout}")

    slots = compute_page_slots(per_page)
    page_idx = cur_idx // per_page
    slot_idx = cur_idx % per_page
    page_start = page_idx * per_page

    cur_lt = line_timings[cur_idx]
    t_since = t_in - cur_lt.start
    page_crossfade = 0.4
    pop_dur = 0.35

    # 前 page が crossfade 中なら、 前 page を fade out で描画
    if slot_idx == 0 and page_idx > 0 and t_since < page_crossfade:
        prev_page_start = (page_idx - 1) * per_page
        p = t_since / page_crossfade
        prev_layer = Image.new("RGBA", (W, H), (252, 250, 245, 0))
        for i in range(per_page):
            global_idx = prev_page_start + i
            if global_idx >= len(apl):
                break
            x, y, w, h = slots[i]
            panel_img = fit_to_slot(refs_dir / apl[global_idx], w, h)
            prev_layer.paste(panel_img, (x, y), panel_img)
        canvas.alpha_composite(alpha_modulate(prev_layer, 1 - p))
        # current page を fade in (まだ最初の slot のみ)
        cur_layer = Image.new("RGBA", (W, H), (252, 250, 245, 0))
        x, y, w, h = slots[0]
        panel_img = fit_to_slot(refs_dir / apl[page_start], w, h)
        cur_layer.paste(panel_img, (x, y), panel_img)
        canvas.alpha_composite(alpha_modulate(cur_layer, p))
        return

    # current page: 既出 slot + 現 slot pop_in
    for i in range(per_page):
        global_idx = page_start + i
        if global_idx > cur_idx or global_idx >= len(apl):
            break  # 未来 line は表示しない
        x, y, w, h = slots[i]
        panel_img = fit_to_slot(refs_dir / apl[global_idx], w, h)
        if global_idx == cur_idx and slot_idx > 0 and t_since < pop_dur:
            # pop_in アニメ (現 slot のみ)
            scale = anim_pop_scale(t_since, pop_dur)
            bw = max(8, int(w * scale))
            bh = max(8, int(h * scale))
            scaled = panel_img.resize((bw, bh), Image.LANCZOS)
            canvas.alpha_composite(
                scaled,
                (x + (w - bw) // 2, y + (h - bh) // 2),
            )
        else:
            canvas.alpha_composite(panel_img, (x, y))


# ─────────────── Panel detection (from v6/v7) ───────────────


def detect_panels(image: Image.Image):
    arr = np.array(image.convert("RGB"))
    iH, iW = arr.shape[:2]
    white = (arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240)
    col_ratio = white.mean(axis=0)
    is_v = col_ratio > 0.93
    cols = []
    in_col, cs = False, 0
    for x in range(iW):
        if not is_v[x]:
            if not in_col:
                cs, in_col = x, True
        else:
            if in_col and x - cs > iW * 0.1:
                cols.append((cs, x - 1))
                in_col = False
            elif in_col:
                in_col = False
    if in_col and iW - cs > iW * 0.1:
        cols.append((cs, iW - 1))
    panels = []
    for x1, x2 in cols:
        col_mask = white[:, x1:x2 + 1]
        row_ratio = col_mask.mean(axis=1)
        is_h = row_ratio > 0.93
        rows = []
        in_row, rs = False, 0
        for y in range(iH):
            if not is_h[y]:
                if not in_row:
                    rs, in_row = y, True
            else:
                if in_row and y - rs > iH * 0.1:
                    rows.append((rs, y - 1))
                    in_row = False
                elif in_row:
                    in_row = False
        if in_row and iH - rs > iH * 0.1:
            rows.append((rs, iH - 1))
        if not rows:
            rows = [(0, iH - 1)]
        for y1, y2 in rows:
            panels.append((x1, y1, x2, y2))
    return panels


def load_panels_from_json(asset_dir: Path, asset_name: str):
    """panels.json を読み込み panel dict のリストを返す。
    schema (後方互換):
      矩形のみ: {"bbox": [x0,y0,x1,y1]}
      多角形: {"polygon": [[x,y], [x,y], ...]}  ← 追加
    Returns: [{"bbox": tuple, "polygon": list[tuple]|None}, ...] or None
    """
    p = asset_dir / f"{asset_name.replace('.png', '')}_panels.json"
    if not p.exists():
        return None
    panels = []
    with open(p) as f:
        for x in json.load(f)["panels"]:
            if "polygon" in x and x["polygon"]:
                poly = [tuple(pt) for pt in x["polygon"]]
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                bbox = (min(xs), min(ys), max(xs), max(ys))
                panels.append({"bbox": bbox, "polygon": poly})
            else:
                panels.append({"bbox": tuple(x["bbox"]), "polygon": None})
    return panels


def get_page(asset_path: Path, panels_json_dir: Path):
    key = str(asset_path)
    if key in _page_cache:
        return _page_cache[key]
    src = load(asset_path)
    panels_native = load_panels_from_json(panels_json_dir, asset_path.name)
    if panels_native is None:
        # detect_panels の bbox-only 出力を dict に正規化
        panels_native = [{"bbox": b, "polygon": None} for b in detect_panels(src)]
    pw, ph = src.size
    scale = H / ph
    new_w = int(pw * scale)
    resized = src.resize((new_w, H), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
    ox = (W - new_w) // 2
    canvas.paste(resized, (ox, 0))
    panels_canvas = []
    for p in panels_native:
        b = p["bbox"]
        bbox_c = (int(b[0] * scale) + ox, int(b[1] * scale),
                  int(b[2] * scale) + ox, int(b[3] * scale))
        poly_c = ([(int(pt[0] * scale) + ox, int(pt[1] * scale)) for pt in p["polygon"]]
                  if p["polygon"] else None)
        panels_canvas.append({"bbox": bbox_c, "polygon": poly_c})
    _page_cache[key] = (canvas, panels_canvas)
    return canvas, panels_canvas


# ─────────────── Wipe ───────────────


def bust_up(asset_path: Path, size: int):
    src = load(asset_path).copy()
    bbox = src.getbbox()
    if bbox:
        src = src.crop(bbox)
    sw, sh = src.size
    head_h = int(sh * 0.28)
    head = src.crop((0, 0, sw, head_h))
    hw, hh = head.size
    if hw > hh:
        head = head.crop(((hw - hh) // 2, 0, (hw - hh) // 2 + hh, hh))
    elif hh > hw:
        head = head.crop((0, 0, hw, hw))
    return head.resize((size, size), Image.LANCZOS)


def polygon_mask(w, h, n=6):
    m = Image.new("L", (w, h), 0)
    cx, cy = w // 2, h // 2
    r = min(w, h) // 2 - 6
    pts = [(cx + r * math.cos(math.pi * 2 * i / n - math.pi / 2),
            cy + r * math.sin(math.pi * 2 * i / n - math.pi / 2)) for i in range(n)]
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return m


def make_wipe(asset_path: Path, size: int = WIPE_SIZE):
    key = f"w_{asset_path.name}_{size}"
    if key in _wipe_cache:
        return _wipe_cache[key]
    bust = bust_up(asset_path, size)
    bg = Image.new("RGBA", (size, size), (255, 255, 255, 255))
    bg.paste(bust, (0, 0), bust)
    mask = polygon_mask(size, size)
    result = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    result.paste(bg, (0, 0), mask)
    draw = ImageDraw.Draw(result)
    cx, cy = size // 2, size // 2
    r = size // 2 - 6
    pts = [(cx + r * math.cos(math.pi * 2 * i / 6 - math.pi / 2),
            cy + r * math.sin(math.pi * 2 * i / 6 - math.pi / 2)) for i in range(6)]
    draw.line(pts + [pts[0]], fill=(40, 40, 40, 255), width=5)
    _wipe_cache[key] = result
    return result


def anim_pop_scale(t, dur):
    if t < 0:
        return 0.0
    if t >= dur:
        return 1.0
    p = t / dur
    c1 = 1.70158
    return 1 + c1 * (p - 1) ** 3 + c1 * (p - 1) ** 2


def alpha_modulate(img: Image.Image, alpha: float):
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * alpha))
    return Image.merge("RGBA", (r, g, b, a))


def render_wipe_pair(canvas: Image.Image, scene: dict, line_timings: list,
                     t_in_scene: float, char_assets_dir: Path, skip_pop: bool = False):
    """各 line に left_expr / right_expr が定義されている前提
    skip_pop=True: scene開始時の pop アニメをスキップ(連続whiteboard用)
    """
    # 現在 line を特定
    cur_idx = None
    for lt in line_timings:
        if lt.start <= t_in_scene < lt.end_with_pause:
            cur_idx = lt.idx
            break
    if cur_idx is None:
        cur_idx = len(line_timings) - 1  # last line continued
    lines = scene["lines"]
    cur_left = lines[cur_idx].get("left_expr")
    cur_right = lines[cur_idx].get("right_expr")
    if not cur_left or not cur_right:
        return
    # 前 line で表情変化があったか
    prev_left = lines[cur_idx - 1].get("left_expr") if cur_idx > 0 else None
    prev_right = lines[cur_idx - 1].get("right_expr") if cur_idx > 0 else None
    cur_start = line_timings[cur_idx].start
    t_since = t_in_scene - cur_start

    left_pos = (WIPE_MARGIN, H - WIPE_MARGIN - WIPE_SIZE)
    right_pos = (W - WIPE_MARGIN - WIPE_SIZE, H - WIPE_MARGIN - WIPE_SIZE)

    # scene開始 pop(skip_pop時は飛ばす)
    if not skip_pop and t_in_scene < WIPE_POP_DUR:
        scale = anim_pop_scale(t_in_scene, WIPE_POP_DUR)
        bw = max(8, int(WIPE_SIZE * scale))
        left_wipe = make_wipe(char_assets_dir / f"{cur_left}.png").resize((bw, bw), Image.LANCZOS)
        right_wipe = make_wipe(char_assets_dir / f"{cur_right}.png").resize((bw, bw), Image.LANCZOS)
        canvas.alpha_composite(left_wipe, (left_pos[0] + (WIPE_SIZE - bw) // 2, left_pos[1] + (WIPE_SIZE - bw) // 2))
        canvas.alpha_composite(right_wipe, (right_pos[0] + (WIPE_SIZE - bw) // 2, right_pos[1] + (WIPE_SIZE - bw) // 2))
        return

    left_cur_wipe = make_wipe(char_assets_dir / f"{cur_left}.png")
    right_cur_wipe = make_wipe(char_assets_dir / f"{cur_right}.png")

    if prev_left and prev_left != cur_left and t_since < WIPE_CROSSFADE_DUR:
        p = t_since / WIPE_CROSSFADE_DUR
        left_prev_wipe = make_wipe(char_assets_dir / f"{prev_left}.png")
        canvas.alpha_composite(alpha_modulate(left_prev_wipe, 1 - p), left_pos)
        canvas.alpha_composite(alpha_modulate(left_cur_wipe, p), left_pos)
    else:
        canvas.alpha_composite(left_cur_wipe, left_pos)

    if prev_right and prev_right != cur_right and t_since < WIPE_CROSSFADE_DUR:
        p = t_since / WIPE_CROSSFADE_DUR
        right_prev_wipe = make_wipe(char_assets_dir / f"{prev_right}.png")
        canvas.alpha_composite(alpha_modulate(right_prev_wipe, 1 - p), right_pos)
        canvas.alpha_composite(alpha_modulate(right_cur_wipe, p), right_pos)
    else:
        canvas.alpha_composite(right_cur_wipe, right_pos)


# ─────────────── Panel reveal ───────────────


def panel_overlay(page, panel, eff, t, dur):
    """panel = {"bbox": (x0,y0,x1,y1), "polygon": list[(x,y)]|None} 形式
    後方互換: tuple/list の bbox-only も受ける"""
    progress = max(0.0, min(1.0, t / dur)) if dur > 0 else 1.0
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if isinstance(panel, dict):
        bbox = panel["bbox"]
        polygon = panel.get("polygon")
    else:
        bbox = panel
        polygon = None
    x1, y1, x2, y2 = bbox
    pw, ph = x2 - x1, y2 - y1
    if polygon:
        # polygon mask で page から切り抜き、 bbox で crop
        full_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(full_mask).polygon(polygon, fill=255)
        masked = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        masked.paste(page, (0, 0), full_mask)
        region = masked.crop((x1, y1, x2, y2))
    else:
        region = page.crop((x1, y1, x2, y2))
    if eff == "instant":
        overlay.paste(region, (x1, y1), region)
    elif eff == "fade_in":
        m = alpha_modulate(region, progress)
        overlay.paste(m, (x1, y1), m)
    elif eff == "slide_in_right":
        off = int((1 - progress) * pw)
        overlay.paste(region, (x1 + off, y1), region)
    elif eff == "slide_in_bottom":
        off = int((1 - progress) * ph)
        overlay.paste(region, (x1, y1 + off), region)
    elif eff == "zoom_in":
        s = 0.3 + 0.7 * progress
        nw, nh = int(pw * s), int(ph * s)
        rs = region.resize((nw, nh), Image.LANCZOS)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        m = alpha_modulate(rs, progress)
        overlay.paste(m, (cx - nw // 2, cy - nh // 2), m)
    elif eff == "shake_in":
        if progress < 1.0:
            rgn = alpha_modulate(region, progress)
        else:
            rgn = region
        dx = int(15 * (1 - progress) * math.sin(t * 50))
        dy = int(15 * (1 - progress) * math.cos(t * 47))
        overlay.paste(rgn, (x1 + dx, y1 + dy), rgn)
    return overlay


def apply_zoom_canvas(canvas, t, dur, s0, s1):
    p = min(1.0, t / dur)
    scale = s0 + (s1 - s0) * p
    nw, nh = int(W * scale), int(H * scale)
    rs = canvas.resize((nw, nh), Image.LANCZOS)
    return rs.crop(((nw - W) // 2, (nh - H) // 2, (nw - W) // 2 + W, (nh - H) // 2 + H))


# ─────────────── Subtitle ───────────────


def _wrap_all_lines(text: str, max_w: int, draw: ImageDraw.ImageDraw, font) -> list[str]:
    """text を max_w 内で wrap、 行数制限なしで全部分割して返す。"""
    out: list[str] = []
    remaining = text
    while remaining:
        if draw.textlength(remaining, font=font) <= max_w:
            out.append(remaining)
            break
        split_at = -1
        for sep in ["。", "、", " "]:
            for i in range(len(remaining) - 1, len(remaining) // 3, -1):
                if remaining[i] == sep:
                    if draw.textlength(remaining[: i + 1], font=font) <= max_w:
                        split_at = i + 1
                        break
            if split_at > 0:
                break
        if split_at <= 0:
            for i in range(len(remaining), 0, -1):
                if draw.textlength(remaining[:i], font=font) <= max_w:
                    split_at = i
                    break
        out.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return out


def make_sub_pages(text: str, has_wipe: bool = False):
    """text を 2 行ずつ複数 page に分割、 各 page の overlay と char count を返す。
    Returns: (list[Image], list[int])  ─ image と char count の同位 list
    page 切替は compose 側で char count 累積比率 × line_dur で計算 (char数比例 → 実 speech に近い切替)"""
    key = (text, has_wipe)
    if key in _sub_cache:
        return _sub_cache[key]
    font = FONT_SUBTITLE
    tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    max_w = SUBTITLE_MAX_W_NARROW if has_wipe else SUBTITLE_MAX_W
    all_lines = _wrap_all_lines(text, max_w, tmp_draw, font)
    if not all_lines:
        _sub_cache[key] = ([], [])
        return ([], [])
    pages: list[Image.Image] = []
    page_chars: list[int] = []
    line_h = 64
    for p_start in range(0, len(all_lines), 2):
        page_lines = all_lines[p_start:p_start + 2]
        char_count = sum(len(ln) for ln in page_lines)
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        y_off = SUBTITLE_Y_FROM_BOTTOM_1LINE if len(page_lines) == 1 else SUBTITLE_Y_FROM_BOTTOM_MULTI
        base_y = H - line_h * len(page_lines) - y_off
        for j, ln in enumerate(page_lines):
            bb = draw.textbbox((0, 0), ln, font=font)
            x = (W - (bb[2] - bb[0])) // 2
            y = base_y + j * line_h
            for dx in range(-5, 6, 2):
                for dy in range(-5, 6, 2):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), ln, fill=(0, 0, 0, 255), font=font)
            draw.text((x, y), ln, fill=(255, 255, 255, 255), font=font)
        pages.append(overlay)
        page_chars.append(char_count)
    _sub_cache[key] = (pages, page_chars)
    return (pages, page_chars)


def make_sub(text):
    """legacy 互換 wrapper: 最初の page 1 個だけ返す (使われない想定)"""
    pages, _ = make_sub_pages(text, has_wipe=True)
    if pages:
        return pages[0]
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return overlay


def draw_sub_page(canvas, text: str, has_wipe: bool, line_start: float, line_end: float, t_in: float):
    """指定 line 内の経過時間 (t_in - line_start) に応じて 2-行 page を切替表示。
    ★ page 切替は char count 累積比率に基づく (実 speech タイミングに近い切替)。
    例: 2 page で page1=30字, page2=20字 → 30/50=60% で切替 (8s line なら 4.8s で page2 に)"""
    if not text:
        return
    pages, page_chars = make_sub_pages(text, has_wipe=has_wipe)
    if not pages:
        return
    n = len(pages)
    if n == 1:
        canvas.alpha_composite(pages[0])
        return
    line_dur = max(0.001, line_end - line_start)
    elapsed = max(0.0, t_in - line_start)
    total_chars = max(1, sum(page_chars))
    # 累積 char ratio で 各 page の終了時刻を計算 → elapsed がどの page にあるか判定
    cum = 0
    idx = n - 1  # default: 最後の page (overshoot 防止)
    for i, cnt in enumerate(page_chars):
        cum += cnt
        end_t = (cum / total_chars) * line_dur
        if elapsed < end_t:
            idx = i
            break
    canvas.alpha_composite(pages[idx])


def draw_sub(canvas, text):
    """legacy 互換 (page 切替なし、 1 ページのみ)"""
    if text:
        canvas.alpha_composite(make_sub(text))


def text_without_tags(raw_text: str) -> str:
    """audio tag をテキストから除いた純粋な字幕表示用テキスト"""
    import re
    s = re.sub(r"\[[\s\S]+?\]", "", raw_text).strip()
    return s


# ─────────────── 全体 compose ───────────────


@dataclass
class SceneTiming:
    id: str
    kind: str
    start: float          # 動画内開始時刻
    end: float            # 動画内終了時刻
    line_timings: list    # list[LineTiming] (audio 内 timing) — kind=closing_cta なら []
    scene_def: dict       # 元 SCENES エントリ
    prev_was_whiteboard: bool = False  # 直前 scene が whiteboard か(wipe pop skip 用)


def build_timings(scenes_def: list[dict], scenes_audio_dir: Path) -> list[SceneTiming]:
    """各シーンの audio から timing を計測し、動画全体のタイムラインを構築"""
    result = []
    t_global = 0.0
    prev_kind: str | None = None
    for sd in scenes_def:
        scene_id = sd["id"]
        prev_wb = prev_kind == "whiteboard"
        if not sd.get("lines"):
            dur = sd.get("fixed_duration", 5.0)
            result.append(SceneTiming(
                id=scene_id, kind=sd["kind"],
                start=t_global, end=t_global + dur,
                line_timings=[], scene_def=sd,
                prev_was_whiteboard=prev_wb,
            ))
            t_global += dur
            prev_kind = sd["kind"]
            continue
        scene_subdir = scenes_audio_dir / scene_id
        lt = compute_line_timings(scene_subdir, sd["lines"])
        scene_dur = scene_total_duration(lt) + SCENE_END_BUFFER
        result.append(SceneTiming(
            id=scene_id, kind=sd["kind"],
            start=t_global, end=t_global + scene_dur,
            line_timings=lt, scene_def=sd,
            prev_was_whiteboard=prev_wb,
        ))
        t_global += scene_dur
        prev_kind = sd["kind"]
    return result


def render_scene_frame(st: SceneTiming, t: float,
                       refs_dir: Path, panels_json_dir: Path,
                       char_assets_dir: Path, slide_anim_dir: Path,
                       wb_template: Path) -> Image.Image:
    t_in = t - st.start
    dur = st.end - st.start
    scene = st.scene_def
    kind = st.kind

    if kind == "manga":
        canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
        if scene.get("assets_per_line"):
            layout = scene.get("layout", "1_per_page")
            apl = scene["assets_per_line"]
            n_lt = len(st.line_timings)
            # 現 line 特定
            cur_idx = None
            for i, lt in enumerate(st.line_timings):
                end_limit = (st.end - st.start) if i == n_lt - 1 else lt.end_with_pause
                if lt.start <= t_in < end_limit:
                    cur_idx = i
                    break
            if cur_idx is None:
                cur_idx = n_lt - 1
            render_panel_layout(canvas, layout, apl, st.line_timings, cur_idx, t_in, refs_dir)
        else:
            page_canvas, panels = get_page(refs_dir / scene["asset"], panels_json_dir)
            for pr in scene.get("panels_reveal", []):
                trig_line = pr["trigger_line"]
                if trig_line >= len(st.line_timings):
                    continue
                trig_t = st.line_timings[trig_line].start
                if t_in < trig_t:
                    continue
                t_in_eff = t_in - trig_t
                pidx = pr["panel_idx"]
                if pidx >= len(panels):
                    continue
                canvas.alpha_composite(panel_overlay(page_canvas, panels[pidx], pr["eff"], t_in_eff, pr["dur"]))
    elif kind == "whiteboard":
        slide_path = refs_dir / scene["slide"]
        slide_anim_frames_dir = slide_anim_dir / f"{scene['slide'].replace('.png', '')}_frames"
        canvas = whiteboard_anim_frame(wb_template, slide_path, slide_anim_frames_dir, t_in).copy()
    elif kind == "closing_cta":
        src = load(refs_dir / scene["asset"])
        pw, ph = src.size
        scale = H / ph
        new_w = int(pw * scale)
        if new_w >= W:
            scale = W / pw
            new_h = int(ph * scale)
            resized = src.resize((W, new_h), Image.LANCZOS)
            canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
            canvas.paste(resized, (0, (H - new_h) // 2))
        else:
            resized = src.resize((new_w, H), Image.LANCZOS)
            canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
            canvas.paste(resized, ((W - new_w) // 2, 0))
    else:
        canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

    ceff = scene.get("canvas_effect")
    if ceff == "zoom_slow":
        s0, s1 = scene.get("zoom", (0.97, 1.03))
        canvas = apply_zoom_canvas(canvas, t_in, dur, s0, s1)

    if canvas.mode != "RGBA":
        canvas = canvas.convert("RGBA")

    # subtitle: 現 line の時間内で表示 (scene.no_subtitle=True なら抑制)
    # 漫画系本編も「字幕方式」 に統一 (panel = 絵+強調ラベルのみ、 セリフは字幕レイヤー)
    # panel 側は STYLE_SUFFIX に NO speech bubbles 既設定。
    # 最終 line は SCENE_END_BUFFER 期間も継続表示 → scene 切替と同時に字幕も切替
    # ★ 2-line cap で長文の末尾切れないように page 切替 (line 内 elapsed で N 等分)
    # manga は default 字幕 OFF (吹き出しと二重表示で冗長)、 --manga-subtitle で ON 復帰
    suppress_manga = (kind == "manga" and not MANGA_SUBTITLE_ON)
    if st.line_timings and not scene.get("no_subtitle") and not suppress_manga:
        has_wipe = (kind == "whiteboard")
        n_lt = len(st.line_timings)
        for i, lt in enumerate(st.line_timings):
            end_limit = (st.end - st.start) if i == n_lt - 1 else lt.end_with_pause
            if lt.start <= t_in < end_limit:
                draw_sub_page(canvas, text_without_tags(lt.text), has_wipe,
                              lt.start, end_limit, t_in)
                break

    # wipe: whiteboard 時のみ。直前 scene も whiteboard なら pop skip
    if kind == "whiteboard":
        prev_was_wb = bool(st.prev_was_whiteboard)
        render_wipe_pair(canvas, scene, st.line_timings, t_in, char_assets_dir, skip_pop=prev_was_wb)

    # 動画開始/終了 fade 廃止(ハードカット)

    return canvas


SCENE_TIMINGS: list[SceneTiming] = []   # main()内で書き換え
MANGA_SUBTITLE_ON: bool = True  # default ON 化 (字幕方式統一)、 旧動作は --no-manga-subtitle で復帰可


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-py", type=Path, required=True, help="scenes.py module")
    ap.add_argument("--refs-dir", type=Path, required=True, help="manga/slide PNG dir")
    ap.add_argument("--audio-dir", type=Path, required=True, help="per-line audio cache (scenes/<id>/)")
    ap.add_argument("--char-assets-dir", type=Path, required=True, help="<char_a>_*.png / <char_b>_*.png")
    ap.add_argument("--panels-json-dir", type=Path, required=True, help="*_panels.json")
    ap.add_argument("--slide-blobs-dir", type=Path, required=True, help="blob_split_v2 output root")
    ap.add_argument("--wb-template", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output mp4")
    ap.add_argument("--bgm", type=Path, help="BGM mp3 path (optional, volume=0.10 amix)")
    ap.add_argument("--bgm-volume", type=float, default=0.10, help="BGM volume (default 0.10)")
    ap.add_argument("--bypass-fact-check", action="store_true",
                    help="ファクトチェック必須化をスキップ(緊急時のみ)。 通常は fact_check.yaml verdict=PASS が必須")
    ap.add_argument("--voice-prefix", type=str, default="",
                    help="Voice file prefix (e.g. 'v2_' → review/voice_v2_<id>.mp3). 既定空=旧版互換")
    ap.add_argument("--manga-subtitle", action="store_true",
                    help="(後方互換) manga scene 字幕 ON。 以降 default ON のため通常指定不要")
    ap.add_argument("--no-manga-subtitle", action="store_true",
                    help="manga scene の字幕を OFF (旧 default 復帰用、 旧版互換)。 whiteboard scene は常に ON")
    args = ap.parse_args()

    global MANGA_SUBTITLE_ON
    # default True、 --no-manga-subtitle で OFF、 --manga-subtitle 明示でも ON (後方互換)
    if args.no_manga_subtitle:
        MANGA_SUBTITLE_ON = False
    elif args.manga_subtitle:
        MANGA_SUBTITLE_ON = True
    # else default (True)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    slide_anim_dir = args.work_dir / "slide_anim"
    slide_anim_dir.mkdir(exist_ok=True)
    silent_mp4 = args.work_dir / "silent.mp4"

    # scenes.py を import
    sys.path.insert(0, str(args.scenes_py.parent))
    scenes_mod = __import__(args.scenes_py.stem)
    SCENES_DEF = scenes_mod.SCENES

    # ─── Fact-check gate (過去事故由来 永続化) ───────────────
    # videos/<slug>/fact_check.yaml verdict=PASS が必須
    # スキップしたい場合は環境変数 FACT_CHECK_BYPASS=1
    if not args.bypass_fact_check and not os.environ.get("FACT_CHECK_BYPASS"):
        fc_yaml = args.scenes_py.parent / "fact_check.yaml"
        if not fc_yaml.exists():
            print(f"[FACT-BLOCK] {fc_yaml} が存在しません。"
                  " manga-fact-checker agent を起動し、 fact_check.yaml verdict=PASS を取得してから compose してください。"
                  " 緊急回避: --bypass-fact-check または env FACT_CHECK_BYPASS=1",
                  file=sys.stderr)
            sys.exit(2)
        try:
            import yaml as _yaml
        except ImportError:
            print("[FACT-BLOCK] PyYAML 未インストール: pip install pyyaml", file=sys.stderr)
            sys.exit(2)
        fc = _yaml.safe_load(fc_yaml.read_text()) or {}
        verdict = (fc.get("verdict") or "").upper()
        if verdict != "PASS":
            print(f"[FACT-BLOCK] fact_check.yaml verdict={verdict!r} (PASS必須)", file=sys.stderr)
            sys.exit(2)
        if not fc.get("checked_at") or not fc.get("evidence_urls"):
            print("[FACT-BLOCK] fact_check.yaml に checked_at / evidence_urls 必須", file=sys.stderr)
            sys.exit(2)
        print(f"[FACT-PASS] {fc_yaml.name} verdict=PASS / checked_at={fc.get('checked_at')}")

    # ─── Slide order.json gate (★ 確定、 アニメ事故由来) ────────
    # 全 whiteboard scene の slide blobs dir 直下に order.json が必要。
    # orderer または orderer agent で事前生成すること。
    # 緊急回避: env ORDER_JSON_BYPASS=1
    if not os.environ.get("ORDER_JSON_BYPASS"):
        missing_orders: list[str] = []
        for sd in SCENES_DEF:
            if sd.get("kind") != "whiteboard":
                continue
            slide_name = sd["slide"]
            slug = slide_name.replace(".png", "")
            order_json = args.slide_blobs_dir / slug / "order.json"
            if not order_json.exists():
                missing_orders.append(str(order_json))
        if missing_orders:
            print("[ORDER-BLOCK] whiteboard scene の order.json が見つかりません:", file=sys.stderr)
            for p in missing_orders:
                print(f"  - {p}", file=sys.stderr)
            print(
                "  orderer agent (5 slide まで一括、 または 1 slide ずつ) を起動して "
                "order.json を生成してから compose してください。 緊急回避: env ORDER_JSON_BYPASS=1",
                file=sys.stderr,
            )
            sys.exit(2)
        n_wb = sum(1 for sd in SCENES_DEF if sd.get("kind") == "whiteboard")
        if n_wb > 0:
            print(f"[ORDER-PASS] 全 {n_wb} whiteboard slide で order.json 確認済")

    # pre-render slide animations (diagram_blob の blob_renderer) + frame extract
    print("=== Pre-rendering slide animations ===")
    for sd in SCENES_DEF:
        if sd["kind"] != "whiteboard":
            continue
        slide_path = args.refs_dir / sd["slide"]
        anim_mp4 = ensure_slide_animation(slide_path, args.slide_blobs_dir, slide_anim_dir)
        # frame をPNG群に pre-extract(compose時の高速参照用)
        frames_dir = slide_anim_dir / f"{sd['slide'].replace('.png', '')}_frames"
        pre_extract_slide_anim_frames(anim_mp4, frames_dir)
        n_frames = len(list(frames_dir.glob("frame_*.png")))
        print(f"  -> {frames_dir.name}: {n_frames} frames extracted")

    # build timings
    print("\n=== Building scene timings from audio ===")
    global SCENE_TIMINGS
    SCENE_TIMINGS = build_timings(SCENES_DEF, args.audio_dir)
    for st in SCENE_TIMINGS:
        n_lines = len(st.line_timings)
        print(f"  {st.id:5s} {st.start:7.2f} - {st.end:7.2f} ({st.end - st.start:5.2f}s, {n_lines} lines)")
    total_dur = SCENE_TIMINGS[-1].end
    print(f"\nTotal video duration: {total_dur:.2f}s")

    # render silent video (frame-by-frame stream pipe)
    total_frames = int(total_dur * FPS)
    print(f"\n=== Rendering silent video ({total_frames} frames @ {FPS}fps) ===")
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast",
        str(silent_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    si = 0
    for i in range(total_frames):
        t = i / FPS
        while si + 1 < len(SCENE_TIMINGS) and t >= SCENE_TIMINGS[si + 1].start:
            si += 1
        frame = render_scene_frame(
            SCENE_TIMINGS[si], t,
            args.refs_dir, args.panels_json_dir, args.char_assets_dir,
            slide_anim_dir, args.wb_template,
        )
        proc.stdin.write(frame.convert("RGB").tobytes())
        if i % 60 == 0:
            print(f"  frame {i}/{total_frames} (t={t:.1f}s, {SCENE_TIMINGS[si].id})")
    proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg error: {err[-1500:]}", file=sys.stderr)
        sys.exit(1)
    print(f"silent video: {silent_mp4} ({silent_mp4.stat().st_size // (1024 * 1024)}MB)")

    # build & mux audio (各 scene mp3 を scene start に配置、scene末buffer分silence)
    print("\n=== Building audio track ===")
    audio_concat_list = args.work_dir / "audio_concat.txt"
    padded_audios = []
    for st in SCENE_TIMINGS:
        scene_dur = st.end - st.start
        wav = args.work_dir / f"audio_{st.id}.wav"
        src_mp3 = ROOT / "review" / f"voice_{args.voice_prefix}{st.id}.mp3"
        if not st.line_timings or not src_mp3.exists():
            # lines なし or audio未生成 → 無音
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"anullsrc=duration={scene_dur}:r=44100:cl=stereo",
                "-t", str(scene_dur), "-c:a", "pcm_s16le", str(wav),
            ], check=True)
        else:
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src_mp3),
                "-af", f"atrim=duration={scene_dur},asetpts=PTS-STARTPTS,apad=whole_dur={scene_dur}",
                "-t", str(scene_dur), "-c:a", "pcm_s16le", str(wav),
            ], check=True)
        padded_audios.append(wav)

    with open(audio_concat_list, "w") as f:
        for w in padded_audios:
            f.write(f"file '{w.absolute()}'\n")
    full_audio = args.work_dir / "full_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(audio_concat_list),
        "-c:a", "pcm_s16le", str(full_audio),
    ], check=True)

    # 2-pass loudnorm で voice トラックを I=-16 LUFS に正規化
    # (漫画系統合)
    print("\n=== Voice loudnorm (2-pass, I=-16 LUFS) ===")
    full_audio_norm = pre_bake_loudnorm(full_audio)

    # BGM mix (optional) + 最終 alimiter で真のピーク制御
    if args.bgm and args.bgm.exists():
        print(f"\n=== Mixing BGM ({args.bgm.name}, vol={args.bgm_volume}) + alimiter ===")
        bgm_wav = args.work_dir / "bgm.wav"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-stream_loop", "-1", "-i", str(args.bgm),
            "-t", str(total_dur),
            "-af", f"volume={args.bgm_volume}",
            "-c:a", "pcm_s16le", str(bgm_wav),
        ], check=True)
        mixed_audio = args.work_dir / "mixed_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(full_audio_norm), "-i", str(bgm_wav),
            "-filter_complex",
            f"[0:a][1:a]amix=inputs=2:normalize=0:duration=first,{ALIMITER_SUFFIX}[out]",
            "-map", "[out]", "-c:a", "pcm_s16le", str(mixed_audio),
        ], check=True)
        final_audio = mixed_audio
    else:
        # BGM なしでも alimiter で真ピーク制御
        limited_audio = args.work_dir / "limited_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(full_audio_norm),
            "-af", ALIMITER_SUFFIX,
            "-c:a", "pcm_s16le", str(limited_audio),
        ], check=True)
        final_audio = limited_audio

    print("\n=== Muxing video + audio → final ===")
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent_mp4), "-i", str(final_audio),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a", "-shortest",
        str(args.out),
    ], check=True)
    print(f"\n-> {args.out} ({args.out.stat().st_size // (1024 * 1024)}MB)")


if __name__ == "__main__":
    main()

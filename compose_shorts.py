"""compose_shorts — 漫画系 用 縦型 Shorts ビルダー

責務:
- shorts_scenes.py (P1〜P5 + SCTA) を入力
- 各シーンの per-line audio を実測で timing 計算
- 縦型 1080×1920 で frame-by-frame レンダ
  - 漫画ページ: 中央クロップ+縦埋め(横を切る)
  - 字幕: 大型 (font 80px) 中央寄せ、画面下40%
  - ワイプなし
- SCTA scene: CTA テロップ(文言は scenes 側で指定)を全画面オーバーレイ
- 最後に BGM mix + audio mux → final shorts mp4
"""
from __future__ import annotations

import os
import argparse
import json as _json_mod
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(os.environ.get("WORKSPACE_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_timing import compute_line_timings, scene_total_duration  # type: ignore

W, H = 1080, 1920
FPS = 30
FONT_PATH = os.environ.get("SUBTITLE_FONT_PATH", "")  # 字幕用フォントのパス。環境変数で渡す(既定値なし)


def _load_font(size: int):
    if not FONT_PATH or not Path(FONT_PATH).exists():
        raise RuntimeError("字幕フォントが見つかりません。環境変数 SUBTITLE_FONT_PATH に .ttf/.ttc/.otf のパスを設定してください")
    return ImageFont.truetype(FONT_PATH, size)
FONT_SUB = _load_font(78)
FONT_CTA = _load_font(120)
SUBTITLE_FILL = (245, 245, 245, 255)  # サンプル値(字幕色)
CTA_FILL = (200, 230, 255, 255)       # サンプル値(CTA 文字色)
SUB_MAX_W = W - 80
SUB_GAP_BELOW_IMG = 40            # 画像の真下から字幕までの余白
SCENE_END_BUFFER = 0.4
PAGE_OFFSET_Y_FROM_TOP = 200      # 漫画ページの上余白(コンテンツ全体を 9:12 風に中央寄せ)
_page_layout: dict[str, tuple[int, int, int]] = {}  # asset_path -> (oy, new_h, image_bottom_y)

_img_cache: dict = {}
_page_cache: dict = {}
_sub_cache: dict = {}


def load(p: Path) -> Image.Image:
    k = str(p)
    if k not in _img_cache:
        _img_cache[k] = Image.open(p).convert("RGBA")
    return _img_cache[k]


# ─── panel detection (compose.py から移植) ───
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


def load_panels_json(asset_dir: Path, asset_name: str):
    import json as _json
    p = asset_dir / f"{asset_name.replace('.png', '')}_panels.json"
    if not p.exists():
        return None
    with open(p) as f:
        return [tuple(x["bbox"]) for x in _json.load(f)["panels"]]


def get_page_vert(asset_path: Path, panels_json_dir: Path | None = None):
    """漫画ページを縦型1080×Hに配置(v2 と同じロジック)。
    W に合わせて scale → 中央より少し上シフト(下に字幕領域確保)。
    Returns (canvas_full_resolved, panels_canvas, image_bottom_y)
    """
    k = str(asset_path)
    if k in _page_cache:
        return _page_cache[k]
    src = load(asset_path)
    pw, ph = src.size
    scale = W / pw
    new_h = int(ph * scale)
    resized = src.resize((W, new_h), Image.LANCZOS)
    canvas_full = Image.new("RGBA", (W, H), (252, 250, 245, 255))
    if new_h >= H:
        oy = (new_h - H) // 2
        canvas_full.paste(resized.crop((0, oy, W, oy + H)), (0, 0))
        oy_paste = 0  # paste位置(crop後は左上0)
        # bbox 計算用には -oy 補正
        oy_for_panels = -oy
        image_bottom_y = H
    else:
        # v2 と同じ: 中央配置から 280px 上にシフト
        oy_paste = max(0, (H - new_h) // 2 - 280)
        canvas_full.paste(resized, (0, oy_paste))
        oy_for_panels = oy_paste
        image_bottom_y = oy_paste + new_h
    # panels (元座標) → canvas 座標
    panels_native = load_panels_json(panels_json_dir, asset_path.name) if panels_json_dir else None
    if panels_native is None:
        panels_native = detect_panels(src)
    panels_canvas = [
        (int(p[0] * scale), int(p[1] * scale) + oy_for_panels,
         int(p[2] * scale), int(p[3] * scale) + oy_for_panels)
        for p in panels_native
    ]
    _page_cache[k] = (canvas_full, panels_canvas, image_bottom_y)
    return _page_cache[k]


# ─── panel reveal アニメ (compose.py から移植) ───
def alpha_modulate(img: Image.Image, alpha: float):
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * alpha))
    return Image.merge("RGBA", (r, g, b, a))


def panel_overlay(page: Image.Image, bbox, eff: str, t: float, dur: float):
    progress = max(0.0, min(1.0, t / dur)) if dur > 0 else 1.0
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x1, y1, x2, y2 = bbox
    pw, ph = x2 - x1, y2 - y1
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


# ─── subtitle (大型、画像の真下に動的配置) ───
def make_sub(text: str, base_y: int) -> Image.Image:
    cache_key = f"{text}|{base_y}"
    if cache_key in _sub_cache:
        return _sub_cache[cache_key]
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = FONT_SUB
    max_w = SUB_MAX_W
    lines = []
    remaining = text
    while remaining:
        if draw.textlength(remaining, font=font) <= max_w:
            lines.append(remaining); break
        split_at = -1
        for sep in ["。", "、", " "]:
            for i in range(len(remaining) - 1, len(remaining) // 3, -1):
                if remaining[i] == sep and draw.textlength(remaining[:i + 1], font=font) <= max_w:
                    split_at = i + 1
                    break
            if split_at > 0:
                break
        if split_at <= 0:
            for i in range(len(remaining), 0, -1):
                if draw.textlength(remaining[:i], font=font) <= max_w:
                    split_at = i; break
        lines.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
        if len(lines) >= 4:
            lines.append(remaining); break
    line_h = 96
    total_h = line_h * len(lines)
    pad = 24
    bg_y1 = base_y - pad
    bg_y2 = base_y + total_h + pad
    draw.rectangle([0, bg_y1, W, bg_y2], fill=(0, 0, 0, 180))
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (W - (bbox[2] - bbox[0])) // 2
        y = base_y + i * line_h
        for dx in range(-5, 6, 2):
            for dy in range(-5, 6, 2):
                if dx == 0 and dy == 0: continue
                draw.text((x + dx, y + dy), line, fill=(0, 0, 0, 255), font=font)
        draw.text((x, y), line, fill=SUBTITLE_FILL, font=font)
    _sub_cache[cache_key] = overlay
    return overlay


def draw_sub(canvas: Image.Image, text: str, image_bottom_y: int):
    if text:
        base_y = image_bottom_y + SUB_GAP_BELOW_IMG
        canvas.alpha_composite(make_sub(text, base_y))


def text_without_tags(raw: str) -> str:
    import re
    return re.sub(r"\[[\s\S]+?\]", "", raw).strip()


# ─── CTA ───
def draw_cta(canvas: Image.Image, cta_text: str, sub_text: str = ""):
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # 中央に CTA 大文字
    bbox = draw.textbbox((0, 0), cta_text, font=FONT_CTA)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (W - tw) // 2
    y = (H - th) // 2 - 50
    # 黒地ボックス
    pad = 50
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad * 2], fill=(0, 0, 0, 220))
    for dx in range(-6, 7, 2):
        for dy in range(-6, 7, 2):
            if dx == 0 and dy == 0: continue
            draw.text((x + dx, y + dy), cta_text, fill=(0, 0, 0, 255), font=FONT_CTA)
    draw.text((x, y), cta_text, fill=CTA_FILL, font=FONT_CTA)
    # 下部小文字
    if sub_text:
        bbox2 = draw.textbbox((0, 0), sub_text, font=FONT_SUB)
        tw2 = bbox2[2] - bbox2[0]
        x2 = (W - tw2) // 2
        y2 = y + th + pad * 2 + 40
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                if dx == 0 and dy == 0: continue
                draw.text((x2 + dx, y2 + dy), sub_text, fill=(0, 0, 0, 255), font=FONT_SUB)
        draw.text((x2, y2), sub_text, fill=(255, 255, 255, 255), font=FONT_SUB)
    canvas.alpha_composite(overlay)


# ─── compose ───
@dataclass
class SceneTiming:
    id: str
    kind: str
    start: float
    end: float
    line_timings: list
    scene_def: dict


def build_timings(scenes_def: list[dict], audio_dir: Path) -> list[SceneTiming]:
    res = []
    t = 0.0
    for sd in scenes_def:
        sid = sd["id"]
        if not sd.get("lines"):
            dur = sd.get("fixed_duration", 3.0)
            res.append(SceneTiming(sid, sd["kind"], t, t + dur, [], sd))
            t += dur
            continue
        sub = audio_dir / sid
        lt = compute_line_timings(sub, sd["lines"])
        sdur = scene_total_duration(lt) + SCENE_END_BUFFER
        res.append(SceneTiming(sid, sd["kind"], t, t + sdur, lt, sd))
        t += sdur
    return res


def render_cta_centered(asset_path: Path) -> Image.Image:
    """CTA画像を 1080×H の縦型キャンバス中央(縦方向)に配置。
    画像比率は元の(ほぼ正方形)を維持、字幕やテロップは追加しない。
    """
    src = load(asset_path)
    pw, ph = src.size
    scale = W / pw
    new_h = int(ph * scale)
    resized = src.resize((W, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
    oy = max(0, (H - new_h) // 2)
    canvas.paste(resized, (0, oy))
    return canvas


def render_frame(st: SceneTiming, t: float, refs_dir: Path, panels_json_dir: Path) -> Image.Image:
    t_in = t - st.start
    scene = st.scene_def
    kind = scene["kind"]

    if kind == "shorts_cta":
        # CTA: 専用画像のみ。字幕・テロップなし。中央配置。
        return render_cta_centered(refs_dir / scene["asset"]).convert("RGBA")

    if kind == "manga":
        page_canvas, panels, image_bottom_y = get_page_vert(refs_dir / scene["asset"], panels_json_dir)
        # 漫画 reveal: panels_reveal の trigger_line ベース
        canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
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
    else:
        canvas = Image.new("RGBA", (W, H), (252, 250, 245, 255))
        image_bottom_y = H // 2

    if st.line_timings and not scene.get("no_subtitle"):
        cur = ""
        n = len(st.line_timings)
        for i, lt in enumerate(st.line_timings):
            end_limit = (st.end - st.start) if i == n - 1 else lt.end_with_pause
            if lt.start <= t_in < end_limit:
                cur = text_without_tags(lt.text)
                break
        draw_sub(canvas, cur, image_bottom_y)

    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-py", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--panels-json-dir", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bgm", type=Path)
    ap.add_argument("--bgm-volume", type=float, default=0.15)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    silent = args.work_dir / "shorts_silent.mp4"

    sys.path.insert(0, str(args.scenes_py.parent))
    mod = __import__(args.scenes_py.stem)
    SCENES = mod.SCENES

    print("=== Building timings ===")
    sts = build_timings(SCENES, args.audio_dir)
    for st in sts:
        print(f"  {st.id:5s} {st.start:7.2f}-{st.end:7.2f} ({st.end-st.start:5.2f}s)")
    total = sts[-1].end
    print(f"Total: {total:.2f}s")

    total_frames = int(total * FPS)
    print(f"\n=== Rendering silent ({total_frames} frames) ===")
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", str(silent)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    si = 0
    for i in range(total_frames):
        t = i / FPS
        while si + 1 < len(sts) and t >= sts[si + 1].start:
            si += 1
        f = render_frame(sts[si], t, args.refs_dir, args.panels_json_dir)
        proc.stdin.write(f.convert("RGB").tobytes())
        if i % 90 == 0:
            print(f"  frame {i}/{total_frames} (t={t:.1f}s, {sts[si].id})")
    proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg error: {err[-1500:]}", file=sys.stderr); sys.exit(1)

    # build audio
    print("\n=== Building audio ===")
    concat_list = args.work_dir / "shorts_audio_concat.txt"
    wavs = []
    for st in sts:
        sd = st.end - st.start
        wav = args.work_dir / f"shorts_audio_{st.id}.wav"
        src_mp3 = ROOT / "review" / f"voice_{st.id}.mp3"
        if not st.line_timings or not src_mp3.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error",
                            "-f", "lavfi", "-i", f"anullsrc=duration={sd}:r=44100:cl=stereo",
                            "-t", str(sd), "-c:a", "pcm_s16le", str(wav)], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_mp3),
                            "-af", f"atrim=duration={sd},asetpts=PTS-STARTPTS,apad=whole_dur={sd}",
                            "-t", str(sd), "-c:a", "pcm_s16le", str(wav)], check=True)
        wavs.append(wav)
    with open(concat_list, "w") as f:
        for w in wavs: f.write(f"file '{w.absolute()}'\n")
    full_audio = args.work_dir / "shorts_full_audio.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", str(concat_list), "-c:a", "pcm_s16le", str(full_audio)], check=True)

    final_audio = full_audio
    if args.bgm and args.bgm.exists():
        print(f"\n=== Mixing BGM ===")
        bgm_wav = args.work_dir / "shorts_bgm.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-stream_loop", "-1",
                        "-i", str(args.bgm), "-t", str(total), "-af", f"volume={args.bgm_volume}",
                        "-c:a", "pcm_s16le", str(bgm_wav)], check=True)
        mixed = args.work_dir / "shorts_mixed.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(full_audio), "-i", str(bgm_wav),
                        "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0:duration=first[out]",
                        "-map", "[out]", "-c:a", "pcm_s16le", str(mixed)], check=True)
        final_audio = mixed

    print("\n=== Mux video+audio → final ===")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(final_audio),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-map", "0:v", "-map", "1:a",
                    "-shortest", str(args.out)], check=True)
    print(f"\n-> {args.out} ({args.out.stat().st_size//(1024*1024)}MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""マンガ系 Shorts 向け 縦型 (1080×1920) build スクリプト。

レイアウト(サンプル値。下の定数で変更できる):
  - 上部の余白 y=0-280
  - パネル領域 y=280-1360 (幅 1080。パネル画像は領域内に letterbox で配置)
  - 字幕帯 y=1380-1500 (半透明の帯 + 字幕色 SUBT_COLOR。パネル領域の直下で重ねない)
  - 下部の余白 y=1480-1920
  - 背景色 BG_COLOR は余白部分と統一する

入力: scenes.py の SHORTS (本編 compose と別 schema)、 audio_scenes/<sid>/run<NN>_<sp>_(single|line<K>).mp3
出力: 1080×1920 mp4

末尾 line の文言(CTA)は scenes.py 側で指定する。
"""
from __future__ import annotations
import argparse
import os
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 音声マスタリング統合 (漫画系統合)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
from audio_master import pre_bake_loudnorm, ALIMITER_SUFFIX  # noqa: E402

W, H = 1080, 1920
FPS = 30
FONT_PATH = os.environ.get("SUBTITLE_FONT_PATH", "")  # 字幕用フォントのパス。環境変数で渡す(既定値なし)
BG_COLOR = (24, 28, 40, 255)        # サンプル値(背景色。余白部分を含む)
SUBT_COLOR = (240, 240, 240, 255)   # サンプル値(字幕色)

# レイアウトのサンプル値
PANEL_Y1, PANEL_Y2 = 280, 1360      # パネル領域
SUBT_Y1, SUBT_Y2 = 1380, 1500       # 字幕帯(パネル領域の直下)
SUBT_FONT_SIZE = 58
SCENE_END_BUFFER = 0.8

_TAG_RE = re.compile(r"\[[\s\S]+?\]\s*")


def text_without_tags(s: str) -> str:
    return _TAG_RE.sub("", s).strip()


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGBA")


def get_duration(p: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


@dataclass
class LineTiming:
    idx: int
    speaker: str
    text: str
    start: float          # scene 内 t
    end: float            # scene 内 t (発話のみ)
    end_with_pause: float
    mp3: Path


def find_line_mp3(audio_dir: Path, sid: str, idx: int, speaker: str,
                  speaker_indices: list[tuple[int, int]]) -> Path:
    """run grouping (連続 speaker) を再構成して、 line idx → mp3 を解決"""
    # 同 speaker 連続 line の run_idx と k (run 内位置) を計算
    run_idx = 0
    run_start = 0
    runs: list[tuple[str, list[int]]] = []  # (speaker, [line_indices])
    for i, sp in speaker_indices:
        if runs and runs[-1][0] == sp:
            runs[-1][1].append(i)
        else:
            runs.append((sp, [i]))
    for ri, (sp, indices) in enumerate(runs):
        if idx in indices:
            k = indices.index(idx)
            n = len(indices)
            scene_dir = audio_dir / sid
            if n == 1:
                return scene_dir / f"run{ri:02d}_{sp}_single.mp3"
            return scene_dir / f"run{ri:02d}_{sp}_line{k}.mp3"
    raise FileNotFoundError(f"line {idx} ({speaker}) not found in {sid}")


def compute_line_timings(audio_dir: Path, sid: str, lines: list[dict]) -> list[LineTiming]:
    speaker_indices = [(i, ld["speaker"]) for i, ld in enumerate(lines)]
    timings: list[LineTiming] = []
    t = 0.0
    for i, ld in enumerate(lines):
        mp3 = find_line_mp3(audio_dir, sid, i, ld["speaker"], speaker_indices)
        dur = get_duration(mp3)
        pause = float(ld.get("pause_after", 0.3))
        timings.append(LineTiming(
            idx=i, speaker=ld["speaker"], text=ld["text"],
            start=t, end=t + dur, end_with_pause=t + dur + pause,
            mp3=mp3,
        ))
        t += dur + pause
    return timings


# ─────────────── Panel render ───────────────

_panel_cache: dict[str, Image.Image] = {}


def render_panel(panel_path: Path) -> Image.Image:
    """panel PNG を PANEL_Y2-PANEL_Y1 高さの領域に letterbox 配置 (RGBA、 W×(Y2-Y1) サイズ)。
    背景は BG_COLOR と統一"""
    key = str(panel_path)
    if key in _panel_cache:
        return _panel_cache[key]
    src = load_image(panel_path)
    pw, ph = src.size
    box_w, box_h = W, PANEL_Y2 - PANEL_Y1
    scale = min(box_w / pw, box_h / ph)
    nw, nh = int(pw * scale), int(ph * scale)
    resized = src.resize((nw, nh), Image.LANCZOS)
    box = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 255))
    box.paste(resized, ((box_w - nw) // 2, (box_h - nh) // 2), resized)
    _panel_cache[key] = box
    return box


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    """日本語簡易 wrap: 1 文字ずつ蓄積して max_w 超えたら改行"""
    lines: list[str] = []
    buf = ""
    dummy = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    for ch in text:
        test = buf + ch
        bbox = dummy.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] > max_w and buf:
            lines.append(buf)
            buf = ch
        else:
            buf = test
    if buf:
        lines.append(buf)
    return lines


# ─────────────── Subtitle overlay (line ごと) ───────────────


def render_subtitle_band(text: str) -> Image.Image:
    band = Image.new("RGBA", (W, SUBT_Y2 - SUBT_Y1), (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    # 半透明の帯
    bg_h = SUBT_Y2 - SUBT_Y1
    d.rectangle((0, 0, W, bg_h), fill=(20, 20, 20, 200))
    if not FONT_PATH or not Path(FONT_PATH).exists():
        raise RuntimeError("字幕フォントが見つかりません。環境変数 SUBTITLE_FONT_PATH に .ttf/.ttc/.otf のパスを設定してください")
    font = ImageFont.truetype(FONT_PATH, SUBT_FONT_SIZE)
    lines = wrap_text(text, font, W - 80)
    if len(lines) > 2:
        # 2 行までに圧縮 (簡易、 改行を再計算)
        lines = lines[:2]
    total_h = sum(font.getbbox(L)[3] - font.getbbox(L)[1] + 8 for L in lines)
    y = (bg_h - total_h) // 2
    for L in lines:
        bbox = d.textbbox((0, 0), L, font=font)
        tw = bbox[2] - bbox[0]
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            d.text(((W - tw) // 2 + dx, y + dy), L, font=font, fill=(0, 0, 0, 255))
        d.text(((W - tw) // 2, y), L, font=font, fill=SUBT_COLOR)
        y += bbox[3] - bbox[1] + 8
    return band


# ─────────────── Frame composition ───────────────


def render_frame(t: float, scene: dict, timings: list[LineTiming],
                  refs_dir: Path) -> Image.Image:
    canvas = Image.new("RGBA", (W, H), BG_COLOR)  # 背景色で全面を塗る

    # 現 line 特定
    cur_idx = None
    n_lt = len(timings)
    scene_end = timings[-1].end_with_pause + SCENE_END_BUFFER if timings else 0.0
    for i, lt in enumerate(timings):
        end_limit = scene_end if i == n_lt - 1 else lt.end_with_pause
        if lt.start <= t < end_limit:
            cur_idx = i
            break
    if cur_idx is None:
        cur_idx = n_lt - 1
    apl = scene["assets_per_line"]

    # panel: 現 line の panel、 切替時 crossfade 0.25s
    cur_lt = timings[cur_idx]
    t_since = t - cur_lt.start
    crossfade = 0.25
    cur_panel = render_panel(refs_dir / apl[min(cur_idx, len(apl) - 1)])
    if cur_idx > 0 and t_since < crossfade:
        prev_panel = render_panel(refs_dir / apl[min(cur_idx - 1, len(apl) - 1)])
        p = t_since / crossfade
        canvas.paste(alpha_modulate(prev_panel, 1 - p), (0, PANEL_Y1))
        canvas.alpha_composite(alpha_modulate(cur_panel, p), (0, PANEL_Y1))
    else:
        canvas.paste(cur_panel, (0, PANEL_Y1))

    # 字幕 (現 line、 panel 直下に分離配置)
    sub_text = text_without_tags(cur_lt.text)
    sub_band = render_subtitle_band(sub_text)
    canvas.alpha_composite(sub_band, (0, SUBT_Y1))

    return canvas


def alpha_modulate(img: Image.Image, alpha: float) -> Image.Image:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * alpha))
    return Image.merge("RGBA", (r, g, b, a))


# ─────────────── Main ───────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-py", type=Path, required=True)
    ap.add_argument("--refs-dir", type=Path, required=True, help="panel PNG dir")
    ap.add_argument("--audio-dir", type=Path, required=True, help="work/audio_scenes/")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bgm", type=Path)
    ap.add_argument("--bgm-volume", type=float, default=0.10)
    args = ap.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.scenes_py.parent))
    mod = __import__(args.scenes_py.stem)
    shorts = mod.SHORTS
    title = getattr(mod, "TITLE", "Shorts")

    if len(shorts) != 1:
        print("[error] Shorts は 1 scene 構成のみサポート")
        return 2
    scene = shorts[0]
    sid = scene["id"]
    timings = compute_line_timings(args.audio_dir, sid, scene["lines"])
    total_dur = timings[-1].end_with_pause + SCENE_END_BUFFER

    print(f"[info] title: {title}")
    print(f"[info] {len(timings)} lines, dur={total_dur:.2f}s")

    # frames render
    frames_dir = args.work_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    n_frames = int(total_dur * FPS)
    print(f"[info] rendering {n_frames} frames")
    for fi in range(n_frames):
        t = fi / FPS
        if fi % 60 == 0:
            print(f"  frame {fi}/{n_frames} (t={t:.2f}s)")
        frame = render_frame(t, scene, timings, args.refs_dir)
        frame.convert("RGB").save(frames_dir / f"frame_{fi:05d}.png")

    # silent video
    silent_mp4 = args.work_dir / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20",
        str(silent_mp4),
    ], check=True)
    print(f"\nsilent video: {silent_mp4} ({silent_mp4.stat().st_size // (1024 * 1024)} MB)")

    # audio concat (line + pause)
    concat_list = args.work_dir / "audio_concat.txt"
    silence_pool: dict[float, Path] = {}
    parts: list[str] = []
    for lt in timings:
        parts.append(f"file '{lt.mp3.resolve()}'")
        # 末尾 line は pause なし
        pause = float(scene["lines"][lt.idx].get("pause_after", 0.3))
        if pause > 0:
            if pause not in silence_pool:
                sil = args.work_dir / f"_silence_{pause:.2f}.mp3"
                subprocess.run([
                    "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"anullsrc=r=44100:cl=mono:d={pause}",
                    "-c:a", "libmp3lame", "-b:a", "192k", str(sil),
                ], check=True)
                silence_pool[pause] = sil
            parts.append(f"file '{silence_pool[pause].resolve()}'")
    # 末尾 buffer
    buf_sil = args.work_dir / f"_silence_buf.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=mono:d={SCENE_END_BUFFER}",
        "-c:a", "libmp3lame", "-b:a", "192k", str(buf_sil),
    ], check=True)
    parts.append(f"file '{buf_sil.resolve()}'")
    concat_list.write_text("\n".join(parts), encoding="utf-8")

    audio_mp3 = args.work_dir / "audio.mp3"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", str(concat_list), "-c", "copy", str(audio_mp3),
    ], check=True)

    # 2-pass loudnorm で voice トラックを I=-16 LUFS に正規化
    # (漫画系統合)
    print("\n=== Voice loudnorm (2-pass, I=-16 LUFS) ===")
    audio_norm = pre_bake_loudnorm(audio_mp3)

    # mux (BGM 任意) + 最終 alimiter で真のピーク制御 (-0.45 dBFS)
    if args.bgm and args.bgm.exists():
        print(f"\n=== Mixing BGM ({args.bgm.name}, vol={args.bgm_volume}) + alimiter ===")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_mp4), "-i", str(audio_norm), "-stream_loop", "-1", "-i", str(args.bgm),
            "-filter_complex",
            f"[2:a]volume={args.bgm_volume}[bgm];"
            f"[1:a][bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,{ALIMITER_SUFFIX}[a]",
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "[a]",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(args.out),
        ], check=True)
    else:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(silent_mp4), "-i", str(audio_norm),
            "-filter_complex", f"[1:a]{ALIMITER_SUFFIX}[a]",
            "-c:v", "copy",
            "-map", "0:v:0", "-map", "[a]",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(args.out),
        ], check=True)

    # cleanup frames
    for f in frames_dir.glob("frame_*.png"):
        f.unlink()
    print(f"\n-> {args.out} ({args.out.stat().st_size // (1024 * 1024)} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

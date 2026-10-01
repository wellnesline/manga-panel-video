"""audio_timing — 各シーンの line ごとの絶対 timing(scene 内)を計測

scene_synth が出力した per-line mp3 から duration を測り、
pause_after を加味して line ごとの (start, end) を計算する。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


def get_duration(mp3: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(mp3)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


@dataclass
class LineTiming:
    idx: int
    speaker: str
    text: str
    audio_path: Path
    audio_dur: float
    pause_after: float
    start: float          # scene内 開始秒
    end: float            # scene内 発話終了秒
    end_with_pause: float  # scene内 pause含む終了秒


def compute_line_timings(scene_dir: Path, lines_def: list[dict]) -> list[LineTiming]:
    """
    scene_dir: scenes/<SCENE_ID>/
    lines_def: シーン定義の lines list(各要素 speaker / text / pause_after)
    既存の run<NN>_<speaker>_(single|line<K>).mp3 を順次マッピングして timing を計算
    """
    # シーン内 line index → audio file path を解決
    # run の構造を辿る:
    # runs: consecutive same speaker → block or single
    audio_files: list[Path] = []

    # speaker runs を構築
    runs: list[tuple[str, list[int]]] = []
    for i, ld in enumerate(lines_def):
        sp = ld["speaker"]
        if runs and runs[-1][0] == sp:
            runs[-1][1].append(i)
        else:
            runs.append((sp, [i]))

    line_to_file: dict[int, Path] = {}
    for run_idx, (sp, indices) in enumerate(runs):
        if len(indices) == 1:
            f = scene_dir / f"run{run_idx:02d}_{sp}_single.mp3"
            line_to_file[indices[0]] = f
        else:
            for k, i in enumerate(indices):
                f = scene_dir / f"run{run_idx:02d}_{sp}_line{k}.mp3"
                line_to_file[i] = f

    results: list[LineTiming] = []
    t = 0.0
    for i, ld in enumerate(lines_def):
        f = line_to_file[i]
        if not f.exists():
            raise FileNotFoundError(f"missing audio: {f}")
        dur = get_duration(f)
        pause = ld.get("pause_after", 0.3)
        lt = LineTiming(
            idx=i, speaker=ld["speaker"], text=ld["text"], audio_path=f,
            audio_dur=dur, pause_after=pause,
            start=t, end=t + dur, end_with_pause=t + dur + pause,
        )
        results.append(lt)
        t += dur + pause
    return results


def scene_total_duration(timings: list[LineTiming]) -> float:
    """scene 終端時刻(最後の line の end_with_pause、または最後の pause 除外)"""
    if not timings:
        return 0.0
    # 最後の line は pause_after を無視
    last = timings[-1]
    return last.end_with_pause if last.pause_after > 0 else last.end

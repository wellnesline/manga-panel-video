# manga-panel-video

Video builders that turn manga-style panel pages plus per-line narration audio into videos (landscape and vertical Shorts), revealing panels in sync with the audio.

マンガ形式のページ画像と、セリフ単位の音声から、パネルをセリフに合わせて順に見せる動画（横型・縦型 Shorts）を組み立てる Python スクリプト群です。公開用に整理したサンプル実装です。

## 概要

マンガ系の動画では、絵の切り替わりをセリフのタイミングに合わせることが品質を左右します。ここでは「音声が先」の設計にして、先に各セリフの実測の長さを取り、その時刻に合わせて映像をフレーム単位で描きます。映像が音声に合わせる形なので、字幕とのずれが構造的に起きにくくなります。

## 主な機能・設計のポイント

- **音声ファースト（`audio_timing.py` / `compose.py`）**: 各セリフの音声ファイルの長さと、セリフ後の間（`pause_after`）から、シーン内の開始・終了時刻を計算。パネルの出現は「どのセリフで出すか」（trigger_line）で決め、字幕の切り替えも同じ時刻表で行います
- **多角形パネル**: ページ画像のコマは、`<画像名>_panels.json` に頂点列（polygon）があればそれで切り抜き、無ければ矩形検出にフォールバックします。斜めや台形のコマ割りにも対応
- **ホワイトボード場面**: 図解スライドは、冒頭を図の部品が順に出るアニメ（`diagram-blob-animator` のレンダラ）にし、その後は静止させます（`slide_animator.py`）
- **縦型 Shorts**: `compose_shorts.py` はページを縦型 1080×1920 に収め、`compose_shorts_manga.py` は UI に隠れる上下を避けた配置（パネル・字幕・黒帯の領域を分離）でビルドします
- **公開前ゲート（`compose.py`）**: ビルド前に、(1) `fact_check.yaml` が `verdict: PASS` かつ `checked_at` と `evidence_urls` を持つこと、(2) 全ホワイトボード場面の `order.json` が存在すること、を検査し、満たさなければ終了コード 2 で止めます。緊急回避は `--bypass-fact-check` / 環境変数 `FACT_CHECK_BYPASS`、`ORDER_JSON_BYPASS` として明示的に残しており、通常運用では使わない前提です
- **音量**: BGM は `--bgm-volume`（既定 0.10）で混ぜ、ラウドネス調整は `audio_master.py` に切り出しています（公開版はスタブ）

## ディレクトリ構成（主要ファイル）

```
manga-panel-video/
├ compose.py               横型の本編ビルダー（ゲート・パネル・字幕・BGM）
├ compose_shorts.py        縦型 Shorts（ページ画像ベース）
├ compose_shorts_manga.py  縦型 Shorts（マンガパネル用レイアウト）
├ audio_timing.py          セリフごとの絶対時刻の計測
├ slide_animator.py        図解スライドのアニメ化ラッパ
└ audio_master.py          ← 合成スタブ
```

## 動かし方

- 必要なもの: Python 3、ffmpeg / ffprobe、下記「同梱していない依存」の Python パッケージ
- 環境変数: `WORKSPACE_ROOT`（既定はこのファイルの 1 つ上のディレクトリ）、`SUBTITLE_FONT_PATH`（字幕に使うフォントファイル。既定値は無く、未設定だとエラーになります）
- 入力は、シーン定義（`scenes.py` を `--scenes-py` で指定）、ページ画像ディレクトリ、セリフ別の音声ディレクトリ、キャラクター素材、`*_panels.json`、blob 分解済みのスライド、ホワイトボードのテンプレート画像です。引数の全体は `python compose.py --help` を参照してください
- コードは、ホストプロジェクトの `src/` 配下に置かれた `diagram_blob` を `src.diagram_blob` として import する前提です

## 制限事項（閉包）

`ast` と import 文の調査で、リポジトリ内・標準ライブラリに無く、`requirements.txt` も無いため、次を同梱していない依存として挙げます。

- **同梱していない Python パッケージ（3 件）**: `PIL`（Pillow）, `numpy`, `yaml`（PyYAML）
- **他の公開リポジトリへの依存**: `diagram-blob-animator`（`src.diagram_blob.blob_renderer` / `blob_animate`）
- **パッケージ配置の前提**: `src`（上記のとおり、ホストの `src/` 配置が必要）
- **合成スタブ**: `audio_master.py`（`pre_bake_loudnorm` は入力をそのまま返すだけ）
- **非同梱の実行時データ**: `scenes.py`（実行時に動的 import）、ページ画像、セリフ音声、キャラクター素材、パネル定義 JSON、`fact_check.yaml`、`order.json`、BGM
- 外部サービス・素材: 音声合成エンジンや画像生成サービスで作った音声・画像を使う場合は、各サービスの利用規約と、キャラクター・素材の権利者の許諾の範囲で使ってください

## ライセンス

All rights reserved. 閲覧目的で公開しています。詳細は [LICENSE](LICENSE) を参照。

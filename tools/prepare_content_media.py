"""Local media preparation (no API) for the mature development videos.

Runs the SAME preparation code as the content-extractor benchmark. The frame
plan, transcription config and the five preparation functions (frame_times,
sample_frames, decode_audio_16k, audio_stats_and_transcript, load_transcripts)
are loaded straight from the notebook source, so nothing can drift.

What it does
------------
1. Checks that every target video is verified in the media manifest and on disk.
2. Frame check (in memory, nothing saved, same as the notebook's dry run):
   samples each video's frames and records how many decoded.
3. Transcription: for every video without a cached transcript under the
   frozen config, decodes the audio locally (PyAV), measures loudness and
   transcribes it with faster-whisper. Each record is appended to
   data/processed/content_transcripts_v1.jsonl as soon as it is done, so the
   run is resumable; cached videos are skipped. Existing records are never
   rewritten (the script checks that the old file content is an exact prefix
   of the new one).
4. Writes a per-video preparation report (no captions or outcomes) to
   data/processed/content_media_prep_v1.csv.

No Anthropic API calls, no extraction, no model fitting.

Usage (from the project root):
    .venv\\Scripts\\python.exe tools\\prepare_content_media.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "creator_intelligence_data_prep.ipynb"
MODELING_CSV = ROOT / "data/processed/shop_videos_modeling.csv"
BENCHMARK_IDS_CSV = ROOT / "data/processed/content_eval_v1_sample_ids.csv"
MEDIA_MANIFEST = ROOT / "data/processed/media_manifest_dev_v1.csv"
EXTRACTION_CACHE = ROOT / "data/processed/content_extractions_content_v1_p1.jsonl"
REPORT_PATH = ROOT / "data/processed/content_media_prep_v1.csv"
VIDEO_DIR = ROOT / "data/raw/videos"

WANTED_ASSIGNMENTS = {"FRAME_PLAN", "TRANSCRIBE_CONFIG", "CONTENT_TRANSCRIPT_PATH"}
WANTED_FUNCTIONS = {"frame_times", "sample_frames", "decode_audio_16k",
                    "audio_stats_and_transcript", "load_transcripts"}


def load_notebook_pipeline() -> dict:
    """Execute only the wanted top-level assignments / defs from the notebook's code cells."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"np": np, "Path": Path, "json": json}
    found = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if not any(name in src for name in WANTED_ASSIGNMENTS | WANTED_FUNCTIONS):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            name = None
            if isinstance(node, ast.FunctionDef):
                name = node.name
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name in WANTED_ASSIGNMENTS | WANTED_FUNCTIONS:
                if name in found:
                    raise RuntimeError(f"{name} is defined more than once in the notebook")
                exec(compile(ast.Module([node], []), f"<notebook:{name}>", "exec"), ns)
                found.add(name)
    missing = (WANTED_ASSIGNMENTS | WANTED_FUNCTIONS) - found
    if missing:
        raise RuntimeError(f"not found in the notebook: {sorted(missing)}")
    return ns


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> None:
    os.chdir(ROOT)  # the notebook's paths are relative to the project root
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except AttributeError:
        pass
    nbp = load_notebook_pipeline()
    transcript_path = ROOT / nbp["CONTENT_TRANSCRIPT_PATH"]
    print("FRAME_PLAN:", nbp["FRAME_PLAN"])
    print("TRANSCRIBE_CONFIG:", nbp["TRANSCRIBE_CONFIG"])

    # ---- target set: 60 benchmark + 312 manifest videos = the 372 mature development videos
    model = pd.read_csv(MODELING_CSV, usecols=["id", "duration", "mature_split"], dtype={"id": str})
    dev = model[model["mature_split"] == "train"].set_index("id")
    bench = list(pd.read_csv(BENCHMARK_IDS_CSV, dtype=str)["id"])
    manifest = pd.read_csv(MEDIA_MANIFEST, dtype=str, keep_default_na=False)
    assert (manifest["acquisition_status"] == "verified").all() and len(manifest) == 312, \
        "media manifest is not 312/312 verified"
    new_ids = list(manifest.sort_values("seq", key=lambda s: s.astype(int))["id"])
    all_ids = bench + new_ids
    assert set(all_ids) == set(dev.index) and len(all_ids) == 372, "target set != mature development set"
    missing_files = [i for i in all_ids if not (VIDEO_DIR / f"{i}.mp4").exists()]
    assert not missing_files, f"{len(missing_files)} MP4s missing"

    # ---- the frozen config must be the one the 60 benchmark transcripts were made with
    cached = nbp["load_transcripts"]()  # only records whose config == TRANSCRIBE_CONFIG
    assert set(bench) <= set(cached), "benchmark transcripts don't match the current TRANSCRIBE_CONFIG"
    stray = set(cached) - set(all_ids)
    assert not stray, f"transcript cache holds non-development ids: {len(stray)}"

    before_bytes = transcript_path.read_bytes()
    before_cache_sha = sha256_file(EXTRACTION_CACHE)

    # ---- 1. frame check (in memory, nothing written)
    print(f"\nFrame check on {len(all_ids)} videos (in memory only)...")
    rows = []
    for k, vid in enumerate(all_ids, 1):
        dur = int(dev.at[vid, "duration"])
        row = {"id": vid, "group": "benchmark" if vid in set(bench) else "new",
               "api_duration": dur, "frames_planned": len(nbp["frame_times"](dur))}
        try:
            frames = nbp["sample_frames"](VIDEO_DIR / f"{vid}.mp4", dur)
            row.update(frames=len(frames),
                       frame_px=f"{frames[0][2][0]}x{frames[0][2][1]}" if frames else "",
                       jpeg_kb=round(sum(len(b) for _, b, _ in frames) / 1024),
                       frame_error="")
        except Exception as e:
            row.update(frames=0, frame_px="", jpeg_kb=0, frame_error=f"{type(e).__name__}: {e}")
        rows.append(row)
        if k % 50 == 0:
            print(f"  {k}/{len(all_ids)}")
    report = pd.DataFrame(rows).set_index("id")

    # ---- 2. transcription (resumable, append-only)
    todo = [i for i in all_ids if i not in cached]
    print(f"\nTranscription: {len(todo)} to do, {len(all_ids) - len(todo)} already cached")
    failures = {}
    if todo:
        from faster_whisper import WhisperModel

        cfg = nbp["TRANSCRIBE_CONFIG"]
        whisper = WhisperModel(cfg["model"], device=cfg["device"], compute_type=cfg["compute_type"])
        t_start = time.time()
        for k, vid in enumerate(todo, 1):
            t0 = time.time()
            try:
                rec = {"id": vid, "config": cfg,
                       **nbp["audio_stats_and_transcript"](VIDEO_DIR / f"{vid}.mp4", whisper)}
            except Exception as e:
                failures[vid] = f"{type(e).__name__}: {e}"
                print(f"  [{k}/{len(todo)}] {vid} FAILED: {failures[vid]}")
                continue
            with open(transcript_path, "a", encoding="utf-8") as f:   # same format as the notebook
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            elapsed = time.time() - t_start
            print(f"  [{k}/{len(todo)}] {vid} | {len(rec['segments'])} segments | {time.time() - t0:.1f} s "
                  f"| ETA {elapsed / k * (len(todo) - k) / 60:.0f} min")

    # ---- integrity: old transcript records untouched, extraction cache untouched
    after_bytes = transcript_path.read_bytes()
    assert after_bytes.startswith(before_bytes), "existing transcript records changed!"
    assert sha256_file(EXTRACTION_CACHE) == before_cache_sha, "extraction cache changed!"

    # ---- 3. report
    cached = nbp["load_transcripts"]()
    for vid in all_ids:
        t = cached.get(vid)
        if t is None:
            report.loc[vid, "transcript_status"] = "failed" if vid in failures else "missing"
            report.loc[vid, "transcript_error"] = failures.get(vid, "")
            continue
        words = sum(len(s["text"].split()) for s in t["segments"])
        report.loc[vid, "transcript_status"] = (
            "no_audio" if not t["has_audio"] else "speech" if t["segments"] else "no_speech")
        report.loc[vid, "has_audio"] = t["has_audio"]
        report.loc[vid, "rms_dbfs"] = t["rms_dbfs"]
        report.loc[vid, "non_silent_share"] = t["non_silent_share"]
        report.loc[vid, "language"] = t["language"]
        report.loc[vid, "segments"] = len(t["segments"])
        report.loc[vid, "transcript_words"] = words
        report.loc[vid, "transcript_error"] = ""
    report.reset_index().to_csv(REPORT_PATH, index=False)
    print(f"\nReport written: {REPORT_PATH.relative_to(ROOT)}")
    print("Transcript cache: old records are an exact prefix of the new file; extraction cache unchanged.")
    for g, d in report.groupby("group"):
        print(f"[{g}] n={len(d)} | frames ok={int((d['frames'] > 0).sum())} "
              f"| fewer than planned={int((d['frames'] < d['frames_planned']).sum())} "
              f"| transcript status={d['transcript_status'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

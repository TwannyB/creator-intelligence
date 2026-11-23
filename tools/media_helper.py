"""Assisted manual-download helper for the development-set MP4s.

What it does
------------
This helper never downloads anything from TikTok. It walks through the frozen
list of 312 mature development videos that still need an MP4. For each one it
opens the TikTok share URL in your browser. You right-click the video and pick
"Download video". The helper then:

1. waits for exactly one new, completed file in the watch folder;
2. checks that the file is a readable video (first frame decodes, the duration
   matches the Display API duration, there is an audio track, it isn't a copy
   of a file it already has);
3. if every check passes, moves and renames it to ``data/raw/videos/<id>.mp4``;
4. otherwise moves it to ``data/raw/videos_quarantine/`` and asks you what to do.

Progress lives in ``data/processed/media_manifest_dev_v1.csv``. It is saved
atomically after every change, so you can stop at any point (Ctrl+C or closing
the window) and resume later. Every event is also appended to
``data/processed/media_acquisition_log.jsonl``. All three locations are
gitignored.

The manifest holds only identifiers and pre-publication metadata (post time,
duration, caption, share URL) plus file-check results. The helper reads only
those columns from the modeling CSV, never any outcome or engagement column.

Usage (from the project root)
-----------------------------
    .venv\\Scripts\\python.exe tools\\media_helper.py --status   # read-only summary
    .venv\\Scripts\\python.exe tools\\media_helper.py --init     # create the manifest only
    .venv\\Scripts\\python.exe tools\\media_helper.py            # start / resume downloading
    .venv\\Scripts\\python.exe tools\\media_helper.py --review   # revisit ambiguous / failed items
    .venv\\Scripts\\python.exe tools\\media_helper.py --only <id>   # redo one item

Options: --watch-dir PATH (default: %USERPROFILE%\\TikTokDownloads),
--no-open (don't open the browser), --no-clipboard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# sha256 of the sorted, newline-joined IDs of the 312 target videos
# (mature development split minus the frozen 60-video content benchmark).
EXPECTED_FINGERPRINT = "666539162e2fd23fbb59688570c40d51be6c5d1e8144caecfe4275d5a6753525"
EXPECTED_COUNT = 312

DURATION_TOLERANCE_S = 1.5  # benchmark files: file duration - API duration is 0.0-0.9 s
MIN_FILE_BYTES = 50_000
TEMP_SUFFIXES = {".crdownload", ".part", ".partial", ".tmp", ".download", ".opdownload"}

STATUSES = ("missing", "downloaded", "verified", "ambiguous", "failed")
MANIFEST_COLUMNS = [
    "seq", "id", "post_datetime", "duration", "caption", "share_url",
    "expected_local_path", "acquisition_status", "status_reason", "user_override",
    "source_filename", "quarantine_path", "file_size", "sha256", "file_duration",
    "duration_diff", "width", "height", "video_codec", "has_audio", "audio_codec",
    "updated_at",
]
# Safety net: no outcome / engagement information may ever enter the manifest.
FORBIDDEN_COLUMN_PATTERN = re.compile(
    r"order|gmv|sale|tier|target|commission|view|like|comment|predict|revenue|earn|refund",
    re.IGNORECASE,
)
# The only columns ever read from the modeling CSV.
MODELING_COLUMNS = ["id", "post_datetime", "duration", "video_description", "share_url", "mature_split"]


@dataclass
class Config:
    modeling_csv: Path = ROOT / "data/processed/shop_videos_modeling.csv"
    benchmark_ids_csv: Path = ROOT / "data/processed/content_eval_v1_sample_ids.csv"
    videos_dir: Path = ROOT / "data/raw/videos"
    quarantine_dir: Path = ROOT / "data/raw/videos_quarantine"
    manifest_path: Path = ROOT / "data/processed/media_manifest_dev_v1.csv"
    log_path: Path = ROOT / "data/processed/media_acquisition_log.jsonl"
    lock_path: Path = ROOT / "data/processed/media_helper.lock"
    watch_dir: Path = Path.home() / "TikTokDownloads"
    open_browser: bool = True
    copy_clipboard: bool = True
    poll_seconds: float = 1.0
    stable_checks: int = 2  # unchanged size on this many consecutive polls = finished


# --------------------------------------------------------------------------- #
# Target set
# --------------------------------------------------------------------------- #
def id_fingerprint(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def load_target_frame(cfg: Config):
    """Return (target_df, protected_ids, all_known_ids).

    target = mature development videos minus the frozen benchmark, oldest first.
    protected = every other known Shop video (holdout, immature, benchmark).
    """
    df = pd.read_csv(cfg.modeling_csv, usecols=MODELING_COLUMNS, dtype={"id": str})
    benchmark = set(pd.read_csv(cfg.benchmark_ids_csv, dtype=str)["id"])
    dev = df[df["mature_split"] == "train"]
    target = dev[~dev["id"].isin(benchmark)].copy()
    target["_dt"] = pd.to_datetime(target["post_datetime"])
    target = target.sort_values(["_dt", "id"]).drop(columns=["_dt", "mature_split"])
    target = target.rename(columns={"video_description": "caption"}).reset_index(drop=True)

    if len(target) != EXPECTED_COUNT or id_fingerprint(target["id"]) != EXPECTED_FINGERPRINT:
        raise SystemExit(
            f"Target set changed: {len(target)} videos, fingerprint "
            f"{id_fingerprint(target['id'])[:16]}... (expected {EXPECTED_COUNT}, "
            f"{EXPECTED_FINGERPRINT[:16]}...). Stopping; nothing was modified."
        )
    protected = set(df["id"]) - set(target["id"])
    return target, protected, set(df["id"])


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_video(path: Path) -> dict:
    """Open the file with PyAV, decode the first frame, return stream metadata."""
    import av  # local import: only needed when a file is checked

    with av.open(str(path)) as container:
        videos = [s for s in container.streams if s.type == "video"]
        audios = [s for s in container.streams if s.type == "audio"]
        if not videos:
            raise ValueError("no video stream")
        v = videos[0]
        duration = None
        if container.duration:
            duration = container.duration / 1_000_000
        elif v.duration and v.time_base:
            duration = float(v.duration * v.time_base)
        if next(container.decode(video=0), None) is None:
            raise ValueError("first frame did not decode")
        return {
            "file_duration": duration,
            "width": v.codec_context.width,
            "height": v.codec_context.height,
            "video_codec": v.codec_context.name,
            "has_audio": bool(audios),
            "audio_codec": audios[0].codec_context.name if audios else "",
        }


def is_locked(path: Path) -> bool:
    """On Windows a file still being written/scanned can't be opened for update."""
    try:
        with open(path, "rb+"):
            return False
    except OSError:
        return True


def move_file(src: Path, dst: Path) -> None:
    """Move without ever overwriting; retries while Windows holds a lock."""
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_error = None
    for _ in range(20):
        try:
            if src.resolve().drive.lower() == dst.resolve().drive.lower():
                os.rename(src, dst)  # same volume: atomic, and fails if dst exists
            else:
                shutil.copy2(src, dst)
                if dst.stat().st_size != src.stat().st_size:
                    dst.unlink()
                    raise OSError("copy size mismatch")
                src.unlink()
            return
        except PermissionError as e:
            last_error = e
            time.sleep(0.5)
    raise last_error


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# --------------------------------------------------------------------------- #
# Console interaction (replaced by a scripted UI in tests)
# --------------------------------------------------------------------------- #
class ConsoleUI:
    def show(self, msg: str = "") -> None:
        print(msg, flush=True)

    def ask(self, prompt: str, choices: dict) -> str:
        options = " / ".join(f"[{k}] {v}" for k, v in choices.items())
        while True:
            answer = input(f"{prompt}\n  {options}\n  > ").strip().lower()
            if answer in choices:
                return answer
            print("  Please type one of:", ", ".join(choices))

    def poll_key(self):
        """Non-blocking single key press (Windows); None if nothing pressed."""
        try:
            import msvcrt
        except ImportError:
            return None
        if msvcrt.kbhit():
            return msvcrt.getwch().lower()
        return None

    def armed(self, item_id: str) -> None:
        """Called once the helper is waiting for this item's download."""


class InstanceLock:
    """Prevents two helpers from editing the manifest at the same time.

    Uses an OS-level lock, which Windows releases automatically when the
    process ends (even if the terminal is closed), so it never goes stale.
    """

    def __init__(self, path: Path):
        self.path = path
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            pass
        except OSError:
            self.handle.close()
            raise SystemExit("Another media_helper is already running. Close it first.")
        return self

    def __exit__(self, *exc):
        try:
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        except (ImportError, OSError):
            pass
        self.handle.close()


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #
class Helper:
    def __init__(self, cfg: Config, target_df: pd.DataFrame, protected_ids, all_known_ids, ui=None):
        self.cfg = cfg
        self.ui = ui or ConsoleUI()
        self.target_df = target_df.reset_index(drop=True)
        self.target_ids = set(self.target_df["id"])
        self.protected_ids = set(protected_ids)
        self.all_known_ids = set(all_known_ids) | self.target_ids
        assert not (self.target_ids & self.protected_ids), "target and protected sets overlap"
        self.neighbor_of = self._same_duration_neighbors()
        self.m: pd.DataFrame | None = None
        self.known_hashes: dict[str, str] = {}  # sha256 -> video id

    # ---- manifest -------------------------------------------------------- #
    def _same_duration_neighbors(self) -> dict:
        """Adjacent items with the same API duration: a swap would pass the duration check."""
        out = {}
        d = self.target_df["duration"].tolist()
        ids = self.target_df["id"].tolist()
        for i in range(1, len(ids)):
            if d[i] == d[i - 1]:
                out.setdefault(ids[i], []).append(ids[i - 1])
                out.setdefault(ids[i - 1], []).append(ids[i])
        return out

    def build_manifest(self) -> pd.DataFrame:
        m = pd.DataFrame({
            "seq": range(1, len(self.target_df) + 1),
            "id": self.target_df["id"],
            "post_datetime": self.target_df["post_datetime"],
            "duration": self.target_df["duration"],
            "caption": self.target_df["caption"].fillna(""),
            "share_url": self.target_df["share_url"],
        })
        m["expected_local_path"] = [
            (self.cfg.videos_dir / f"{i}.mp4").relative_to(ROOT).as_posix()
            if self.cfg.videos_dir.is_relative_to(ROOT) else str(self.cfg.videos_dir / f"{i}.mp4")
            for i in m["id"]
        ]
        for col in MANIFEST_COLUMNS:
            if col not in m:
                m[col] = ""
        m["acquisition_status"] = "missing"
        m = m[MANIFEST_COLUMNS].astype(str)
        self._check_manifest(m)
        return m

    def _check_manifest(self, m: pd.DataFrame) -> None:
        bad = [c for c in m.columns if FORBIDDEN_COLUMN_PATTERN.search(c)]
        assert not bad, f"forbidden columns in manifest: {bad}"
        assert list(m.columns) == MANIFEST_COLUMNS, "manifest columns changed"
        assert set(m["id"]) == self.target_ids and len(m) == len(self.target_ids), \
            "manifest IDs differ from the frozen target set"
        assert set(m["acquisition_status"]) <= set(STATUSES), "unknown status in manifest"

    def load_manifest(self, create: bool) -> bool:
        """Load the manifest (or build it in memory). Returns True if it exists on disk."""
        if self.cfg.manifest_path.exists():
            m = pd.read_csv(self.cfg.manifest_path, dtype=str, keep_default_na=False)
            self._check_manifest(m)
            exists = True
        else:
            m = self.build_manifest()
            exists = False
        self.m = m.set_index("id", drop=False)
        if create and not exists:
            self.save()
            self.log("manifest_created", None, count=len(m), fingerprint=EXPECTED_FINGERPRINT[:16])
            exists = True
        return exists

    def save(self) -> None:
        """Atomic save: write a temp file, then swap it in."""
        path = self.cfg.manifest_path
        tmp = path.with_name(path.name + ".tmp")
        self.m.to_csv(tmp, index=False, encoding="utf-8")
        for attempt in range(30):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == 0:
                    self.ui.show("  (manifest is locked - is it open in Excel? retrying...)")
                time.sleep(1)
        raise SystemExit(f"Could not save {path}; close any program that has it open. "
                         f"Your progress is in {tmp}.")

    def log(self, event: str, item_id, **details) -> None:
        rec = {"time": timestamp(), "event": event, "id": item_id, **details}
        with open(self.cfg.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def set(self, item_id: str, **values) -> None:
        for k, v in values.items():
            assert k in MANIFEST_COLUMNS, k
            if v is None:
                v = ""
            elif isinstance(v, float):
                v = f"{v:.3f}"
            self.m.at[item_id, k] = str(v)
        self.m.at[item_id, "updated_at"] = timestamp()

    def status(self, item_id: str) -> str:
        return self.m.at[item_id, "acquisition_status"]

    def target_path(self, item_id: str) -> Path:
        if item_id not in self.target_ids or item_id in self.protected_ids:
            raise RuntimeError(f"refusing to write a file for non-target video {item_id}")
        return self.cfg.videos_dir / f"{item_id}.mp4"

    # ---- hashes ---------------------------------------------------------- #
    def build_known_hashes(self) -> None:
        """sha256 of every MP4 already in the videos folder (benchmark + verified)."""
        self.known_hashes = {}
        if not self.cfg.videos_dir.exists():
            return
        for p in sorted(self.cfg.videos_dir.glob("*.mp4")):
            vid = p.stem
            sha = None
            if self.m is not None and vid in self.m.index and self.status(vid) == "verified" \
                    and self.m.at[vid, "file_size"] == str(p.stat().st_size):
                sha = self.m.at[vid, "sha256"] or None
            self.known_hashes[sha or sha256_file(p)] = vid

    # ---- validation ------------------------------------------------------ #
    def validate(self, path: Path, item_id: str) -> tuple[str, list, dict]:
        """Return (status, reasons, metadata). status: verified / ambiguous / failed."""
        failed, ambiguous = [], []
        meta = {"file_size": path.stat().st_size, "sha256": sha256_file(path)}
        if path.suffix.lower() != ".mp4":
            failed.append(f"not an .mp4 file ({path.suffix or 'no extension'})")
        elif meta["file_size"] < MIN_FILE_BYTES:
            failed.append(f"file too small ({meta['file_size']} bytes)")
        else:
            try:
                meta.update(probe_video(path))
            except Exception as e:  # any decode problem means: not usable
                failed.append(f"not a readable video ({type(e).__name__}: {e})")

        if not failed:
            api = float(self.m.at[item_id, "duration"])
            if meta["file_duration"] is None:
                ambiguous.append("duration could not be read")
            else:
                diff = meta["file_duration"] - api
                meta["duration_diff"] = diff
                if abs(diff) > DURATION_TOLERANCE_S:
                    ambiguous.append(
                        f"duration {meta['file_duration']:.1f} s vs API {api:.0f} s "
                        f"(off by {diff:+.1f} s, tolerance {DURATION_TOLERANCE_S} s)")
            if not meta["has_audio"]:
                ambiguous.append("no audio track")

        other = self.known_hashes.get(meta["sha256"])
        if other and other != item_id:
            ambiguous.append(f"byte-identical to the file already stored for {other}")

        named = [n for n in re.findall(r"\d{18,20}", path.name) if n in self.all_known_ids]
        if named and item_id not in named:
            ambiguous.append(f"file name refers to a different video ({named[0]})")

        status = "failed" if failed else "ambiguous" if ambiguous else "verified"
        return status, failed + ambiguous, meta

    # ---- moving files ---------------------------------------------------- #
    def quarantine(self, src: Path, item_id: str, tag: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        dst = self.cfg.quarantine_dir / f"{item_id}__{tag}__{stamp}{src.suffix.lower()}"
        n = 1
        while dst.exists():
            dst = self.cfg.quarantine_dir / f"{item_id}__{tag}__{stamp}_{n}{src.suffix.lower()}"
            n += 1
        move_file(src, dst)
        return dst

    def place(self, item_id: str, src: Path) -> bool:
        """Move src to data/raw/videos/<id>.mp4. Never overwrites silently.

        Returns True if src was placed, False if the existing file was kept.
        """
        target = self.target_path(item_id)
        if target.exists():
            choice = self.ui.ask(
                f"  {target.name} already exists "
                f"(existing sha {sha256_file(target)[:12]}, new sha {sha256_file(src)[:12]}).",
                {"k": "keep the existing file (new one goes to quarantine)",
                 "r": "replace it (old file is moved to quarantine, not deleted)"})
            if choice == "k":
                q = self.quarantine(src, item_id, "not_used")
                self.log("kept_existing_file", item_id, new_file_quarantined=str(q))
                return False
            old = self.quarantine(target, item_id, "replaced")
            self.log("replaced_existing_file", item_id, old_file_quarantined=str(old))
        move_file(src, target)
        return True

    # ---- one candidate file --------------------------------------------- #
    def handle_candidate(self, item_id: str, path: Path, origin: str) -> str:
        """Validate a newly downloaded file. Returns 'next', 'retry' or 'quit'."""
        self.set(item_id, acquisition_status="downloaded", source_filename=path.name,
                 status_reason=f"checking ({origin})", quarantine_path="", user_override="")
        self.save()
        self.log("file_detected", item_id, source_filename=path.name, origin=origin)

        status, reasons, meta = self.validate(path, item_id)
        self.set(item_id, **{k: meta.get(k) for k in
                             ("file_size", "sha256", "file_duration", "duration_diff", "width",
                              "height", "video_codec", "has_audio", "audio_codec")})
        self._show_check(item_id, status, reasons, meta)

        if status == "verified" and item_id in self.neighbor_of:
            nb = self.neighbor_of[item_id]
            self.ui.show(f"  Note: the neighbouring item(s) {', '.join(nb)} have the same API "
                         f"duration, so the duration check can't tell them apart.")
            self.ui.show(f"  Caption: {self._short(self.m.at[item_id, 'caption'], 200)}")
            if self.ui.ask("  Is this the video you just downloaded from that TikTok?",
                           {"y": "yes, accept", "n": "no / not sure"}) == "n":
                status, reasons = "ambiguous", ["user could not confirm (same-duration neighbour)"]

        if status == "verified":
            placed = self.place(item_id, path)
            if not placed:  # user kept the existing file: re-check that one instead
                return self.adopt(item_id, announce=True)
            self.known_hashes[meta["sha256"]] = item_id
            self.set(item_id, acquisition_status="verified", status_reason="all checks passed")
            self.save()
            self.log("verified", item_id, sha256=meta["sha256"][:16],
                     file_duration=meta.get("file_duration"))
            return "next"

        q = self.quarantine(path, item_id, status)
        self.set(item_id, acquisition_status=status, status_reason="; ".join(reasons),
                 quarantine_path=str(q))
        self.save()
        self.log(status, item_id, reasons=reasons, quarantine_path=str(q))
        return self.decide(item_id)

    def decide(self, item_id: str) -> str:
        """Ask what to do with an ambiguous / failed item."""
        status = self.status(item_id)
        choices = {"r": "retry: download it again now"}
        if status == "ambiguous":
            choices["a"] = "accept the quarantined file anyway (recorded as a manual override)"
        choices["l"] = f"leave it as '{status}' for --review and continue"
        choices["q"] = "quit"
        choice = self.ui.ask(f"  -> {status.upper()}: the file is in quarantine. What now?", choices)
        if choice == "a":
            return self.accept_quarantined(item_id)
        if choice == "r":
            return "retry"
        if choice == "q":
            return "quit"
        return "next"

    def accept_quarantined(self, item_id: str) -> str:
        q = Path(self.m.at[item_id, "quarantine_path"])
        if not q.exists():
            self.ui.show(f"  Quarantined file {q} is gone; marking the item missing.")
            self.set(item_id, acquisition_status="missing", status_reason="quarantined file missing",
                     quarantine_path="")
            self.save()
            return "next"
        reason = self.m.at[item_id, "status_reason"]
        if not self.place(item_id, q):
            return self.adopt(item_id, announce=True)
        self.known_hashes[self.m.at[item_id, "sha256"]] = item_id
        self.set(item_id, acquisition_status="verified", user_override="1", quarantine_path="",
                 status_reason=f"accepted by user despite: {reason}")
        self.save()
        self.log("accepted_override", item_id, original_reason=reason)
        self.ui.show("  Accepted and moved into data/raw/videos (marked as a manual override).")
        return "next"

    def adopt(self, item_id: str, announce: bool = False) -> str:
        """Validate a file that is already at data/raw/videos/<id>.mp4."""
        target = self.target_path(item_id)
        status, reasons, meta = self.validate(target, item_id)
        self.set(item_id, source_filename="(already on disk)",
                 **{k: meta.get(k) for k in ("file_size", "sha256", "file_duration", "duration_diff",
                                             "width", "height", "video_codec", "has_audio",
                                             "audio_codec")})
        if status == "verified":
            self.known_hashes[meta["sha256"]] = item_id
            self.set(item_id, acquisition_status="verified", quarantine_path="",
                     status_reason="file already on disk; all checks passed")
            self.log("adopted_existing_file", item_id, sha256=meta["sha256"][:16])
        else:
            q = self.quarantine(target, item_id, f"ondisk_{status}")
            self.set(item_id, acquisition_status=status, status_reason="; ".join(reasons),
                     quarantine_path=str(q))
            self.log(f"existing_file_{status}", item_id, reasons=reasons, quarantine_path=str(q))
        self.save()
        if announce:
            self.ui.show(f"  Existing file re-checked: {status} {'; '.join(reasons)}")
        return "next"

    # ---- reconcile ------------------------------------------------------- #
    def reconcile(self, apply: bool) -> list:
        """Bring the manifest in line with the files on disk. Returns the changes."""
        changes = []
        for item_id in self.m["id"]:
            st = self.status(item_id)
            target = self.cfg.videos_dir / f"{item_id}.mp4"
            if target.exists():
                if st == "verified" and self.m.at[item_id, "file_size"] == str(target.stat().st_size):
                    continue
                why = "verified file changed on disk" if st == "verified" else f"file present but status '{st}'"
                changes.append((item_id, "re-check file on disk", why))
                if apply:
                    self.adopt(item_id)
            elif st == "verified":
                changes.append((item_id, "-> missing", "verified file no longer on disk"))
                if apply:
                    self.set(item_id, acquisition_status="missing",
                             status_reason="verified file disappeared from disk")
                    self.log("reconciled_missing", item_id, reason="file disappeared")
            elif st == "downloaded":
                changes.append((item_id, "-> missing", "interrupted before the file was placed"))
                if apply:
                    self.set(item_id, acquisition_status="missing",
                             status_reason="interrupted while checking; download again")
                    self.log("reconciled_missing", item_id, reason="interrupted")
        if apply and changes:
            self.save()
        return changes

    # ---- watch folder ---------------------------------------------------- #
    def watch_entries(self) -> list:
        return sorted(p for p in self.cfg.watch_dir.iterdir() if p.is_file())

    def ensure_watch_dir_empty(self) -> bool:
        """The watch folder must be empty before an item is armed. False = quit."""
        while True:
            entries = self.watch_entries()
            if not entries:
                return True
            self.ui.show(f"\n  The watch folder is not empty ({len(entries)} file(s)):")
            for p in entries[:10]:
                self.ui.show(f"    {p.name}")
            choice = self.ui.ask(
                "  Files there could be attached to the wrong video.",
                {"m": "move them to quarantine/_stray", "c": "I cleared it myself, check again",
                 "q": "quit"})
            if choice == "q":
                return False
            if choice == "m":
                stray = self.cfg.quarantine_dir / "_stray"
                for p in entries:
                    if p.suffix.lower() in TEMP_SUFFIXES and is_locked(p):
                        continue  # a download still in progress; can't move it yet
                    dst = stray / f"{datetime.now():%Y%m%dT%H%M%S}__{p.name}"
                    move_file(p, dst)
                    self.log("stray_file_quarantined", None, source_filename=p.name, to=str(dst))

    def wait_for_download(self, snapshot: set):
        """Wait for files not in `snapshot` to finish downloading.

        Returns ('files', [paths]) / ('skip',) / ('quit',).
        """
        sizes: dict[str, tuple[int, int]] = {}
        while True:
            key = self.ui.poll_key()
            if key == "s":
                return ("skip",)
            if key == "q":
                return ("quit",)
            entries = [p for p in self.watch_entries() if p.name not in snapshot]
            in_progress = [p for p in entries if p.suffix.lower() in TEMP_SUFFIXES]
            done = [p for p in entries if p not in in_progress]
            stable = []
            for p in done:
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                prev = sizes.get(p.name)
                count = prev[1] + 1 if prev and prev[0] == size and size > 0 else 0
                sizes[p.name] = (size, count)
                if count >= self.cfg.stable_checks and not is_locked(p):
                    stable.append(p)
            if done and not in_progress and len(stable) == len(done):
                return ("files", done)
            time.sleep(self.cfg.poll_seconds)

    # ---- one item -------------------------------------------------------- #
    @staticmethod
    def _short(text: str, n: int) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= n else text[: n - 1] + "…"

    def _show_check(self, item_id, status, reasons, meta) -> None:
        api = self.m.at[item_id, "duration"]
        dur = meta.get("file_duration")
        dur_txt = f"{dur:.1f} s vs API {api} s" if dur is not None else "duration n/a"
        size_txt = f"{meta['file_size'] / 1e6:.1f} MB"
        dims = f"{meta.get('width')}x{meta.get('height')} {meta.get('video_codec', '')}" if "width" in meta else ""
        audio = "audio ok" if meta.get("has_audio") else "NO audio"
        mark = {"verified": "OK", "ambiguous": "AMBIGUOUS", "failed": "FAILED"}[status]
        self.ui.show(f"  Check: {mark} | {dur_txt} | {dims} | {audio} | {size_txt}")
        for r in reasons:
            self.ui.show(f"    - {r}")

    def show_item(self, item_id: str) -> None:
        r = self.m.loc[item_id]
        done = int((self.m["acquisition_status"] == "verified").sum())
        self.ui.show("")
        self.ui.show(f"=== [{r['seq']}/{len(self.m)}]  ({done} verified so far) " + "=" * 30)
        self.ui.show(f"  id        {item_id}")
        self.ui.show(f"  posted    {str(r['post_datetime'])[:16]}")
        self.ui.show(f"  duration  {r['duration']} s (Display API)")
        self.ui.show(f"  caption   {self._short(r['caption'], 160)}")
        self.ui.show(f"  url       {r['share_url']}")

    def acquire(self, item_id: str) -> str:
        """Run the download loop for one item. Returns 'next' or 'quit'."""
        while True:
            if not self.ensure_watch_dir_empty():
                return "quit"
            # Snapshot BEFORE the browser opens, so even a very fast download counts as new.
            snapshot = {p.name for p in self.watch_entries()}
            self.show_item(item_id)
            url = self.m.at[item_id, "share_url"]
            if self.cfg.copy_clipboard and sys.platform == "win32":
                subprocess.run("clip", input=url.encode("ascii", "ignore"), check=False)
            if self.cfg.open_browser:
                webbrowser.open(url)
            self.ui.show(f'  -> Right-click the video and choose "Download video".')
            self.ui.show(f"     Waiting for one new .mp4 in {self.cfg.watch_dir}   "
                         f"(press s = skip/unavailable, q = quit)")
            self.log("armed", item_id)
            self.ui.armed(item_id)
            result = self.wait_for_download(snapshot)

            if result[0] == "quit":
                self.log("quit_while_waiting", item_id)
                return "quit"
            if result[0] == "skip":
                self.set(item_id, acquisition_status="failed",
                         status_reason="skipped by user (video unavailable?)")
                self.save()
                self.log("skipped", item_id)
                self.ui.show("  Skipped -> marked failed (revisit with --review).")
                return "next"

            files = result[1]
            if len(files) > 1:
                qs = [str(self.quarantine(p, item_id, "multiple")) for p in files]
                self.set(item_id, acquisition_status="ambiguous", quarantine_path=qs[0],
                         status_reason=f"{len(files)} new files appeared at once: "
                                       + ", ".join(p.name for p in files))
                self.save()
                self.log("ambiguous", item_id, reasons=["multiple new files"], quarantined=qs)
                self.ui.show(f"  {len(files)} new files appeared at once - can't tell which is "
                             f"which. All moved to quarantine.")
                outcome = self.decide(item_id)
            else:
                outcome = self.handle_candidate(item_id, files[0], origin="download")
            if outcome != "retry":
                return outcome

    # ---- modes ----------------------------------------------------------- #
    def counts(self) -> dict:
        vc = self.m["acquisition_status"].value_counts()
        return {s: int(vc.get(s, 0)) for s in STATUSES}

    def first_unresolved(self):
        rows = self.m[self.m["acquisition_status"] == "missing"]
        return rows["id"].iloc[0] if len(rows) else None

    def run(self, only: str | None = None) -> None:
        self.cfg.watch_dir.mkdir(parents=True, exist_ok=True)
        queue = [only] if only else list(self.m.loc[self.m["acquisition_status"] == "missing", "id"])
        if only and only not in self.target_ids:
            raise SystemExit(f"{only} is not one of the {len(self.target_ids)} target videos.")
        self.ui.show(f"{len(queue)} item(s) to go. Counts now: {self.counts()}")
        for item_id in queue:
            if self.acquire(item_id) == "quit":
                break
        self.ui.show(f"\nStopped. Counts: {self.counts()}  (progress is saved; rerun to resume)")

    def review(self) -> None:
        self.cfg.watch_dir.mkdir(parents=True, exist_ok=True)
        rows = list(self.m.loc[self.m["acquisition_status"].isin(["ambiguous", "failed"]), "id"])
        self.ui.show(f"{len(rows)} item(s) to review.")
        for item_id in rows:
            self.show_item(item_id)
            st = self.status(item_id)
            q = self.m.at[item_id, "quarantine_path"]
            self.ui.show(f"  status    {st}: {self.m.at[item_id, 'status_reason']}")
            self.ui.show(f"  file      {q or '(none)'}")
            choices = {"d": "download it again now"}
            if q and Path(q).exists() and st == "ambiguous":
                choices["a"] = "accept the quarantined file (manual override)"
            choices.update({"m": "mark missing (back in the normal queue)", "n": "next", "q": "quit"})
            choice = self.ui.ask("  What now?", choices)
            if choice == "q":
                break
            if choice == "a":
                self.accept_quarantined(item_id)
            elif choice == "m":
                self.set(item_id, acquisition_status="missing", status_reason="reset during review")
                self.save()
                self.log("reset_to_missing", item_id)
            elif choice == "d":
                if self.acquire(item_id) == "quit":
                    break
        self.ui.show(f"\nCounts: {self.counts()}")

    def print_status(self, on_disk: bool) -> None:
        self.ui.show(f"Manifest: {self.cfg.manifest_path} "
                     f"({'exists' if on_disk else 'NOT created yet - built in memory only'})")
        self.ui.show(f"Target set: {len(self.m)} videos, fingerprint {EXPECTED_FINGERPRINT[:16]}... OK")
        self.ui.show(f"Counts: {self.counts()}")
        changes = self.reconcile(apply=False)
        self.ui.show(f"Reconcile would change {len(changes)} row(s)" + (":" if changes else "."))
        for c in changes[:20]:
            self.ui.show(f"  {c[0]}  {c[1]}  ({c[2]})")
        wd = self.cfg.watch_dir
        if wd.exists():
            n = len(self.watch_entries())
            self.ui.show(f"Watch folder: {wd} ({'empty' if n == 0 else f'{n} file(s) - must be emptied first'})")
        else:
            self.ui.show(f"Watch folder: {wd} (does not exist yet; created on first run)")
        if "onedrive" in str(wd).lower():
            self.ui.show("  WARNING: the watch folder is inside OneDrive; use a local folder instead.")
        nxt = self.first_unresolved()
        if nxt:
            self.ui.show("Next item:")
            self.show_item(nxt)


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--status", action="store_true", help="read-only summary; changes nothing")
    mode.add_argument("--init", action="store_true", help="create the manifest and stop")
    mode.add_argument("--review", action="store_true", help="revisit ambiguous / failed items")
    mode.add_argument("--only", metavar="ID", help="(re)do a single item")
    ap.add_argument("--watch-dir", type=Path, default=Config.watch_dir)
    ap.add_argument("--no-open", action="store_true", help="don't open URLs in the browser")
    ap.add_argument("--no-clipboard", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config(watch_dir=args.watch_dir, open_browser=not args.no_open,
                 copy_clipboard=not args.no_clipboard)
    target, protected, known = load_target_frame(cfg)
    helper = Helper(cfg, target, protected, known)

    if args.status:
        on_disk = helper.load_manifest(create=False)
        helper.print_status(on_disk)
        return

    if "onedrive" in str(cfg.watch_dir).lower():
        raise SystemExit("The watch folder is inside OneDrive; choose a local folder (see --watch-dir).")
    with InstanceLock(cfg.lock_path):
        existed = cfg.manifest_path.exists()
        helper.load_manifest(create=True)
        if args.init:
            print(f"Manifest {'already existed' if existed else 'created'}: {cfg.manifest_path} "
                  f"({len(helper.m)} rows). Counts: {helper.counts()}")
            return
        helper.build_known_hashes()
        changes = helper.reconcile(apply=True)
        if changes:
            print(f"Reconciled {len(changes)} row(s) with the files on disk.")
        if args.review:
            helper.review()
        else:
            helper.run(only=args.only)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Progress up to the last completed step is saved; rerun to resume.")

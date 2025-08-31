import argparse
import os
import re
from typing import Set, List

OFFICIAL_UPLOADERS: List[str] = [
    "UFC",
    "UFC - Ultimate Fighting Championship",
    "UFC Brasil",
    "UFC Europe",
    "UFC Español",
    "UFC Espanol",
    "UFC Arabia",
    "UFC India",
    "UFC FIGHT PASS",
    "UFC Fight Pass",
]

ALLOWED_EXTS = {".mp4", ".mkv", ".webm", ".m4a"}
FORMAT_SUFFIX_RE = re.compile(r"\.f\d+(?:-\d+)?$")


def normalize_stem(filename: str) -> str:
    base, ext = os.path.splitext(filename)
    # Drop yt-dlp format code suffixes like .f399, .f140-9
    base = FORMAT_SUFFIX_RE.sub("", base)
    return base


def is_official_by_filename(stem: str) -> bool:
    # Our outtmpl: "%(uploader)s_%(upload_date)s_%(title).200B.%(ext)s"
    # Uploader is the substring before the first underscore
    if "_" not in stem:
        return False
    uploader = stem.split("_", 1)[0]
    return uploader in OFFICIAL_UPLOADERS


def count_fights(directory: str, official_only: bool) -> int:
    seen: Set[str] = set()
    if not os.path.isdir(directory):
        return 0
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith(".part"):
                continue
            _, ext = os.path.splitext(name)
            if ext.lower() not in ALLOWED_EXTS:
                continue
            stem = normalize_stem(name)
            if official_only and not is_official_by_filename(stem):
                continue
            seen.add(stem)
    return len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description="Count unique fights in a directory by deduplicating video/audio pairs and yt-dlp format suffixes.")
    parser.add_argument("--dir", default="ufc_videos", help="Directory to scan (default: ufc_videos)")
    parser.add_argument("--official-only", action="store_true", help="Only count fights from official UFC uploaders (based on filename prefix)")
    args = parser.parse_args()

    count = count_fights(args.dir, args.official_only)
    print(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 
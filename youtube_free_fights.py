import argparse
import datetime
import os
import sys
from typing import List, Optional, Dict, Any, Set
import unicodedata

try:
    import yt_dlp as ytdlp  # Preferred maintained fork
except ImportError:
    # Fallback to youtube_dl if yt_dlp is not available
    import youtube_dl as ytdlp  # type: ignore


# Official UFC channel handles
OFFICIAL_UFC_HANDLES: List[str] = [
    "@UFC",
    "@UFCBrasil",
    "@UFCEurope",
    "@UFCEspanol",
    "@UFCArabia",
    "@UFCIndia",
    "@UFCFightPass",
]


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_date_filter(days_back: int = 365 * 10) -> str:
    # yt-dlp accepts date strings in YYYYMMDD
    today = datetime.date.today()
    after = today - datetime.timedelta(days=days_back)
    return after.strftime("%Y%m%d")


def strip_accents(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def title_matches_full_or_free(title: str) -> bool:
    if not title:
        return False
    t = strip_accents(title).lower()
    patterns = [
        # English
        "full fight",
        "free fight",
        # Spanish
        "pelea completa",
        "combate completo",
        "pelea gratis",
        # Portuguese
        "luta completa",
        "luta gratis",
        # French
        "combat complet",
        # Arabic
        "نزال كامل",
        "النزال الكامل",
        "نزال مجاني",
    ]
    return any(p in t for p in patterns)

# New: marathon exclusion
MARATHON_MIN_DURATION_SEC = 90 * 60

def title_is_marathon(title: str) -> bool:
    if not title:
        return False
    t = strip_accents(title).lower()
    patterns = [
        "marathon",      # EN/FR
        "maraton",       # ES (without accent)
        "maratón",       # ES
        "maratona",      # PT
    ]
    return any(p in t for p in patterns)

def is_marathon_by_duration(info: Dict[str, Any]) -> bool:
    dur = info.get("duration")
    try:
        return bool(dur) and int(dur) >= MARATHON_MIN_DURATION_SEC
    except Exception:
        return False


def load_urls(file_path: str) -> List[str]:
    urls: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def make_ydl_opts(output_dir: str, dateafter: Optional[str]) -> dict:
    # Safer output template for Windows
    outtmpl = os.path.join(output_dir, "%(uploader)s_%(upload_date)s_%(title).200B.%(ext)s")

    # Allowlist of official UFC channel handles and common uploader names
    allowed_handles = {
        "@UFC",
        "@UFCBrasil",
        "@UFCEurope",
        "@UFCEspanol",
        "@UFCArabia",
        "@UFCIndia",
        "@UFCFightPass",
    }
    allowed_uploaders = {
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
    }

    # Match-filter: only allow official UFC channels and titles that contain the exact phrase "Full Fight"
    def only_official_ufc_full_fight(info_dict, *, incomplete: bool):
        # Defer filtering until metadata is complete to avoid false negatives during search listing
        title = (info_dict.get("title") or "").strip()
        channel_url = (info_dict.get("channel_url") or info_dict.get("uploader_url") or "").strip()
        uploader = (info_dict.get("uploader") or info_dict.get("channel") or "").strip()

        # Try to extract @handle from channel/uploader URL if present
        handle = ""
        if "/@" in channel_url:
            try:
                handle = channel_url.split("/@", 1)[1].split("/", 1)[0]
                handle = f"@{handle}"
            except Exception:
                handle = ""

        # If metadata is incomplete, let yt-dlp fetch full info first
        if incomplete:
            return None

        # Channel allowlist check
        allowed_channel = False
        if handle and handle in allowed_handles:
            allowed_channel = True
        elif uploader and uploader in allowed_uploaders:
            allowed_channel = True

        if not allowed_channel:
            return "skip: not an official UFC channel"

        # Title phrase check (multilingual Full/Free fight) and marathon exclusion
        if not title:
            return "skip: missing title"
        if title_is_marathon(title) or is_marathon_by_duration(info_dict):
            return "skip: marathon"
        if not title_matches_full_or_free(title):
            return "skip: title does not match Full/Free Fight"

        return None

    opts = {
        "outtmpl": outtmpl,
        "restrictfilenames": True,
        "nooverwrites": True,
        "continuedl": True,
        "ignoreerrors": True,
        "concurrent_fragment_downloads": 4,
        # Prefer mp4 where possible; fall back to best available
        "format": "bv*[ext=mp4]+ba[ext=m4a]/bv*+ba/best",
        "merge_output_format": "mp4",
        # Keep things quiet but informative
        "quiet": False,
        "noprogress": False,
        # Throttle retries
        "retries": 5,
        "fragment_retries": 5,
        # Skip live streams
        "skip_download": False,
        # Enforce: official UFC channels + "Full Fight" title
        "match_filter": only_official_ufc_full_fight,
        # Maintain an archive of downloaded video IDs to avoid re-downloading
        "download_archive": os.path.join(output_dir, "downloaded_archive.txt"),
    }

    if dateafter:
        # Filter videos uploaded after the given date
        # Works for YouTube channels/playlists/search results
        opts["dateafter"] = dateafter

    return opts


def download_targets(targets: List[str], ydl_opts: dict) -> None:
    if not targets:
        print("No targets to download.")
        return
    with ytdlp.YoutubeDL(ydl_opts) as ydl:
        for t in targets:
            try:
                ydl.download([t])
            except Exception as e:
                print(f"Failed: {t} -> {e}")


def build_search_query(query: str, max_results: int) -> str:
    # Use ytsearchdate to prioritize newest, which we filter further via dateafter
    # Example: ytsearchdate100:UFC Free Fight
    return f"ytsearchdate{max_results}:{query}"


def channel_videos_url(handle: str) -> str:
    return f"https://www.youtube.com/{handle}/videos"


def within_date_bounds(info: Dict[str, Any], after: Optional[str], before: Optional[str]) -> bool:
    upload_date: Optional[str] = info.get("upload_date")
    if not after and not before:
        return True
    if not upload_date:
        return False
    if after and upload_date < after:
        return False
    if before and upload_date > before:
        return False
    return True


def index_official_full_fights(after: Optional[str], before: Optional[str], max_per_channel: int) -> List[str]:
    urls: List[str] = []
    seen_ids: Set[str] = set()

    flat_opts = {
        "skip_download": True,
        "extract_flat": True,
        "quiet": True,
        "noprogress": True,
        # Make sure we still resolve additional pages where possible
        "playlistend": max_per_channel if max_per_channel > 0 else None,
    }

    # Use one extractor instance for flat listing
    with ytdlp.YoutubeDL(flat_opts) as ydl:
        for handle in OFFICIAL_UFC_HANDLES:
            url = channel_videos_url(handle)
            try:
                info = ydl.extract_info(url, download=False)
            except Exception as e:
                print(f"Index failed for {handle}: {e}")
                continue

            if not info or "entries" not in info:
                continue

            count = 0
            for entry in info.get("entries", []) or []:
                if not isinstance(entry, dict):
                    continue
                title = (entry.get("title") or "")
                # Skip marathons early by title
                if title_is_marathon(title):
                    continue
                if not title_matches_full_or_free(title):
                    continue

                # If flat mode didn't include upload_date, fetch full metadata to check date
                video_id = entry.get("id") or ""
                video_url = entry.get("url") or (f"https://www.youtube.com/watch?v={video_id}" if video_id else "")
                if not video_url:
                    continue

                # Prevent duplicates across channels
                if video_id and video_id in seen_ids:
                    continue

                # Fetch full metadata to verify date bounds when needed
                need_meta_check = bool(after or before)
                if need_meta_check:
                    try:
                        full_info = ydl.extract_info(video_url, download=False)
                    except Exception:
                        continue
                    # Exclude by duration if clearly a marathon
                    if is_marathon_by_duration(full_info or {}):
                        continue
                    if not within_date_bounds(full_info or {}, after, before):
                        continue

                urls.append(video_url)
                if video_id:
                    seen_ids.add(video_id)
                count += 1
                if max_per_channel and count >= max_per_channel:
                    break

    return urls


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download YouTube UFC Free Fight videos from the last 10 years (or custom range) using yt_dlp.")

    parser.add_argument(
        "--query",
        default="UFC Free Fight",
        help="Search query for yt_dlp ytsearch (default: 'UFC Free Fight')",
    )
    parser.add_argument(
        "--urls-file",
        default="",
        help="Optional path to a text file with one URL per line to download (overrides --query if provided)",
    )
    parser.add_argument(
        "--outdir",
        default="ufc_videos",
        help="Output directory (default: ufc_videos)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=200,
        help="Maximum number of search results to consider (default: 200)",
    )
    parser.add_argument(
        "--after",
        default="",
        help="Only download videos uploaded after this date (YYYYMMDD). If not set, last 10 years is assumed.",
    )
    parser.add_argument(
        "--before",
        default="",
        help="Optional date upper bound (YYYYMMDD).",
    )
    parser.add_argument(
        "--noprompt",
        action="store_true",
        help="Run non-interactively without prompts.",
    )
    # New: official index mode
    parser.add_argument(
        "--official-index",
        action="store_true",
        help="Index 'Full Fight' videos from official UFC channels and download from that curated list.",
    )
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Only perform indexing (no downloads).",
    )
    parser.add_argument(
        "--index-file",
        default="ufc_full_fights_urls.txt",
        help="Path to save the indexed list of URLs (used with --official-index).",
    )
    parser.add_argument(
        "--max-per-channel",
        type=int,
        default=500,
        help="Maximum videos to index per official channel (default: 500).",
    )

    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    ensure_output_dir(args.outdir)

    # Compute dateafter filter
    dateafter = args.after or build_date_filter(days_back=365 * 10)

    ydl_opts = make_ydl_opts(args.outdir, dateafter=dateafter)

    # Optional datebefore support
    if args.before:
        ydl_opts["datebefore"] = args.before

    targets: List[str]

    if args.urls_file:
        # Load explicit list
        if not os.path.isfile(args.urls_file):
            print(f"URLs file not found: {args.urls_file}")
            return 1
        targets = load_urls(args.urls_file)
    elif args.official_index:
        # Build curated list from official channels first
        after = args.after or None
        before = args.before or None
        print("Indexing official UFC channels for 'Full Fight' videos...")
        targets = index_official_full_fights(after=after, before=before, max_per_channel=args.max_per_channel)
        print(f"Indexed {len(targets)} videos.")

        # Save the index if requested
        if args.index_file:
            try:
                with open(args.index_file, "w", encoding="utf-8") as f:
                    for u in targets:
                        f.write(u + "\n")
                print(f"Saved index to {args.index_file}")
            except Exception as e:
                print(f"Warning: failed to write index file: {e}")

        if args.index_only:
            return 0
    else:
        # Fallback: ytsearch
        targets = [build_search_query(args.query, args.max_results)]

    # Confirm scope when interactive
    if not args.noprompt and not args.urls_file:
        mode = "official-index" if args.official_index else "search"
        print(f"About to download in mode={mode}: targets={len(targets)}, after={dateafter}, before={args.before or 'None'}")
        resp = input("Proceed? [y/N]: ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return 0

    download_targets(targets, ydl_opts)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
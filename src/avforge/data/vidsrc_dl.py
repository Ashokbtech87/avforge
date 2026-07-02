"""VidSrc m3u8 extractor + downloader.

VidSrc embed pages block yt-dlp via TLS fingerprinting. This module:
  1. Fetches the embed page using curl-cffi (browser TLS impersonation)
  2. Extracts the m3u8 stream URL from the page
  3. Downloads the stream using ffmpeg (which handles HLS natively)

This is the same approach Streambert uses (via deno), but in pure Python.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def fetch_embed_page(url: str) -> str:
    """Fetch a VidSrc embed page using curl-cffi (browser TLS impersonation)."""
    from curl_cffi import requests as cffi_requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Referer": "https://vidsrc.to/",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    resp = cffi_requests.get(url, headers=headers, impersonate="chrome", timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_m3u8(html: str) -> str | None:
    """Extract the m3u8 URL from a VidSrc embed page."""
    # VidSrc pages typically have the m3u8 URL in a JSON config or script tag.
    # Common patterns:
    #   {"sources":[{"file":"https://.../playlist.m3u8"}]}
    #   src: "https://.../index.m3u8"
    #   file:"https://.../master.m3u8"
    patterns = [
        r'"file"\s*:\s*"(https?://[^"]+\.m3u8[^"]*)"',
        r'src\s*[:=]\s*["\'](https?://[^"\']+\.m3u8[^"\']*)["\']',
        r'"sources"\s*:\s*\[.*?"file"\s*:\s*"(https?://[^"]+\.m3u8[^"]*)"',
        r'(https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*)',
    ]
    for pat in patterns:
        match = re.search(pat, html, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    return None


def extract_iframe_src(html: str) -> str | None:
    """VidSrc embed pages may have a nested iframe with the actual player."""
    # Look for iframe src pointing to vidsrc.to/vapi or similar
    patterns = [
        r'<iframe[^>]+src=["\']([^"\']+)["\']',
        r'src=["\']([^"\']*vapi[^"\']*)["\']',
        r'src=["\']([^"\']*player[^"\']*)["\']',
    ]
    for pat in patterns:
        match = re.search(pat, html, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def resolve_m3u8(embed_url: str, max_depth: int = 3) -> str | None:
    """Fetch the embed page and resolve to a direct m3u8 URL.

    VidSrc may nest iframes; follow them up to max_depth levels.
    """
    url = embed_url
    for _ in range(max_depth):
        print(f"  fetching: {url}", flush=True)
        try:
            html = fetch_embed_page(url)
        except Exception as e:
            print(f"  fetch failed: {e}", flush=True)
            return None

        # Try to find m3u8 directly
        m3u8 = extract_m3u8(html)
        if m3u8:
            print(f"  found m3u8: {m3u8[:80]}...", flush=True)
            return m3u8

        # Try to find a nested iframe
        iframe = extract_iframe_src(html)
        if iframe:
            if iframe.startswith("//"):
                iframe = "https:" + iframe
            elif iframe.startswith("/"):
                iframe = "https://vidsrc.to" + iframe
            print(f"  following iframe: {iframe[:80]}...", flush=True)
            url = iframe
            continue

        # No m3u8 and no iframe — try to find any .m3u8 in the raw HTML
        print(f"  no m3u8 or iframe found in page ({len(html)} bytes)", flush=True)
        # Debug: print script tags
        scripts = re.findall(r'<script[^>]*>(.*?)</script>', html, re.DOTALL)
        for i, s in enumerate(scripts[:5]):
            if ".m3u8" in s or "source" in s.lower() or "player" in s.lower():
                print(f"  script[{i}]: {s[:200]}...", flush=True)
        return None

    return None


def download_m3u8(m3u8_url: str, out_path: Path, ffmpeg: str,
                  resolution: str = "720") -> bool:
    """Download an HLS m3u8 stream using ffmpeg."""
    W, H = {"720": (1280, 720), "1080": (1920, 1080), "best": (1920, 1080)}.get(resolution, (1280, 720))
    cmd = [
        ffmpeg, "-y",
        "-i", m3u8_url,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        "-user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-headers", "Referer: https://vidsrc.to/\r\n",
        str(out_path),
    ]
    print(f"  downloading via ffmpeg HLS...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(f"  ffmpeg failed: {r.stderr[:300]}", flush=True)
        return False
    return out_path.exists()


def download_vidsrc(tmdb_id: int, media_type: str, out_path: Path,
                    ffmpeg: str, resolution: str = "720",
                    season: int = 1, episode: int = 1) -> bool:
    """Full VidSrc download: embed page → m3u8 → ffmpeg download.

    Args:
        tmdb_id: TMDB movie/TV ID.
        media_type: "movie" or "tv".
        out_path: target .mp4 path.
        ffmpeg: path to ffmpeg binary.
        resolution: "720", "1080", "best".
        season/episode: for TV series.
    Returns:
        True on success.
    """
    if media_type == "movie":
        embed_url = f"https://vidsrc.to/embed/movie/{tmdb_id}"
    else:
        embed_url = f"https://vidsrc.to/embed/tv/{tmdb_id}/{season}/{episode}"

    print(f"  VidSrc URL: {embed_url}", flush=True)

    # Try multiple VidSrc mirror domains
    mirrors = [
        embed_url,
        f"https://vidsrc.net/embed/movie/{tmdb_id}" if media_type == "movie"
        else f"https://vidsrc.net/embed/tv/{tmdb_id}/{season}/{episode}",
    ]

    for url in mirrors:
        m3u8 = resolve_m3u8(url)
        if m3u8:
            if download_m3u8(m3u8, out_path, ffmpeg, resolution):
                return True
            print(f"  download failed for {url}, trying next mirror...", flush=True)

    print(f"  all mirrors failed for {media_type} {tmdb_id}", flush=True)
    return False


if __name__ == "__main__":
    # Quick test
    from avforge.data.tmdb_pipeline import _find_ffmpeg, load_api_key
    from pathlib import Path

    ff = _find_ffmpeg()
    print(f"ffmpeg: {ff}")

    # Test with The Shawshank Redemption (tmdb_id=278, rating 8.7)
    out = Path("data/tmdb_test/raw/movie_278.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    ok = download_vidsrc(278, "movie", out, ff, "720")
    if ok:
        print(f"\n✓ Downloaded: {out} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("\n✗ Download failed")
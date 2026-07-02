"""Try all known VidSrc mirror domains to find one that's reachable."""
from curl_cffi import requests as cffi
import time

MIRRORS = [
    "https://vidsrc.to/embed/movie/278",
    "https://vidsrc.net/embed/movie/278",
    "https://vidsrc.xyz/embed/movie/278",
    "https://vidsrc.me/embed/movie/278",
    "https://vidsrc.org/embed/movie/278",
    "https://vidsrc.icu/embed/movie/278",
    "https://2embed.to/embed/movie/278",
    "https://multiembed.to/embed/movie/278",
    "https://embedsu.com/embed/movie/278",
    "https://moviee.tv/embed/movie/278",
    "https://playembed.net/embed/movie/278",
    "https://vembed.net/embed/movie/278",
    "https://embed.wf/embed/movie/278",
    "https://gomo.to/embed/movie/278",
    "https://api.123embed.net/embed/movie/278",
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://vidsrc.to/",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

for url in MIRRORS:
    try:
        r = cffi.get(url, impersonate="chrome", timeout=10, headers=headers)
        print(f"✓ {url} → {r.status_code} ({len(r.text)} bytes)")
        if r.status_code == 200 and len(r.text) > 500:
            # Check for m3u8 or iframe
            import re
            m3u8 = re.search(r'(https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*)', r.text)
            iframe = re.search(r'<iframe[^>]+src=["\']([^"\']+)', r.text, re.I)
            if m3u8:
                print(f"  → m3u8: {m3u8.group(1)[:100]}")
            if iframe:
                print(f"  → iframe: {iframe.group(1)[:100]}")
    except Exception as e:
        err = str(e)[:80]
        print(f"✗ {url} → {err}")
    time.sleep(0.5)
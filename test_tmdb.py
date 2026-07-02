"""Quick test of TMDB API connection."""
from pathlib import Path
from avforge.data.tmdb_pipeline import load_api_key, fetch_trending, fetch_popular, fetch_top_rated

key_file = Path(r"D:\AIModels\TMDB_API.txt")
token = load_api_key(key_file)
print(f"Token loaded: {token[:25]}...")

print("\n--- Trending Movies (this week) ---")
d = fetch_trending(token, "movie", "week", 1)
results = d.get("results", [])
print(f"Got {len(results)} results")
for r in results[:5]:
    title = r.get("title", "")
    year = (r.get("release_date", ""))[:4]
    print(f"  {r['id']:>8}  {title} ({year})")

print("\n--- Popular TV ---")
d2 = fetch_popular(token, "tv", 1)
results2 = d2.get("results", [])
print(f"Got {len(results2)} results")
for r in results2[:5]:
    title = r.get("name", "")
    year = (r.get("first_air_date", ""))[:4]
    print(f"  {r['id']:>8}  {title} ({year})")

print("\n✓ TMDB API connection works!")
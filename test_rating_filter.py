"""Test the TMDB rating filter — fetch top_rated movies and check how many have rating >= 8.0."""
from pathlib import Path
from avforge.data.tmdb_pipeline import load_api_key, fetch_top_rated, fetch_trending

token = load_api_key(Path(r"D:\AIModels\TMDB_API.txt"))
MIN_RATING = 8.0

print(f"--- Top Rated Movies (rating >= {MIN_RATING}) ---")
total = 0
passed = 0
for page in range(1, 6):
    data = fetch_top_rated(token, "movie", page)
    for item in data.get("results", []):
        total += 1
        vote = item.get("vote_average", 0)
        if vote >= MIN_RATING:
            passed += 1
            title = item.get("title", "")
            year = (item.get("release_date", ""))[:4]
            print(f"  ⭐{vote:.1f}  {item['id']:>8}  {title} ({year})")
    if page == 1:
        print(f"  ... page 1: {passed}/{total} titles have rating >= {MIN_RATING}")

print(f"\nTotal scanned: {total}, passed rating filter (>= {MIN_RATING}): {passed}")

print(f"\n--- Trending Movies This Week (rating >= {MIN_RATING}) ---")
data = fetch_trending(token, "movie", "week", 1)
trend_total = 0
trend_passed = 0
for item in data.get("results", []):
    trend_total += 1
    vote = item.get("vote_average", 0)
    if vote >= MIN_RATING:
        trend_passed += 1
        print(f"  ⭐{vote:.1f}  {item.get('title','')}")
print(f"Trending: {trend_passed}/{trend_total} have rating >= {MIN_RATING}")
"""Download and process GokuScraper/seedance-2-prompts-datasets for AV-Forge training.

Pulls the Hugging Face dataset (metadata.jsonl + videos + covers), maps each prompt
to SeedVideoBench 2.0 fine-grained categories from docs/03_DATA.md / Seedance.pdf,
and writes a training-ready manifest under data/seedance_prompts/.

Usage:
  python -m avforge.data.seedance_prompts_pipeline --download --token $HF_TOKEN
  python -m avforge.data.seedance_prompts_pipeline --process
  python -m avforge.data.seedance_prompts_pipeline --download --process --token $HF_TOKEN
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# SeedVideoBench 2.0 fine-grained categories (docs/03_DATA.md Tables 3–8)
MOTION_CATEGORIES = [
    "Holidays / Festivals", "Consumer Visual Effects", "Counter-Reality Instructions",
    "Cinematic Visual Effects", "Same-Type Interaction", "Cross-Type Interaction",
    "Group Coordinated Motion", "Advanced Camera Movement", "Special Camera Shots",
    "Editing Rhythm", "Combined Shot Instructions", "Physical Feedback",
    "Physical Phenomena", "Natural Phenomena", "Text Overlay", "Short Text",
    "Creative Text", "Long Script", "Abstract Challenges", "Multi-Entity Feature Match",
    "Knowledge Assessment", "Compound Multi-Instructions", "Surreal Motion",
    "Intense Sports Motion", "Fine Hand Motion", "Anthropomorphic Motion",
    "Emotion & Expression", "Visual Style", "Lighting & Color Tone",
    "Framing / Composition",
]

AUDIO_CATEGORIES = [
    "Chinese Dialect / Accent", "Chinese Multi-Person Dialogue",
    "Chinese Variety Show Voice", "Chinese Opera", "English", "Minority Languages",
    "Singing / Rap", "Spatial Scene", "Off-Screen Voice", "Non-Verbal Voice",
    "Voice + Action Interaction", "Object Interaction Sound", "Animal Sound",
    "Ambient / Background Sound", "Special Effects (ASMR, etc.)",
    "Instruments & Audio", "Dual-Channel Audio",
]

# Keyword → category mapping (bilingual)
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Intense Sports Motion": [
        "skating", "parkour", "combat", "fight", "sport", "athlete", "jump", "run",
        "martial", "boxing", "wrestling", "花样滑冰", "格斗", "运动", "竞技", "武术",
    ],
    "Cinematic Visual Effects": [
        "vfx", "visual effect", "explosion", "cgi", "特效", "视觉特效", "慢动作", "slow motion",
        "shockwave", "激波",
    ],
    "Natural Phenomena": [
        "rain", "storm", "water", "fire", "snow", "wind", "weather", "glacier", "ocean",
        "雨", "雪", "风", "水", "火", "自然", "冰川", "海浪",
    ],
    "Physical Phenomena": [
        "collision", "shatter", "glass", "fluid", "physics", "gravity", "碰撞", "破碎",
        "物理", "重力",
    ],
    "Emotion & Expression": [
        "emotion", "tear", "smile", "gaze", "expression", "heart", "love", "nostalgia",
        "情感", "表情", "眼神", "泪", "爱", "情绪",
    ],
    "Fine Hand Motion": [
        "hand", "finger", "tweezer", "close-up of hands", "手", "手指", "特写",
    ],
    "Long Script": [
        "0-4 seconds", "4-9 seconds", "9-15 seconds", "shot", "cut to", "camera",
        "镜头", "秒", "分镜", "cut",
    ],
    "Compound Multi-Instructions": [
        "then", "after", "while", "simultaneously", "同时", "然后", "接着", "随后",
    ],
    "Counter-Reality Instructions": [
        "impossible", "upward rain", "floating", "counter-gravity", "surreal physics",
        "不可能", "反重力", "漂浮", "逆",
    ],
    "Surreal Motion": [
        "surreal", "dream", "fantasy", "magical", "超现实", "梦幻", "奇幻",
    ],
    "Multi-Entity Feature Match": [
        "twins", "sisters", "multiple", "crowd", "group of", "five", "four", "three people",
        "多人", "双胞胎", "群体", "姐妹",
    ],
    "Advanced Camera Movement": [
        "dolly", "crane", "tracking", "push-in", "pull-back", "handheld", "fpv", "drone",
        "推镜", "拉镜", "摇镜", "跟拍", "航拍",
    ],
    "Special Camera Shots": [
        "close-up", "wide shot", "over-the-shoulder", "top-down", "low angle", "特写",
        "俯拍", "仰拍", "过肩",
    ],
    "Editing Rhythm": [
        "cut", "montage", "rhythm", "beat", "transition", "剪辑", "节奏", "转场",
    ],
    "Text Overlay": [
        "subtitle", "text overlay", "neon sign", "letter", "字幕", "文字", "霓虹",
    ],
    "Creative Text": [
        "calligraphy", "neon", "sign", "书法", "招牌",
    ],
    "Chinese Opera": ["opera", "京剧", "戏曲", "昆曲"],
    "Chinese Dialect / Accent": ["dialect", "sichuan", "cantonese", "方言", "四川", "粤语", "东北"],
    "Singing / Rap": ["sing", "rap", "melody", "song", "唱", "说唱", "歌声", "音乐"],
    "Animal Sound": ["animal", "bird", "dog", "cat", "动物", "鸟", "虎", "蛇"],
    "Ambient / Background Sound": ["ambient", "background", "bgm", "环境音", "背景音"],
    "Instruments & Audio": ["piano", "guitar", "instrument", "violin", "乐器", "钢琴", "吉他"],
    "Voice + Action Interaction": ["lip-sync", "dialogue", "speech", "voiceover", "对白", "台词", "旁白"],
    "Commercial": ["product demo", "brand video", "advertisement", "unboxing", "广告片", "品牌宣传", "产品展示"],
    "Visual Style": ["anime", "oil painting", "gongbi", "felt", "style", "风格", "工笔", "油画"],
    "Anthropomorphic Motion": ["anthropomorphic", "talking animal", "拟人", "会说话"],
    "Holidays / Festivals": ["festival", "holiday", "christmas", "春节", "节日", "庆典"],
    "Knowledge Assessment": ["historical", "scientific", "documentary", "历史", "科学", "纪录"],
    "Abstract Challenges": ["abstract", "concept", "metaphor", "抽象", "概念", "隐喻"],
    "Group Coordinated Motion": ["marching", "formation", "choreograph", "列队", "编队", "整齐"],
    "Cross-Type Interaction": ["human and", "animal and", "人与", "跨物种"],
    "Same-Type Interaction": ["two people", "duel", "confrontation", "对峙", "两人", "双人"],
    "Spatial Scene": ["binaural", "spatial", "stereo", "立体声", "空间", "双耳"],
    "Object Interaction Sound": ["foley", "clink", "footstep", "脚步声", "碰撞声"],
    "Off-Screen Voice": ["off-screen", "narration", "画外音", "旁白"],
    "Non-Verbal Voice": ["breath", "sigh", "laugh", "呼吸", "叹息"],
    "English": ["english", "english dialogue"],
    "Minority Languages": ["japanese", "korean", "spanish", "indonesian", "日语", "韩语"],
    "Dual-Channel Audio": ["binaural", "dual-channel", "stereo", "双耳", "双声道"],
    "Special Effects (ASMR, etc.)": ["asmr", "whisper", "texture sound"],
    "Physical Feedback": ["impact", "recoil", "bounce", "反弹", "冲击"],
    "Framing / Composition": ["rule of thirds", "composition", "framing", "构图", "取景"],
    "Lighting & Color Tone": ["lighting", "golden hour", "color grade", "灯光", "色调", "光影"],
    "Consumer Visual Effects": ["ugc", "tiktok", "filter effect", "滤镜"],
}

# HF source category → default task type
_HF_CATEGORY_TASK = {
    "Entertainment": "T2V",
    "Content Creation": "T2V",
    "Commercial": "T2V",
}


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "seedance_prompts"


def download_dataset(out_dir: Path, token: str | None = None, max_workers: int = 2) -> Path:
    """Download full HF dataset snapshot into out_dir/raw_hf (resumable, rate-limit safe)."""
    import time

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    raw_dir = out_dir / "raw_hf"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"Downloading GokuScraper/seedance-2-prompts-datasets → {raw_dir}")

    for attempt in range(1, 51):
        try:
            path = snapshot_download(
                repo_id="GokuScraper/seedance-2-prompts-datasets",
                repo_type="dataset",
                local_dir=str(raw_dir),
                token=tok,
                max_workers=max_workers,
                resume_download=True,
            )
            print(f"Download complete: {path}")
            return raw_dir
        except HfHubHTTPError as exc:
            if "429" in str(exc) and attempt < 50:
                wait_s = min(300, 30 * attempt)
                print(f"Rate limited (attempt {attempt}/50). Waiting {wait_s}s then resuming...")
                time.sleep(wait_s)
                continue
            raise

    raise RuntimeError("Download failed after 50 rate-limit retries")


def download_missing_media(
    raw_dir: Path,
    token: str | None = None,
    media_type: str = "videos",
    delay_s: float = 3.0,
    limit: int | None = None,
) -> dict:
    """Download missing video/cover files one-by-one (rate-limit safe, resumable)."""
    import time

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    tok = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    meta_path = raw_dir / "metadata.jsonl"
    records = _load_metadata(meta_path)

    key = "v" if media_type == "videos" else "c"
    missing: list[str] = []
    for rec in records:
        rel = rec.get("media", {}).get(key) or (rec.get("file_name", "") if key == "v" else "")
        if not rel:
            continue
        if not (raw_dir / rel).exists():
            missing.append(rel)

    if limit:
        missing = missing[:limit]

    print(f"Downloading {len(missing)} missing {media_type} files...")
    ok = fail = 0
    for i, rel in enumerate(missing, 1):
        for attempt in range(1, 11):
            try:
                hf_hub_download(
                    "GokuScraper/seedance-2-prompts-datasets",
                    rel,
                    repo_type="dataset",
                    token=tok,
                    local_dir=str(raw_dir),
                )
                ok += 1
                if i % 25 == 0 or i == len(missing):
                    print(f"  [{i}/{len(missing)}] ok={ok} fail={fail} latest={rel}")
                break
            except HfHubHTTPError as exc:
                if "429" in str(exc) and attempt < 10:
                    wait_s = min(300, 60 * attempt)
                    print(f"  rate limited on {rel}, waiting {wait_s}s...")
                    time.sleep(wait_s)
                    continue
                fail += 1
                print(f"  FAILED {rel}: {exc}")
                break
        time.sleep(delay_s)

    return {"requested": len(missing), "ok": ok, "fail": fail}


def _load_metadata(meta_path: Path) -> list[dict]:
    records: list[dict] = []
    with meta_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _collect_text(record: dict) -> str:
    parts = [
        record.get("raw_p", ""),
        record.get("i18n", {}).get("en", {}).get("p", ""),
        record.get("i18n", {}).get("en", {}).get("t", ""),
        record.get("i18n", {}).get("zh", {}).get("p", ""),
        record.get("i18n", {}).get("zh", {}).get("t", ""),
        " ".join(record.get("i18n", {}).get("en", {}).get("tags", [])),
        " ".join(record.get("i18n", {}).get("zh", {}).get("tags", [])),
        record.get("slug", ""),
        record.get("category", ""),
    ]
    return " ".join(p for p in parts if p).lower()


def classify_prompt(record: dict) -> list[str]:
    """Map a prompt record to SeedVideoBench fine-grained categories."""
    text = _collect_text(record)
    matched: list[str] = []

    for cat, keywords in _CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in text:
                matched.append(cat)
                break

    # Heuristics from prompt structure
    prompt_len = len(record.get("raw_p", "") or record.get("i18n", {}).get("en", {}).get("p", ""))
    if prompt_len > 400 or re.search(r"\d+-\d+\s*seconds", text):
        if "Long Script" not in matched:
            matched.append("Long Script")
    if prompt_len > 200 and "Compound Multi-Instructions" not in matched:
        if text.count("then") + text.count("cut") + text.count("然后") >= 2:
            matched.append("Compound Multi-Instructions")

    spec = record.get("spec", {})
    if spec.get("duration", 0) >= 10 and "Editing Rhythm" not in matched:
        if "shot" in text or "镜头" in text or "cut" in text:
            matched.append("Editing Rhythm")

    if not matched:
        hf_cat = record.get("category", "Entertainment")
        if hf_cat == "Commercial":
            matched = ["Commercial", "Visual Style", "Long Script"]
        elif hf_cat == "Content Creation":
            matched = ["Long Script", "Visual Style", "Editing Rhythm"]
        else:
            matched = ["Visual Style", "Emotion & Expression", "Cinematic Visual Effects"]

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in matched:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def infer_task_type(record: dict, categories: list[str]) -> str:
    """Infer T2V / I2V / R2V from metadata."""
    text = _collect_text(record)
    if any(k in text for k in ("reference image", "first frame", "参考图", "首帧")):
        return "I2V"
    if any(k in text for k in ("edit", "continuation", "extend", "编辑", "续写", "延长")):
        return "R2V"
    return _HF_CATEGORY_TASK.get(record.get("category", ""), "T2V")


def process_dataset(raw_dir: Path, out_dir: Path) -> dict:
    """Process metadata + media into structured training manifest."""
    meta_path = raw_dir / "metadata.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found in {raw_dir}. Run --download first.")

    videos_dir = out_dir / "videos"
    covers_dir = out_dir / "covers"
    videos_dir.mkdir(parents=True, exist_ok=True)
    covers_dir.mkdir(parents=True, exist_ok=True)

    records = _load_metadata(meta_path)

    manifest_rows: list[dict] = []
    category_counts: Counter = Counter()
    task_counts: Counter = Counter()
    missing_video = 0
    missing_cover = 0

    for rec in records:
        rid = rec.get("id", "")
        media = rec.get("media", {})
        video_rel = media.get("v") or rec.get("file_name", "")
        cover_rel = media.get("c", "")

        video_src = raw_dir / video_rel if video_rel else None
        cover_src = raw_dir / cover_rel if cover_rel else None

        video_dst = videos_dir / f"{rid}.mp4"
        cover_dst = covers_dir / f"{rid}.jpg"

        has_video = video_src and video_src.exists()
        has_cover = cover_src and cover_src.exists()

        # Reference raw_hf paths directly; optionally hardlink into flat dirs when present
        video_ref = ""
        cover_ref = ""
        if has_video:
            video_ref = str(video_src.relative_to(out_dir))
            if not video_dst.exists():
                try:
                    video_dst.hardlink_to(video_src)
                except OSError:
                    pass
        else:
            missing_video += 1

        if has_cover:
            cover_ref = str(cover_src.relative_to(out_dir))
            if not cover_dst.exists():
                try:
                    cover_dst.hardlink_to(cover_src)
                except OSError:
                    pass
        else:
            missing_cover += 1

        categories = classify_prompt(rec)
        task_type = infer_task_type(rec, categories)
        for c in categories:
            category_counts[c] += 1
        task_counts[task_type] += 1

        spec = rec.get("spec", {})
        en = rec.get("i18n", {}).get("en", {})
        zh = rec.get("i18n", {}).get("zh", {})

        manifest_rows.append({
            "id": rid,
            "slug": rec.get("slug", ""),
            "task_type": task_type,
            "hf_category": rec.get("category", ""),
            "categories": ";".join(categories),
            "prompt_en": (en.get("p") or rec.get("raw_p", "")).replace("\n", " ").strip(),
            "prompt_zh": (zh.get("p") or "").replace("\n", " ").strip(),
            "title_en": en.get("t", ""),
            "title_zh": zh.get("t", ""),
            "tags_en": ";".join(en.get("tags", [])),
            "tags_zh": ";".join(zh.get("tags", [])),
            "duration_s": spec.get("duration", ""),
            "width": spec.get("width", ""),
            "height": spec.get("height", ""),
            "resolution": f"{spec.get('width', '')}x{spec.get('height', '')}",
            "safety_rating": spec.get("safety_rating", ""),
            "model_name": rec.get("model_info", {}).get("name", "seedance"),
            "model_version": rec.get("model_info", {}).get("version", "2.0"),
            "is_featured": rec.get("is_featured", False),
            "date": rec.get("date", ""),
            "platform": rec.get("platform", ""),
            "source_link": rec.get("sourceLink", ""),
            "video_path": video_ref,
            "cover_path": cover_ref,
            "video_path_raw": video_rel if has_video else "",
            "cover_path_raw": cover_rel if has_cover else "",
            "has_video": has_video,
            "has_cover": has_cover,
            "license": "CC-BY-4.0",
            "source": "GokuScraper/seedance-2-prompts-datasets",
        })

    manifest_path = out_dir / "manifest.csv"
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else []
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # JSONL for direct training ingestion (prompt → video pairs)
    train_jsonl = out_dir / "train.jsonl"
    with train_jsonl.open("w", encoding="utf-8") as f:
        for row in manifest_rows:
            if not row["has_video"]:
                continue
            entry = {
                "id": row["id"],
                "task": row["task_type"],
                "prompt": row["prompt_en"] or row["prompt_zh"],
                "prompt_zh": row["prompt_zh"],
                "prompt_en": row["prompt_en"],
                "video": row["video_path"],
                "cover": row["cover_path"],
                "categories": row["categories"].split(";"),
                "duration_s": row["duration_s"],
                "resolution": row["resolution"],
                "metadata": {
                    "hf_category": row["hf_category"],
                    "model": f"{row['model_name']}-{row['model_version']}",
                    "source_link": row["source_link"],
                    "license": row["license"],
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Category coverage report aligned with Seedance.pdf evaluation dimensions
    summary = {
        "dataset": "GokuScraper/seedance-2-prompts-datasets",
        "license": "CC-BY-4.0",
        "target_model": "Seedance 2.0 / AV-Forge 2.0",
        "total_records": len(records),
        "with_video": sum(1 for r in manifest_rows if r["has_video"]),
        "with_cover": sum(1 for r in manifest_rows if r["has_cover"]),
        "missing_video": missing_video,
        "missing_cover": missing_cover,
        "task_type_distribution": dict(task_counts),
        "hf_category_distribution": dict(Counter(r["hf_category"] for r in manifest_rows)),
        "seedvideobench_category_coverage": dict(category_counts.most_common()),
        "motion_categories_covered": sorted(
            set(category_counts) & set(MOTION_CATEGORIES)
        ),
        "audio_categories_covered": sorted(
            set(category_counts) & set(AUDIO_CATEGORIES)
        ),
        "motion_categories_missing": sorted(
            set(MOTION_CATEGORIES) - set(category_counts)
        ),
        "audio_categories_missing": sorted(
            set(AUDIO_CATEGORIES) - set(category_counts)
        ),
        "duration_stats": {
            "min": min((r["duration_s"] for r in manifest_rows if r["duration_s"]), default=0),
            "max": max((r["duration_s"] for r in manifest_rows if r["duration_s"]), default=0),
            "avg": round(
                sum(r["duration_s"] for r in manifest_rows if r["duration_s"])
                / max(1, sum(1 for r in manifest_rows if r["duration_s"])),
                2,
            ),
        },
        "outputs": {
            "manifest_csv": str(manifest_path),
            "train_jsonl": str(train_jsonl),
            "videos_dir": str(videos_dir),
            "covers_dir": str(covers_dir),
        },
    }

    summary_path = out_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(records)} records")
    print(f"  videos: {summary['with_video']}  covers: {summary['with_cover']}")
    print(f"  missing videos: {missing_video}  missing covers: {missing_cover}")
    print(f"  manifest: {manifest_path}")
    print(f"  train.jsonl: {train_jsonl}")
    print(f"  summary: {summary_path}")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="avforge.data.seedance_prompts_pipeline",
        description="Download & process Seedance 2.0 prompts dataset from Hugging Face",
    )
    p.add_argument("--out", type=Path, default=_default_data_dir(),
                   help="Output directory (default: data/seedance_prompts)")
    p.add_argument("--download", action="store_true", help="Download full HF dataset snapshot")
    p.add_argument("--download-videos", action="store_true",
                   help="Download missing videos one-by-one (rate-limit safe)")
    p.add_argument("--download-covers", action="store_true",
                   help="Download missing covers one-by-one")
    p.add_argument("--delay", type=float, default=3.0, help="Seconds between media downloads")
    p.add_argument("--limit", type=int, default=None, help="Max files to download this run")
    p.add_argument("--process", action="store_true", help="Process metadata into manifest")
    p.add_argument("--token", default=None, help="Hugging Face token (or set HF_TOKEN env)")
    p.add_argument("--max-workers", type=int, default=2,
                   help="Parallel HF downloads (lower = fewer rate-limit hits)")
    args = p.parse_args(argv)

    if not any([args.download, args.download_videos, args.download_covers, args.process]):
        p.print_help()
        return 1

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw_hf"

    if args.download:
        download_dataset(out_dir, args.token, max_workers=args.max_workers)

    if args.download_videos:
        download_missing_media(raw_dir, args.token, "videos", args.delay, args.limit)

    if args.download_covers:
        download_missing_media(raw_dir, args.token, "covers", args.delay, args.limit)

    if args.process:
        process_dataset(raw_dir, out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
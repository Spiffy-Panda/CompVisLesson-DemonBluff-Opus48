#!/usr/bin/env python3
"""
wiki_harvest.py — fetch-once harvester for the Demon Bluff wiki.

Pulls role pages (villager/minion/outcast/demon) and selected knowledge
categories from demonbluff.wiki.gg via the MediaWiki API, caching each page
exactly once, and downloads the per-role card art.

Design notes (see knowledge-base/KNOWLEDGE-BASE.md):
  * Uses the MediaWiki API (action=parse / action=query), NOT HTML scraping.
  * "Fetch once": a page/image already present in the cache is skipped.
  * Raw verbatim cache  -> knowledge-base/wiki/_raw_cache/   (gitignored)
    Card art            -> knowledge-base/card-art/<class>/  (gitignored)
    Manifest (tracked)  -> knowledge-base/wiki/harvest-manifest.json
  * stdlib only (urllib) so the harvest needs no pip install.

Politeness: descriptive User-Agent, maxlag, and a small delay between calls.
Anchored to the repo root so it runs from any CWD (Rule 1).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# --- repo-root anchoring (Rule 1) ------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
KB = REPO_ROOT / "knowledge-base"
RAW_CACHE = KB / "wiki" / "_raw_cache"
CARD_ART = KB / "card-art"
MANIFEST_PATH = KB / "wiki" / "harvest-manifest.json"

# --- wiki config ------------------------------------------------------------
API = "https://demonbluff.wiki.gg/api.php"
USER_AGENT = "DemonBluffCV-Harvester/0.1 (educational computer-vision course; non-commercial)"
REQUEST_DELAY_S = 0.6          # be polite to wiki.gg
MAXLAG = 5

# role categories -> our role-class label
ROLE_CATEGORIES = {
    "Villagers": "villager",
    "Minions": "minion",
    "Outcasts": "outcast",
    "Demons": "demon",
}
# extra knowledge categories cached for mechanics (no art download)
KNOWLEDGE_CATEGORIES = ["Gameplay", "Relics", "Unused Roles"]

# images we keep as "card art": raster, and either named after the role or
# carrying an art-ish keyword. Everything else (edit icons, license badges,
# wiki chrome) is skipped and recorded so the choice is auditable.
ART_EXTS = (".png", ".jpg", ".jpeg", ".webp")
ART_KEYWORDS = ("card", "token", "icon", "art", "portrait", "role")
SKIP_NAME_HINTS = ("edit", "license", "cc-", "wiki.png", "logo", "favicon",
                   "ambox", "commons", "disambig", "padlock", "question")


def slugify(text: str) -> str:
    text = text.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]", "_", text)


def api_get(params: dict) -> dict:
    params = {**params, "format": "json", "maxlag": MAXLAG}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if "error" in data and data["error"].get("code") == "maxlag":
                wait = 2 * (attempt + 1)
                print(f"    maxlag, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            time.sleep(REQUEST_DELAY_S)
            return data
        except Exception as exc:  # noqa: BLE001 - scrap script, log and back off
            wait = 2 * (attempt + 1)
            print(f"    request failed ({exc}); retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"API call failed after retries: {url}")


def category_members(category: str, namespace: int = 0) -> list[str]:
    titles, cont = [], {}
    while True:
        data = api_get({
            "action": "query", "list": "categorymembers",
            "cmtitle": f"Category:{category}", "cmlimit": "500",
            "cmnamespace": str(namespace), **cont,
        })
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        if "continue" in data:
            cont = data["continue"]
        else:
            break
    return titles


def parse_page(title: str, cache_subdir: Path) -> dict | None:
    """Fetch a page once (wikitext+html+images+categories) and cache the JSON."""
    cache_subdir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_subdir / f"{slugify(title)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    data = api_get({
        "action": "parse", "page": title,
        "prop": "wikitext|text|images|categories|displaytitle",
        "redirects": "1",
    })
    if "error" in data:
        print(f"    !! parse error for {title}: {data['error'].get('info')}",
              file=sys.stderr)
        return None
    parse = data.get("parse", {})
    record = {
        "title": parse.get("title", title),
        "wikitext": parse.get("wikitext", {}).get("*", ""),
        "html": parse.get("text", {}).get("*", ""),
        "images": parse.get("images", []),
        "categories": [c.get("*", "") for c in parse.get("categories", [])],
    }
    cache_file.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return record


def resolve_image_urls(file_titles: list[str]) -> dict[str, str]:
    """File titles (with 'File:' prefix) -> direct URL, batched."""
    urls: dict[str, str] = {}
    for i in range(0, len(file_titles), 40):
        batch = file_titles[i:i + 40]
        data = api_get({
            "action": "query", "prop": "imageinfo",
            "iiprop": "url|size|mime", "titles": "|".join(batch),
        })
        for page in data.get("query", {}).get("pages", {}).values():
            info = page.get("imageinfo")
            if info:
                urls[page["title"]] = info[0]["url"]
    return urls


def is_card_art(filename: str) -> bool:
    low = filename.lower()
    if not low.endswith(ART_EXTS):
        return False
    if any(h in low for h in SKIP_NAME_HINTS):
        return False
    return True  # keep raster images on a role page; keyword filter is a fallback


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        time.sleep(REQUEST_DELAY_S)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    !! download failed {url}: {exc}", file=sys.stderr)
        return False


def main() -> int:
    RAW_CACHE.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"roles": {}, "knowledge": {}, "skipped_images": []}

    # 1) role pages + card art, grouped by class
    for category, role_class in ROLE_CATEGORIES.items():
        print(f"[{role_class}] category members of {category} ...")
        titles = category_members(category)
        print(f"    {len(titles)} pages")
        for title in titles:
            record = parse_page(title, RAW_CACHE / role_class)
            if record is None:
                continue
            role_slug = slugify(title)
            art_dir = CARD_ART / role_class / role_slug
            # candidate images on the page
            file_titles = [f"File:{img}" for img in record["images"]
                           if is_card_art(img)]
            skipped = [img for img in record["images"] if not is_card_art(img)]
            urls = resolve_image_urls(file_titles) if file_titles else {}
            saved = []
            for ftitle, url in urls.items():
                fname = slugify(ftitle.split(":", 1)[-1])
                if download(url, art_dir / fname):
                    saved.append(fname)
            manifest["roles"].setdefault(role_class, {})[title] = {
                "cache": str((RAW_CACHE / role_class /
                              f"{role_slug}.json").relative_to(REPO_ROOT)),
                "art_dir": str(art_dir.relative_to(REPO_ROOT)),
                "art_files": saved,
            }
            manifest["skipped_images"].extend(skipped)
            print(f"    {title}: {len(saved)} art file(s)")

    # 2) knowledge categories (cache pages only, no art)
    for category in KNOWLEDGE_CATEGORIES:
        print(f"[knowledge] category members of {category} ...")
        try:
            titles = category_members(category)
        except Exception as exc:  # noqa: BLE001
            print(f"    skipped {category}: {exc}", file=sys.stderr)
            continue
        subdir = RAW_CACHE / "knowledge" / slugify(category)
        for title in titles:
            record = parse_page(title, subdir)
            if record is None:
                continue
            manifest["knowledge"].setdefault(category, []).append({
                "title": title,
                "cache": str((subdir / f"{slugify(title)}.json")
                             .relative_to(REPO_ROOT)),
            })
        print(f"    {len(titles)} pages cached")

    manifest["skipped_images"] = sorted(set(manifest["skipped_images"]))
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # summary
    n_roles = sum(len(v) for v in manifest["roles"].values())
    n_art = sum(len(r["art_files"])
                for cls in manifest["roles"].values() for r in cls.values())
    n_know = sum(len(v) for v in manifest["knowledge"].values())
    print("\n=== harvest summary ===")
    for cls, roles in manifest["roles"].items():
        print(f"  {cls:9s}: {len(roles)} roles")
    print(f"  knowledge pages: {n_know}")
    print(f"  total roles: {n_roles} | total art files: {n_art}")
    print(f"  manifest: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

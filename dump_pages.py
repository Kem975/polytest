"""
Liquipedia HTML Structure Dumper
Saves raw HTML snippets from key page types so we can identify
the exact selectors needed for match data extraction.

Saves files:
  - dump_team_results.html      (100 Thieves results section)
  - dump_tournament_page.html   (PGL/2025/Bucharest full page)
  - dump_tournament_bracket.html (StarLadder/2025/Major full page)

Run this, then send back the 3 HTML files.
"""

import requests
import time
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://liquipedia.net/counterstrike/api.php"
RATE = 30
OUT = Path(".")

session = requests.Session()
session.headers.update({
    "User-Agent": "DebugDumper/1.0 (debug@debug.com)",
    "Accept-Encoding": "gzip",
})
last = 0.0

def fetch(page: str) -> BeautifulSoup:
    global last
    wait = RATE - (time.time() - last)
    if wait > 0:
        print(f"  [waiting {wait:.0f}s rate limit...]")
        time.sleep(wait)
    print(f"  Fetching: {page}")
    r = session.get(BASE_URL, params={
        "action": "parse", "page": page,
        "prop": "text", "disableeditsection": "1", "format": "json"
    }, timeout=20)
    last = time.time()
    html = r.json().get("parse", {}).get("text", {}).get("*", "")
    return BeautifulSoup(html, "lxml")

def save(filename: str, content: str):
    path = OUT / filename
    path.write_text(content, encoding="utf-8")
    size = len(content) // 1024
    print(f"  ✅ Saved {filename} ({size} KB)")


# ── DUMP 1: 100 Thieves — just the Results section HTML ──────────────────
print("\n[1] Dumping 100 Thieves results section...")
soup = fetch("100_Thieves")

import re
results_h = soup.find(id=re.compile(r"Results", re.I))
if results_h:
    # Grab everything from Results header until next h2
    container = results_h.find_parent("div") or results_h.parent
    # Walk forward siblings until next major heading
    parts = [str(container)]
    sib = container.find_next_sibling()
    for _ in range(10):
        if sib is None:
            break
        if sib.name == "div" and sib.find(re.compile(r"h2")):
            break
        parts.append(str(sib))
        sib = sib.find_next_sibling()
    save("dump_team_results.html", "\n".join(parts))
else:
    # Fallback: save entire page
    print("  ⚠️  No Results header found, saving full page")
    save("dump_team_results.html", str(soup))


# ── DUMP 2: PGL/2025/Bucharest — recent real tournament ──────────────────
print("\n[2] Dumping PGL/2025/Bucharest...")
soup2 = fetch("PGL/2025/Bucharest")
save("dump_tournament_pgl2025.html", str(soup2))


# ── DUMP 3: StarLadder/2025/Major ────────────────────────────────────────
print("\n[3] Dumping StarLadder/2025/Major...")
soup3 = fetch("StarLadder/2025/Major")
save("dump_tournament_starladder2025.html", str(soup3))


# ── QUICK ANALYSIS: print unique class names from each tournament page ────
print("\n── Quick class analysis: PGL/2025/Bucharest ──")
for el in soup2.find_all(class_=True):
    for c in el.get("class", []):
        if any(x in c for x in ["match", "bracket", "group", "score", "result", "team", "versus", "opponent"]):
            print(f"  .{c}")

print("\nAll done! Send back the 3 HTML dump files.")
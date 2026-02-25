"""
Liquipedia Counter-Strike Data Gatherer
========================================
Scrapes match-level data from Liquipedia for ELO / betting analysis.

Data sources (by priority):
  1. Team page → "Recent Matches" tab   → individual match results (best source)
  2. Team page → "Achievements" tab     → tournament placements
  3. Tournament pages                   → brkts-matchlist (Swiss/groups)
                                          brkts-bracket   (playoffs)
                                          brkts-match-info-flat (inline)

HTML structures verified against live page dumps (Feb 2026).

Install:
    pip install requests beautifulsoup4 lxml
"""

import requests
import time
import re
import json
import csv
import os
import logging
from urllib.parse import unquote as url_unquote
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from bs4 import BeautifulSoup, Tag

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MapResult:
    """Per-map round scores from a match popup."""
    map_name: str           # "Ancient", "Inferno", "Anubis", ...
    team1_score: int        # total rounds won by team1 on this map
    team2_score: int
    team1_t_rounds: int = 0   # team1's T-side rounds
    team1_ct_rounds: int = 0  # team1's CT-side rounds
    team2_t_rounds: int = 0
    team2_ct_rounds: int = 0
    winner: str = ""          # team name that won this map

    @property
    def is_overtime(self) -> bool:
        return (self.team1_score + self.team2_score) > 24


@dataclass
class Match:
    """A single match between two teams with score and optional map details."""
    date: str               # YYYY-MM-DD
    timestamp: int          # unix epoch (0 if unknown)
    team1: str              # full canonical team name
    team2: str
    score1: int             # maps won by team1
    score2: int             # maps won by team2
    winner: str             # full team name of winner ("" if draw/unknown)
    tournament: str         # display name
    tournament_page: str    # wiki slug
    tier: str = ""          # S, A, B, C, D
    match_type: str = ""    # Online / Offline
    best_of: str = ""       # Bo1, Bo3, Bo5
    stage: str = ""         # Round 1, Playoffs, etc.
    maps: list = field(default_factory=list)  # list[MapResult]

    def involves(self, team: str) -> bool:
        t = team.strip().lower()
        return self.team1.strip().lower() == t or self.team2.strip().lower() == t

    def did_win(self, team: str) -> Optional[bool]:
        t = team.strip().lower()
        if not self.involves(team):
            return None
        return self.winner.strip().lower() == t


@dataclass
class TournamentResult:
    """A team's placement in a tournament (from Achievements tab)."""
    date: str
    placement: str          # "1st", "2nd", "5th - 6th"
    tier: str               # S, A, B, C, D
    event_type: str         # Online / Offline
    tournament: str
    tournament_page: str
    result: str             # "1 : 3" or "2/3" (group stage record)
    opponent: str           # final opponent (if available)
    prize: str


@dataclass
class RosterPlayer:
    gamertag: str
    full_name: str
    country: str
    join_date: str


@dataclass
class UpcomingMatch:
    tournament_name: str
    tournament_page: str
    date_range: str
    timestamp: int

    def date(self) -> str:
        if self.timestamp:
            return datetime.fromtimestamp(self.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")
        return "?"


@dataclass
class HeadToHead:
    team1: str
    team2: str
    team1_wins: int = 0
    team2_wins: int = 0
    matches: list = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.team1_wins + self.team2_wins

    def winrate(self, team: str) -> float:
        if self.total == 0:
            return 0.0
        wins = self.team1_wins if team.strip().lower() == self.team1.strip().lower() else self.team2_wins
        return round(wins / self.total, 4)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _extract_map_scores(table: Tag) -> tuple[int, int, int]:
    """
    Extract (total_score, t_rounds, ct_rounds) from one side's score table
    inside a brkts-popup-body-game element.

    Structure:
      <table>
        <tr><td rowspan="2">13</td><td class="...score-color-t">6</td></tr>
        <tr><td class="...score-color-ct">7</td></tr>
        (optional OT rows)
      </table>
    """
    total, t_rounds, ct_rounds = 0, 0, 0

    rows = table.find_all("tr")
    if not rows:
        return total, t_rounds, ct_rounds

    first_row_tds = rows[0].find_all("td")
    if first_row_tds:
        try:
            total = int(first_row_tds[0].get_text(strip=True))
        except (ValueError, TypeError):
            pass

    for td in table.find_all("td", class_=True):
        cls = " ".join(td.get("class", []))
        try:
            val = int(td.get_text(strip=True))
        except (ValueError, TypeError):
            continue
        if "score-color-t" in cls and "score-color-ct" not in cls:
            t_rounds += val
        elif "score-color-ct" in cls:
            ct_rounds += val

    return total, t_rounds, ct_rounds


class LiquipediaCSClient:
    BASE_URL = "https://liquipedia.net/counterstrike/api.php"
    RATE_LIMIT_SECONDS = 30

    TIER_CATEGORIES = {
        "S": "S-Tier_Tournaments",
        "A": "A-Tier_Tournaments",
        "B": "B-Tier_Tournaments",
        "C": "C-Tier_Tournaments",
        "D": "D-Tier_Tournaments",
    }
    TIER_PAGES = {
        "S": "S-Tier_Tournaments",
        "A": "A-Tier_Tournaments",
        "B": "B-Tier_Tournaments",
        "C": "C-Tier_Tournaments",
        "Q": "Qualifier_Tournaments",
    }
    TIER_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}

    def __init__(self, app_name: str = "LiquipediaCSClient/1.0",
                 rate_limit: float = RATE_LIMIT_SECONDS):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": app_name,
            "Accept-Encoding": "gzip",
        })
        self.rate_limit = rate_limit
        self._last: float = 0.0

    # ------------------------------------------------------------------ #
    # Core HTTP helpers                                                    #
    # ------------------------------------------------------------------ #

    def _wait(self):
        wait = self.rate_limit - (time.time() - self._last)
        if wait > 0:
            logger.info(f"Rate limit: waiting {wait:.1f}s ...")
            time.sleep(wait)

    def _get(self, params: dict) -> dict:
        self._wait()
        params.setdefault("format", "json")
        resp = self.session.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        self._last = time.time()
        return resp.json()

    def _fetch_html(self, page: str) -> BeautifulSoup:
        logger.info(f"Fetching: {page}")
        data = self._get({
            "action": "parse",
            "page": page,
            "prop": "text",
            "disableeditsection": "1",
        })
        html = data.get("parse", {}).get("text", {}).get("*", "")
        return BeautifulSoup(html, "lxml")

    def _category_members(self, category: str, limit: int = 500,
                           sort: str = "timestamp", direction: str = "desc") -> list[dict]:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category}",
            "cmlimit": min(limit, 500),
            "cmtype": "page",
            "cmsort": sort,
            "cmdir": direction,
        }
        return self._get(params).get("query", {}).get("categorymembers", [])

    # ------------------------------------------------------------------ #
    # Shared helpers                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_tier(cell: Tag) -> str:
        """
        Extract tier letter from a tier cell.
        Handles: link text 'S-Tier'/'A-Tier', data-sort-value 'A1'/'A2',
        and link href containing tier name.
        """
        link = cell.find("a")
        if link:
            href = link.get("href", "")
            if "S-Tier" in href:
                return "S"
            if "A-Tier" in href:
                return "A"
            if "B-Tier" in href:
                return "B"
            if "C-Tier" in href:
                return "C"
            if "D-Tier" in href:
                return "D"

        text = cell.get_text(strip=True)
        m = re.match(r"([SABCD])-?Tier", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

        dsv = cell.get("data-sort-value", "")
        tier_map = {"A1": "S", "A2": "A", "A3": "B", "A4": "C", "A5": "D"}
        if dsv in tier_map:
            return tier_map[dsv]

        return text[:1].upper() if text else "?"

    @staticmethod
    def _extract_team_from_block(el: Tag) -> str:
        """
        Extract full canonical team name from a block-team element.
        Prefers the link title attribute (e.g. 'Ninjas in Pyjamas' not 'NIP').
        """
        if not el:
            return ""
        name_span = el.find(class_="name")
        if name_span:
            a = name_span.find("a")
            if a:
                return a.get("title", a.get_text(strip=True))
            return name_span.get_text(strip=True)
        for a in el.find_all("a"):
            title = a.get("title", "")
            if title and not title.endswith((".png", ".svg", ".jpg")):
                return title
        return ""

    @staticmethod
    def _extract_team_from_header_opponent(opp_div: Tag) -> str:
        """Extract team name from a match-info-header-opponent element."""
        name_span = opp_div.find(class_="name")
        if name_span:
            a = name_span.find("a")
            if a:
                return a.get("title", a.get_text(strip=True))
            return name_span.get_text(strip=True)
        for a in opp_div.find_all("a"):
            title = a.get("title", "")
            if title and not title.endswith((".png", ".svg", ".jpg")):
                return title
        return ""

    @staticmethod
    def _ts_to_date(ts: int) -> str:
        if ts:
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return ""

    # ================================================================== #
    # 1. TEAM RECENT MATCHES                                              #
    #    Primary data source for individual match results.                 #
    #    Structure: content2 tab on team page → wikitable with            #
    #    match-table-score cells.                                         #
    #    Cols: Date|Tier|Type|Game|TournIcon|Tournament|Score|Opp|VODs    #
    # ================================================================== #

    def get_team_matches(self, team_name: str) -> list[Match]:
        """
        Get individual match results from a team page's 'Recent Matches' tab.
        Returns Match objects with opponent, score, date, tier, tournament.
        """
        soup = self._fetch_html(team_name.replace(" ", "_"))
        return self._parse_recent_matches_table(soup, team_name)

    def get_team_matches_extended(self, team_name: str) -> list[Match]:
        """
        Fetch the {Team}/Results page and parse the recent matches table there.
        Contains more history than the main team page.
        """
        page = team_name.replace(" ", "_") + "/Results"
        try:
            soup = self._fetch_html(page)
            return self._parse_recent_matches_table(soup, team_name)
        except Exception as e:
            logger.warning(f"Could not fetch extended results for '{team_name}': {e}")
            return []

    def _parse_recent_matches_table(self, soup: BeautifulSoup, team_name: str) -> list[Match]:
        """Parse any wikitable that contains match-table-score cells."""
        matches = []

        for table in soup.find_all("table", class_=lambda c: c and "wikitable" in c):
            if not table.find("td", class_="match-table-score"):
                continue

            for row in table.find_all("tr"):
                cells = row.find_all("td")
                if len(cells) < 7:
                    continue

                score_cell = row.find("td", class_="match-table-score")
                if not score_cell:
                    continue

                # -- Date (cell 0) --
                timer = cells[0].find(class_="timer-object")
                ts = int(timer.get("data-timestamp", "0")) if timer else 0
                date_str = self._ts_to_date(ts)

                # -- Tier (cell 1) --
                tier = self._extract_tier(cells[1])

                # -- Type (cell 2) --
                match_type = cells[2].get_text(strip=True)

                # -- Tournament (cells 4-5 area, find the link with text) --
                tourn_name, tourn_page = "", ""
                for cell in cells[4:7]:
                    a = cell.find("a")
                    if a and len(a.get_text(strip=True)) > 2:
                        tourn_name = a.get_text(strip=True)
                        href = a.get("href", "")
                        tourn_page = href.replace("/counterstrike/", "").split("#")[0].strip("/")
                        break

                # -- Score --
                score_text = score_cell.get_text(strip=True)
                parts = re.split(r"\s*[-:]\s*", score_text)
                if len(parts) != 2:
                    continue
                try:
                    left_score = int(parts[0].strip())
                    right_score = int(parts[1].strip())
                except ValueError:
                    continue

                # -- Opponent --
                opponent = ""
                for cell in cells:
                    block = cell.find(class_="block-team")
                    if block:
                        opponent = self._extract_team_from_block(block)
                        break

                if not opponent:
                    continue

                # -- Winner: row class is definitive --
                row_cls = " ".join(row.get("class", []))
                if "recent-matches-bg-win" in row_cls:
                    winner = team_name
                elif "recent-matches-bg-lose" in row_cls:
                    winner = opponent
                else:
                    winner = team_name if left_score > right_score else (
                        opponent if right_score > left_score else "")

                matches.append(Match(
                    date=date_str, timestamp=ts,
                    team1=team_name, team2=opponent,
                    score1=left_score, score2=right_score,
                    winner=winner,
                    tournament=tourn_name, tournament_page=tourn_page,
                    tier=tier, match_type=match_type,
                ))

        logger.info(f"Parsed {len(matches)} recent matches for '{team_name}'")
        return matches

    # ================================================================== #
    # 2. TEAM ACHIEVEMENTS                                                #
    #    Tournament placement history from Achievements tab.              #
    #    Cols: Date|Place|Tier|Type|Game|TournIcon|TournName|Score|Opp|$  #
    # ================================================================== #

    def get_team_results(self, team_name: str, limit: int = 100,
                         min_tier: Optional[str] = None) -> list[TournamentResult]:
        soup = self._fetch_html(team_name.replace(" ", "_"))

        min_rank = self.TIER_ORDER.get(min_tier, 99) if min_tier else 99
        results = []

        for table in soup.find_all("table", class_=lambda c: c and "wikitable" in c and "sortable" in c):
            if table.find("td", class_="match-table-score"):
                continue

            for row in table.find_all("tr"):
                if row.find("th"):
                    continue
                if "display:none" in row.get("style", ""):
                    continue

                cells = row.find_all("td")
                if len(cells) < 8:
                    continue

                date = cells[0].get_text(strip=True)
                placement = cells[1].get_text(strip=True)
                tier = self._extract_tier(cells[2])
                evt_type = cells[3].get_text(strip=True)

                # cells[5]=icon, cells[6]=name+link
                tourn_name, tourn_page = "", ""
                for cell in cells[5:8]:
                    a = cell.find("a")
                    if a and len(a.get_text(strip=True)) > 3:
                        tourn_name = a.get_text(strip=True)
                        href = a.get("href", "")
                        tourn_page = href.replace("/counterstrike/", "").split("#")[0].strip("/")
                        break

                result_text = cells[7].get_text(strip=True) if len(cells) > 7 else ""

                opponent = ""
                if len(cells) > 8:
                    block = cells[8].find(class_="block-team")
                    if block:
                        opponent = self._extract_team_from_block(block)
                    else:
                        opponent = cells[8].get_text(strip=True)

                prize = cells[-1].get_text(strip=True) if len(cells) > 9 else ""

                if self.TIER_ORDER.get(tier, 99) > min_rank:
                    continue
                if not date or not placement:
                    continue

                results.append(TournamentResult(
                    date=date, placement=placement, tier=tier,
                    event_type=evt_type, tournament=tourn_name,
                    tournament_page=tourn_page, result=result_text,
                    opponent=opponent, prize=prize,
                ))

        results.sort(key=lambda r: r.date, reverse=True)
        logger.info(f"Found {len(results)} tournament results for '{team_name}'")
        return results[:limit]

    # ================================================================== #
    # 3. TOURNAMENT MATCHES                                               #
    #    Three formats found on tournament pages:                         #
    #    A) brkts-matchlist  — Swiss / group stage match lists            #
    #    B) brkts-bracket    — playoff bracket trees                      #
    #    C) brkts-match-info-flat — inline flat match results             #
    # ================================================================== #

    def get_tournament_matches(self, tournament_page: str,
                                soup: BeautifulSoup = None) -> list[Match]:
        if soup is None:
            soup = self._fetch_html(tournament_page)
        matches: list[Match] = []

        matches.extend(self._parse_matchlists(soup, tournament_page))
        matches.extend(self._parse_brackets(soup, tournament_page))
        matches.extend(self._parse_flat_popups(soup, tournament_page))

        seen: set[tuple] = set()
        unique: list[Match] = []
        for m in matches:
            key = (m.team1.lower(), m.team2.lower(), m.score1, m.score2, m.timestamp or m.date)
            if key not in seen:
                seen.add(key)
                unique.append(m)

        unique.sort(key=lambda m: m.timestamp or 0, reverse=True)
        logger.info(f"Found {len(unique)} matches on '{tournament_page}'")
        return unique

    # --- 3A: brkts-matchlist (Swiss / group stage) ---

    def _parse_matchlists(self, soup: BeautifulSoup, tourn_page: str) -> list[Match]:
        matches = []
        for matchlist in soup.find_all(class_="brkts-matchlist"):
            stage = ""
            title_el = matchlist.find(class_="general-collapsible-default-title")
            if title_el:
                stage = title_el.get_text(strip=True)

            collapse = matchlist.find(class_="brkts-matchlist-collapse-area")
            container = collapse if collapse else matchlist

            current_date = ""
            for child in container.children:
                if not isinstance(child, Tag):
                    continue
                classes = child.get("class", [])

                if "brkts-matchlist-header" in classes:
                    current_date = child.get_text(strip=True)
                    continue

                if "brkts-matchlist-match" in classes:
                    m = self._parse_matchlist_entry(child, current_date, stage, tourn_page)
                    if m:
                        matches.append(m)

        return matches

    def _parse_matchlist_entry(self, match_el: Tag, header_date: str,
                                stage: str, tourn_page: str) -> Optional[Match]:
        opponents = match_el.find_all(class_="brkts-matchlist-opponent")
        scores = match_el.find_all(class_="brkts-matchlist-score")
        if len(opponents) < 2 or len(scores) < 2:
            return None

        team1 = opponents[0].get("aria-label", "").strip()
        team2 = opponents[1].get("aria-label", "").strip()

        s1_text = scores[0].get_text(strip=True)
        s2_text = scores[1].get_text(strip=True)
        try:
            score1 = int(s1_text)
            score2 = int(s2_text)
        except (ValueError, TypeError):
            return None

        winner = ""
        opp1_cls = " ".join(opponents[0].get("class", []))
        opp2_cls = " ".join(opponents[1].get("class", []))
        if "brkts-matchlist-slot-winner" in opp1_cls:
            winner = team1
        elif "brkts-matchlist-slot-winner" in opp2_cls:
            winner = team2
        else:
            winner = team1 if score1 > score2 else (team2 if score2 > score1 else "")

        date_str, ts, best_of = header_date, 0, ""
        map_results = []
        popup = match_el.find(class_="brkts-match-info-popup")
        if popup:
            ts, date_str, best_of = self._extract_popup_meta(popup, date_str)
            map_results = self._parse_popup_maps(popup, team1, team2)

        if not team1 or not team2:
            return None

        tourn_display = tourn_page.replace("_", " ")

        return Match(
            date=date_str, timestamp=ts,
            team1=team1, team2=team2,
            score1=score1, score2=score2, winner=winner,
            tournament=tourn_display, tournament_page=tourn_page,
            best_of=best_of, stage=stage,
            maps=map_results,
        )

    # --- 3B: brkts-bracket (playoffs) ---

    def _parse_brackets(self, soup: BeautifulSoup, tourn_page: str) -> list[Match]:
        matches = []
        for bracket in soup.find_all(class_="brkts-bracket-wrapper"):
            for popup in bracket.find_all(class_="brkts-popup"):
                m = self._parse_match_popup(popup, tourn_page, stage="Playoffs")
                if m:
                    matches.append(m)
        return matches

    # --- 3C: brkts-match-info-flat (inline) ---

    def _parse_flat_popups(self, soup: BeautifulSoup, tourn_page: str) -> list[Match]:
        matches = []
        for popup in soup.find_all(class_="brkts-match-info-flat"):
            m = self._parse_match_popup(popup, tourn_page)
            if m:
                matches.append(m)
        return matches

    # --- Shared popup parser ---

    def _parse_match_popup(self, popup: Tag, tourn_page: str,
                            stage: str = "") -> Optional[Match]:
        """Parse a brkts-popup match info element (shared by brackets, flat, matchlists)."""
        header = popup.find(class_="match-info-header")
        if not header:
            return None

        opp_divs = header.find_all(class_=re.compile(r"match-info-header-opponent"))
        if len(opp_divs) < 2:
            return None

        team1 = self._extract_team_from_header_opponent(opp_divs[0])
        team2 = self._extract_team_from_header_opponent(opp_divs[1])
        if not team1 or not team2:
            return None

        score_els = header.find_all(class_="match-info-header-scoreholder-score")
        if len(score_els) < 2:
            return None
        try:
            score1 = int(score_els[0].get_text(strip=True))
            score2 = int(score_els[1].get_text(strip=True))
        except (ValueError, TypeError):
            return None

        winner = ""
        opp1_cls = " ".join(opp_divs[0].get("class", []))
        opp2_cls = " ".join(opp_divs[1].get("class", []))
        if "match-info-header-winner" in opp1_cls:
            winner = team1
        elif "match-info-header-winner" in opp2_cls:
            winner = team2
        else:
            winner = team1 if score1 > score2 else (team2 if score2 > score1 else "")

        ts, date_str, best_of = self._extract_popup_meta(popup, "")
        map_results = self._parse_popup_maps(popup, team1, team2)
        tourn_display = tourn_page.replace("_", " ")

        return Match(
            date=date_str, timestamp=ts,
            team1=team1, team2=team2,
            score1=score1, score2=score2, winner=winner,
            tournament=tourn_display, tournament_page=tourn_page,
            best_of=best_of, stage=stage,
            maps=map_results,
        )

    @staticmethod
    def _extract_popup_meta(popup: Tag, fallback_date: str) -> tuple[int, str, str]:
        """Extract timestamp, date string, and best-of from a match popup."""
        ts = 0
        date_str = fallback_date
        best_of = ""

        timer = popup.find(class_="timer-object")
        if timer:
            try:
                ts = int(timer.get("data-timestamp", "0"))
            except ValueError:
                ts = 0
            if ts:
                date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

        lower = popup.find(class_="match-info-header-scoreholder-lower")
        if lower:
            bo_match = re.search(r"Bo(\d)", lower.get_text(strip=True), re.IGNORECASE)
            if bo_match:
                best_of = f"Bo{bo_match.group(1)}"

        return ts, date_str, best_of

    @staticmethod
    def _parse_popup_maps(popup: Tag, team1: str, team2: str) -> list[MapResult]:
        """
        Extract per-map results from a match popup body.

        Each brkts-popup-body-game contains:
          - Left table (team1): total score + T/CT half scores
          - Center: map name link
          - Right table (team2): total score + T/CT half scores
          - Win/loss icons on each side
        """
        maps = []
        body = popup.find(class_="brkts-popup-body")
        if not body:
            return maps

        for game in body.find_all(class_="brkts-popup-body-game"):
            tables = game.find_all("table")
            if len(tables) < 2:
                continue

            # --- Map name (center link) ---
            map_link = game.find("a", href=lambda h: h and "/counterstrike/" in h)
            map_name = map_link.get_text(strip=True) if map_link else "?"

            # --- Team 1 scores (left table, direction:ltr) ---
            t1_total, t1_t, t1_ct = _extract_map_scores(tables[0])

            # --- Team 2 scores (right table, direction:rtl) ---
            t2_total, t2_t, t2_ct = _extract_map_scores(tables[1])

            # --- Winner: check win/loss icons ---
            icons = game.find_all(class_="brkts-popup-winloss-icon")
            map_winner = ""
            if len(icons) >= 2:
                left_icon = icons[0].find("i")
                if left_icon:
                    icon_cls = " ".join(left_icon.get("class", []))
                    if "fa-check" in icon_cls:
                        map_winner = team1
                    elif "fa-times" in icon_cls:
                        map_winner = team2
            if not map_winner and t1_total != t2_total:
                map_winner = team1 if t1_total > t2_total else team2

            if t1_total == 0 and t2_total == 0:
                continue

            maps.append(MapResult(
                map_name=map_name,
                team1_score=t1_total, team2_score=t2_total,
                team1_t_rounds=t1_t, team1_ct_rounds=t1_ct,
                team2_t_rounds=t2_t, team2_ct_rounds=t2_ct,
                winner=map_winner,
            ))

        return maps

    # ================================================================== #
    # 4. ROSTER                                                            #
    # ================================================================== #

    def get_team_roster(self, team_name: str) -> list[RosterPlayer]:
        soup = self._fetch_html(team_name.replace(" ", "_"))
        for div in soup.find_all(class_="table2"):
            table = div.find("table")
            if not table:
                continue
            header_row = table.find("tr", class_="table2__row--head")
            if not header_row:
                continue
            headers = [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]
            if "id" not in headers or "name" not in headers:
                continue
            if "leave date" in headers or "inactive date" in headers:
                continue

            players = []
            for row in table.find_all("tr", class_=re.compile(r"table2__row--body")):
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue
                gamertag = re.sub(r"\[\d+\]", "", cells[0].get_text(strip=True))
                name_el = cells[1].find("a") or cells[1]
                full_name = re.sub(r"\[\d+\]", "", name_el.get_text(strip=True))
                flag = cells[1].find("img", alt=True)
                country = flag["alt"] if flag else ""
                join_date = re.sub(r"\[\d+\]", "", cells[-1].get_text(strip=True))
                if gamertag:
                    players.append(RosterPlayer(gamertag, full_name, country, join_date))
            if players:
                return players
        return []

    # ================================================================== #
    # 5. UPCOMING                                                          #
    # ================================================================== #

    def get_team_upcoming(self, team_name: str) -> list[UpcomingMatch]:
        soup = self._fetch_html(team_name.replace(" ", "_"))
        table = soup.find("table", class_="infobox_matches_content")
        if not table:
            return []

        results = []
        rows = table.find_all("tr")
        i = 0
        while i < len(rows):
            row = rows[i]
            versus_td = row.find("td", class_="versus")
            if not versus_td:
                i += 1
                continue

            link = versus_td.find("a")
            if not link:
                i += 1
                continue
            name = link.get_text(strip=True)
            page = link.get("href", "").replace("/counterstrike/", "").strip("/")

            date_range, ts = "", 0
            if i + 1 < len(rows):
                filler = rows[i + 1].find("td", class_="match-filler")
                if filler:
                    span = filler.find(class_="tournament-span")
                    date_range = span.get_text(strip=True) if span else ""
                    timer = filler.find(class_=re.compile(r"timer-object"))
                    if timer:
                        try:
                            ts = int(timer.get("data-timestamp", "0"))
                        except ValueError:
                            ts = 0

            results.append(UpcomingMatch(name, page, date_range, ts))
            i += 2

        logger.info(f"Found {len(results)} upcoming for '{team_name}'")
        return results

    # ================================================================== #
    # 6. TEAM / TOURNAMENT DISCOVERY                                       #
    # ================================================================== #

    def get_teams(self, category: str = "Teams", limit: int = 200) -> list[dict]:
        members = self._category_members(category, limit=limit, sort="sortkey", direction="asc")
        return [{"name": m["title"], "page": m["title"].replace(" ", "_")} for m in members]

    def search_team(self, query: str, limit: int = 10) -> list[str]:
        data = self._get({"action": "query", "list": "prefixsearch",
                          "pssearch": query, "pslimit": limit})
        return [r["title"] for r in data.get("query", {}).get("prefixsearch", [])]

    def get_tournaments(self, tiers: list[str] = None,
                        limit: int = 50) -> list[dict]:
        if tiers is None:
            tiers = ["S", "A"]
        all_t = []
        for tier in tiers:
            cat = self.TIER_CATEGORIES.get(tier.upper())
            if not cat:
                continue
            for m in self._category_members(cat, limit=limit):
                if not m["title"].startswith("User:"):
                    all_t.append({
                        "name": m["title"],
                        "page": m["title"].replace(" ", "_"),
                        "tier": tier.upper(),
                    })
        logger.info(f"Found {len(all_t)} tournaments for tiers {tiers}")
        return all_t

    # ================================================================== #
    # 6B. PORTAL-BASED TOURNAMENT DISCOVERY                                #
    # ================================================================== #

    @staticmethod
    def _parse_portal_start_date(date_str: str):
        """Parse start date from portal date string like 'Feb 14 - 22, 2026'."""
        date_str = date_str.strip().replace("\xa0", " ")
        year_match = re.search(r"(\d{4})", date_str)
        if not year_match:
            return None
        year = year_match.group(1)
        start_match = re.match(r"([A-Z][a-z]{2})\s+(\d{1,2})", date_str)
        if not start_match:
            return None
        try:
            return datetime.strptime(
                f"{start_match.group(1)} {start_match.group(2)}, {year}", "%b %d, %Y"
            ).date()
        except ValueError:
            return None

    @staticmethod
    def _find_tournament_link(cell: Tag) -> tuple[str, str]:
        """Extract (name, page_slug) from a portal tournament cell.
        Prefers bold links (specific tournament) over series icon links.
        URL-decodes the slug so MediaWiki API receives the real title."""
        for a in cell.find_all("a"):
            if a.find("b"):
                href = a.get("href", "")
                if href.startswith("/counterstrike/"):
                    page = url_unquote(
                        href.replace("/counterstrike/", "").split("#")[0].strip("/"))
                    return a.get_text(strip=True), page
        best_name, best_page = "", ""
        for a in cell.find_all("a"):
            href = a.get("href", "")
            text = a.get_text(strip=True)
            if (href.startswith("/counterstrike/")
                    and len(text) > len(best_name)
                    and not href.endswith((".png", ".svg", ".jpg"))
                    and "Category:" not in href):
                best_page = url_unquote(
                    href.replace("/counterstrike/", "").split("#")[0].strip("/"))
                best_name = text
        return best_name, best_page

    def _parse_grid_tournaments(self, soup: BeautifulSoup, tier: str,
                                 cutoff, today) -> list[dict]:
        """Extract tournaments from gridTable elements on a tier page."""
        results: list[dict] = []
        for grid in soup.find_all("div", class_="gridTable"):
            header = grid.find("div", class_="gridHeader")
            if not header:
                continue
            hdr_cells = header.find_all("div", class_="gridCell")
            hdr_texts = [c.get_text(strip=True).lower() for c in hdr_cells]

            tourn_idx = next((i for i, h in enumerate(hdr_texts)
                              if "tournament" in h), None)
            date_idx = next((i for i, h in enumerate(hdr_texts)
                             if "date" in h), None)
            if tourn_idx is None or date_idx is None:
                continue

            for row in grid.find_all("div", class_="gridRow"):
                cells = row.find_all("div", class_="gridCell", recursive=False)
                if len(cells) <= max(tourn_idx, date_idx):
                    continue

                tourn_name, tourn_page = self._find_tournament_link(
                    cells[tourn_idx])
                if not tourn_page:
                    continue

                date_str = cells[date_idx].get_text(strip=True)
                start_date = self._parse_portal_start_date(date_str)
                if start_date is None:
                    continue
                if start_date < cutoff or start_date > today:
                    continue

                results.append({
                    "name": tourn_name,
                    "page": tourn_page,
                    "tier": tier,
                    "start_date": start_date.isoformat(),
                    "date_str": date_str,
                })
        return results

    def get_portal_tournaments(self, tiers: list[str] = None,
                                cutoff_date: str = "2025-11-23") -> list[dict]:
        """Fetch dedicated tier pages and extract tournaments within the date
        window [cutoff_date .. today].  One API call per tier.
        Returns list of dicts: {name, page, tier, start_date, date_str}.
        """
        from datetime import date as date_type

        if tiers is None:
            tiers = ["S", "A", "B", "C"]

        today = date_type.today()
        cutoff = date_type.fromisoformat(cutoff_date)

        seen: set[str] = set()
        tournaments: list[dict] = []

        for tier in tiers:
            page_name = self.TIER_PAGES.get(tier.upper())
            if not page_name:
                continue
            soup = self._fetch_html(page_name)
            page_results = self._parse_grid_tournaments(
                soup, tier.upper(), cutoff, today)
            for t in page_results:
                if t["page"] not in seen:
                    seen.add(t["page"])
                    tournaments.append(t)
            logger.info(f"  {tier}-tier page: {len(page_results)} tournaments "
                         f"in [{cutoff_date} .. {today}]")

        tournaments.sort(key=lambda t: t.get("start_date", ""), reverse=True)
        logger.info(f"Total: {len(tournaments)} tournaments across "
                     f"tiers {tiers}")
        return tournaments

    # ================================================================== #
    # 6C. TOURNAMENT LINEUPS (teamcards)                                   #
    # ================================================================== #

    @staticmethod
    def get_tournament_lineups(soup: BeautifulSoup) -> list[dict]:
        """Parse teamcard elements from a tournament page.
        Returns list of dicts: {team, player_number, player_gamertag,
                                player_country, is_coach}.
        """
        lineups: list[dict] = []
        for tc in soup.find_all(class_="teamcard"):
            center = tc.find("center")
            if not center:
                continue
            team_link = center.find("a")
            if not team_link:
                continue
            team_name = team_link.get_text(strip=True)

            table = tc.find("table", class_=lambda c: c and "active" in c)
            if not table:
                table = tc.find("table")
            if not table:
                continue

            for row in table.find_all("tr"):
                th = row.find("th")
                td = row.find("td")
                if not th or not td:
                    continue

                th_text = th.get_text(strip=True)
                is_coach = bool(th.find("abbr", title=lambda t: t and "coach" in t.lower()))
                player_num = th_text if th_text.isdigit() else ("C" if is_coach else "")
                if not player_num:
                    continue

                player_link = td.find(
                    "a", href=lambda h: (h and "/counterstrike/" in h
                                         and "Category:" not in h))
                gamertag = (player_link.get_text(strip=True)
                            if player_link else "")
                flag_img = td.find("img", alt=True)
                country = flag_img["alt"] if flag_img else ""

                if gamertag:
                    lineups.append({
                        "tournament": "",
                        "tournament_page": "",
                        "team": team_name,
                        "player_number": player_num,
                        "player_gamertag": gamertag,
                        "player_country": country,
                        "is_coach": is_coach,
                    })

        return lineups

    # ================================================================== #
    # 7. HEAD-TO-HEAD                                                      #
    #    Efficient: uses team match histories instead of scanning          #
    #    tournament pages one by one.                                      #
    # ================================================================== #

    def get_head_to_head(self, team1: str, team2: str,
                          tournaments: list[dict] = None,
                          max_tournaments: int = 10) -> HeadToHead:
        """
        Build H2H record. Two strategies:
        1. If tournaments are provided, scan each tournament's match page.
        2. Otherwise, fetch team1's recent matches and filter for team2.
        """
        h2h = HeadToHead(team1=team1, team2=team2)
        t1l, t2l = team1.strip().lower(), team2.strip().lower()

        if tournaments:
            count = 0
            for t in tournaments:
                if count >= max_tournaments:
                    break
                page = t.get("page") or t.get("tournament_page", "")
                if not page:
                    continue
                count += 1
                try:
                    for m in self.get_tournament_matches(page):
                        pair = {m.team1.strip().lower(), m.team2.strip().lower()}
                        if pair == {t1l, t2l}:
                            h2h.matches.append(m)
                            w = m.winner.strip().lower()
                            if w == t1l:
                                h2h.team1_wins += 1
                            elif w == t2l:
                                h2h.team2_wins += 1
                except Exception as e:
                    logger.warning(f"  Failed '{page}': {e}")
        else:
            matches = self.get_team_matches(team1)
            for m in matches:
                if m.team2.strip().lower() == t2l:
                    h2h.matches.append(m)
                    w = m.winner.strip().lower()
                    if w == t1l:
                        h2h.team1_wins += 1
                    elif w == t2l:
                        h2h.team2_wins += 1

        h2h.matches.sort(key=lambda m: m.timestamp or 0, reverse=True)
        logger.info(f"H2H: {team1} {h2h.team1_wins}-{h2h.team2_wins} {team2} ({h2h.total} matches)")
        return h2h


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def print_matches(matches: list[Match], title: str = "Matches", limit: int = 50,
                   show_maps: bool = True):
    print(f"\n{'='*80}\n  {title} ({len(matches)})\n{'='*80}")
    if not matches:
        print("  None found.")
        return
    for m in matches[:limit]:
        bo = f" [{m.best_of}]" if m.best_of else ""
        tier = f"[{m.tier}]" if m.tier else ""
        w_marker = ""
        if m.winner:
            w_marker = f"  -> {m.winner}"
        print(f"  {m.date:10s}  {tier:4s} {m.team1:22s} {m.score1}-{m.score2}  "
              f"{m.team2:22s}{bo}{w_marker}")
        if m.tournament:
            print(f"{'':14s}  {m.tournament}")
        if show_maps and m.maps:
            for mp in m.maps:
                ot = " (OT)" if mp.is_overtime else ""
                sides1 = f"({mp.team1_t_rounds}t/{mp.team1_ct_rounds}ct)" if (mp.team1_t_rounds or mp.team1_ct_rounds) else ""
                sides2 = f"({mp.team2_t_rounds}t/{mp.team2_ct_rounds}ct)" if (mp.team2_t_rounds or mp.team2_ct_rounds) else ""
                print(f"{'':16s}{mp.map_name:12s}  "
                      f"{mp.team1_score:2d}{sides1} - {mp.team2_score:2d}{sides2}{ot}")


def print_results(results: list[TournamentResult], title: str = "Results"):
    print(f"\n{'='*80}\n  {title} ({len(results)})\n{'='*80}")
    if not results:
        print("  None found.")
        return
    for r in results:
        opp = f" vs {r.opponent}" if r.opponent else ""
        prize = f"  {r.prize}" if r.prize and "$" in r.prize else ""
        print(f"  {r.date}  {r.placement:12s} [{r.tier}] {r.event_type:8s}  "
              f"{r.tournament[:40]}{opp}{prize}")


def print_h2h(h2h: HeadToHead):
    print(f"\n{'='*80}\n  H2H: {h2h.team1} vs {h2h.team2}\n{'='*80}")
    print(f"  {h2h.team1}: {h2h.team1_wins} wins ({h2h.winrate(h2h.team1)*100:.1f}%)")
    print(f"  {h2h.team2}: {h2h.team2_wins} wins ({h2h.winrate(h2h.team2)*100:.1f}%)")
    print(f"  Total: {h2h.total}")
    print_matches(h2h.matches, "H2H Match History")


def print_roster(roster: list[RosterPlayer], team: str):
    print(f"\n{'='*80}\n  Roster: {team}\n{'='*80}")
    print(f"  {'Gamertag':20s} {'Full Name':22s} {'Country':15s} Joined")
    print(f"  {'-'*70}")
    for p in roster:
        print(f"  {p.gamertag:20s} {p.full_name:22s} {p.country:15s} {p.join_date}")


def print_upcoming(items: list[UpcomingMatch], title: str = "Upcoming"):
    print(f"\n{'='*80}\n  {title} ({len(items)})\n{'='*80}")
    for t in items:
        print(f"  {t.date():10s}  {t.date_range:20s}  {t.tournament_name}")
        print(f"{'':14s}  -> page: {t.tournament_page}")


# ---------------------------------------------------------------------------
# Offline verification against HTML dumps
# ---------------------------------------------------------------------------

def verify_with_dumps():
    """Test parsers against saved HTML dump files (no API calls needed)."""
    import os

    client = LiquipediaCSClient()
    ok = True

    # --- Test 1: Team results (Achievements) ---
    dump = "dump_team_results.html"
    if os.path.exists(dump):
        print("\n[TEST] Parsing team results from dump_team_results.html ...")
        html = open(dump, encoding="utf-8").read()
        soup = BeautifulSoup(html, "lxml")

        achievements = []
        recent = []
        for table in soup.find_all("table", class_=lambda c: c and "wikitable" in c):
            if table.find("td", class_="match-table-score"):
                recent = client._parse_recent_matches_table(
                    BeautifulSoup(str(table), "lxml"), "100 Thieves")
            elif "sortable" in " ".join(table.get("class", [])):
                for row in table.find_all("tr"):
                    if row.find("th") or "display:none" in row.get("style", ""):
                        continue
                    cells = row.find_all("td")
                    if len(cells) >= 8:
                        tier = client._extract_tier(cells[2])
                        achievements.append(tier)

        print(f"  Achievements tiers found: {achievements}")
        print(f"  Recent matches found: {len(recent)}")
        if achievements:
            print(f"  PASS: Tiers extracted correctly")
        else:
            print(f"  FAIL: No tiers extracted")
            ok = False

        if recent:
            print(f"  PASS: Recent matches parsed")
            for m in recent[:3]:
                print(f"    {m.date} {m.team1} vs {m.team2}  {m.score1}-{m.score2}  "
                      f"W:{m.winner}  [{m.tier}] {m.tournament}")
        else:
            print(f"  FAIL: No recent matches parsed")
            ok = False
    else:
        print(f"  SKIP: {dump} not found")

    # --- Test 2: PGL tournament matches ---
    dump = "dump_tournament_pgl2025.html"
    if os.path.exists(dump):
        print(f"\n[TEST] Parsing tournament matches from {dump} ...")
        html = open(dump, encoding="utf-8").read()
        soup = BeautifulSoup(html, "lxml")

        matchlist_matches = client._parse_matchlists(soup, "PGL/2025/Bucharest")
        bracket_matches = client._parse_brackets(soup, "PGL/2025/Bucharest")

        print(f"  Matchlist matches: {len(matchlist_matches)}")
        print(f"  Bracket matches:  {len(bracket_matches)}")

        if matchlist_matches:
            print(f"  PASS: Matchlist parsing works")
            for m in matchlist_matches[:3]:
                print(f"    {m.date} {m.team1} vs {m.team2}  {m.score1}-{m.score2}  "
                      f"[{m.best_of}] W:{m.winner}  Stage:{m.stage}")
                for mp in m.maps:
                    ot = " (OT)" if mp.is_overtime else ""
                    print(f"      {mp.map_name:12s} {mp.team1_score}-{mp.team2_score}"
                          f"  ({mp.team1_t_rounds}t/{mp.team1_ct_rounds}ct vs "
                          f"{mp.team2_t_rounds}t/{mp.team2_ct_rounds}ct){ot}")
        else:
            print(f"  FAIL: No matchlist matches found")
            ok = False

        maps_with_data = sum(1 for m in matchlist_matches if m.maps)
        print(f"  Matches with map data: {maps_with_data}/{len(matchlist_matches)}")
        if maps_with_data > 0:
            print(f"  PASS: Map-level data extracted")
        else:
            print(f"  FAIL: No map data extracted from matchlists")
            ok = False

        if bracket_matches:
            print(f"  PASS: Bracket parsing works")
            for m in bracket_matches[:3]:
                print(f"    {m.date} {m.team1} vs {m.team2}  {m.score1}-{m.score2}  "
                      f"[{m.best_of}] W:{m.winner}")
                for mp in m.maps:
                    ot = " (OT)" if mp.is_overtime else ""
                    print(f"      {mp.map_name:12s} {mp.team1_score}-{mp.team2_score}"
                          f"  ({mp.team1_t_rounds}t/{mp.team1_ct_rounds}ct vs "
                          f"{mp.team2_t_rounds}t/{mp.team2_ct_rounds}ct){ot}")
        else:
            print(f"  FAIL: No bracket matches found")
            ok = False

        bracket_maps = sum(1 for m in bracket_matches if m.maps)
        print(f"  Bracket matches with map data: {bracket_maps}/{len(bracket_matches)}")
    else:
        print(f"  SKIP: {dump} not found")

    # --- Test 3: StarLadder tournament ---
    dump = "dump_tournament_starladder2025.html"
    if os.path.exists(dump):
        print(f"\n[TEST] Parsing tournament matches from {dump} ...")
        html = open(dump, encoding="utf-8").read()
        soup = BeautifulSoup(html, "lxml")

        bracket_matches = client._parse_brackets(soup, "StarLadder/2025/Major")
        flat_matches = client._parse_flat_popups(soup, "StarLadder/2025/Major")

        print(f"  Bracket matches: {len(bracket_matches)}")
        print(f"  Flat matches:    {len(flat_matches)}")

        total = len(bracket_matches) + len(flat_matches)
        if total > 0:
            print(f"  PASS: Found {total} matches total")
            for m in (bracket_matches + flat_matches)[:3]:
                print(f"    {m.date} {m.team1} vs {m.team2}  {m.score1}-{m.score2}  "
                      f"[{m.best_of}] W:{m.winner}")
                for mp in m.maps:
                    ot = " (OT)" if mp.is_overtime else ""
                    print(f"      {mp.map_name:12s} {mp.team1_score}-{mp.team2_score}"
                          f"  ({mp.team1_t_rounds}t/{mp.team1_ct_rounds}ct vs "
                          f"{mp.team2_t_rounds}t/{mp.team2_ct_rounds}ct){ot}")
            sl_maps = sum(1 for m in bracket_matches + flat_matches if m.maps)
            print(f"  Matches with map data: {sl_maps}/{total}")
        else:
            print(f"  FAIL: No matches found")
            ok = False
    else:
        print(f"  SKIP: {dump} not found")

    return ok


# ---------------------------------------------------------------------------
# CSV export + full scrape pipeline
# ---------------------------------------------------------------------------

MAP_CSV_FIELDS = [
    "date", "timestamp", "tournament", "tournament_page", "tier",
    "stage", "best_of", "team1", "team2",
    "series_score_team1", "series_score_team2", "series_winner",
    "map_number", "map_name",
    "map_team1_score", "map_team2_score",
    "map_team1_t", "map_team1_ct",
    "map_team2_t", "map_team2_ct",
    "map_winner", "overtime",
]
LINEUP_CSV_FIELDS = [
    "tournament", "tournament_page", "team", "player_number",
    "player_gamertag", "player_country", "is_coach",
]


def write_maps_csv(matches: list[Match], tier: str, filepath: str,
                   write_header: bool = False):
    """Append match/map rows to maps.csv.  One row per map, or one row per
    series when no map-level data is available (map_number=0)."""
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAP_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for m in matches:
            base = {
                "date": m.date, "timestamp": m.timestamp,
                "tournament": m.tournament, "tournament_page": m.tournament_page,
                "tier": tier, "stage": m.stage, "best_of": m.best_of,
                "team1": m.team1, "team2": m.team2,
                "series_score_team1": m.score1,
                "series_score_team2": m.score2,
                "series_winner": m.winner,
            }
            if m.maps:
                for j, mp in enumerate(m.maps, 1):
                    writer.writerow({
                        **base,
                        "map_number": j, "map_name": mp.map_name,
                        "map_team1_score": mp.team1_score,
                        "map_team2_score": mp.team2_score,
                        "map_team1_t": mp.team1_t_rounds,
                        "map_team1_ct": mp.team1_ct_rounds,
                        "map_team2_t": mp.team2_t_rounds,
                        "map_team2_ct": mp.team2_ct_rounds,
                        "map_winner": mp.winner,
                        "overtime": mp.is_overtime,
                    })
            else:
                writer.writerow({
                    **base,
                    "map_number": 0, "map_name": "", "map_team1_score": "",
                    "map_team2_score": "", "map_team1_t": "", "map_team1_ct": "",
                    "map_team2_t": "", "map_team2_ct": "",
                    "map_winner": "", "overtime": "",
                })


def write_lineups_csv(lineups: list[dict], filepath: str,
                      write_header: bool = False):
    """Append lineup rows to lineups.csv."""
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LINEUP_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for entry in lineups:
            writer.writerow(entry)


def scrape_all_tournaments(tiers: list[str] = None,
                           rate_limit: float = 30,
                           cutoff_date: str = "2025-11-23"):
    """Main scraping pipeline: tier pages -> tournament pages -> CSV files."""
    PROGRESS_FILE = "_progress.json"
    MAPS_FILE = "maps.csv"
    LINEUPS_FILE = "lineups.csv"

    if tiers is None:
        tiers = ["S", "A", "B", "C"]

    client = LiquipediaCSClient(
        app_name="CSBettingResearch/1.0 (contact@email.com)",
        rate_limit=rate_limit,
    )

    print(f"[1/3] Fetching tier pages (cutoff: {cutoff_date}) ...")
    tournaments = client.get_portal_tournaments(
        tiers=tiers, cutoff_date=cutoff_date)
    print(f"      Found {len(tournaments)} tournaments in date window")

    done: set[str] = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, encoding="utf-8") as f:
            done = set(json.load(f))

    remaining = [t for t in tournaments if t["page"] not in done]
    print(f"      Already scraped: {len(done)}, remaining: {len(remaining)}")

    need_header = not done
    if need_header:
        write_maps_csv([], "", MAPS_FILE, write_header=True)
        write_lineups_csv([], LINEUPS_FILE, write_header=True)

    total_matches = 0
    total_lineups = 0

    print(f"\n[2/3] Scraping tournament pages ...\n")

    for i, tourn in enumerate(remaining, 1):
        page = tourn["page"]
        tier = tourn["tier"]
        name = tourn["name"]

        print(f"  [{i}/{len(remaining)}] {name} [{tier}] ...", end="", flush=True)

        try:
            soup = client._fetch_html(page)
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        matches = client.get_tournament_matches(page, soup=soup)
        for m in matches:
            m.tier = tier

        lineups = client.get_tournament_lineups(soup)
        for entry in lineups:
            entry["tournament"] = name
            entry["tournament_page"] = page

        write_maps_csv(matches, tier, MAPS_FILE)
        write_lineups_csv(lineups, LINEUPS_FILE)

        unique_teams = len({e["team"] for e in lineups})
        total_matches += len(matches)
        total_lineups += len(lineups)

        print(f"  {len(matches)} matches, {unique_teams} teams")

        done.add(page)
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(done), f)

    print(f"\n[3/3] Done.")
    print(f"{'='*70}")
    print(f"  Tournaments scraped : {len(done)}")
    print(f"  Total match rows    : {total_matches}")
    print(f"  Total lineup rows   : {total_lineups}")
    print(f"  Output files        : {MAPS_FILE}, {LINEUPS_FILE}")
    print(f"  Progress file       : {PROGRESS_FILE}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--verify" in sys.argv:
        print("Running offline verification against HTML dumps...\n")
        success = verify_with_dumps()
        print(f"\n{'='*80}")
        print(f"  Overall: {'ALL TESTS PASSED' if success else 'SOME TESTS FAILED'}")
        print(f"{'='*80}")
        sys.exit(0 if success else 1)

    if "--scrape" in sys.argv:
        tier_arg = None
        rl = 30.0
        cutoff = "2025-11-23"
        for j, arg in enumerate(sys.argv):
            if arg == "--tiers" and j + 1 < len(sys.argv):
                tier_arg = [t.strip().upper()
                            for t in sys.argv[j + 1].split(",")]
            if arg == "--rate-limit" and j + 1 < len(sys.argv):
                rl = float(sys.argv[j + 1])
            if arg == "--cutoff" and j + 1 < len(sys.argv):
                cutoff = sys.argv[j + 1]
        scrape_all_tournaments(tiers=tier_arg, rate_limit=rl,
                               cutoff_date=cutoff)
        sys.exit(0)

    client = LiquipediaCSClient(app_name="MyBettingResearch/1.0 (you@email.com)")

    # 1. List some teams
    print("\n[1] Teams (first 5)")
    teams = client.get_teams(limit=5)
    print(f"    {[t['name'] for t in teams]}")

    # 2. Upcoming tournaments for Team Vitality
    print("\n[2] Upcoming — Team Vitality")
    upcoming = client.get_team_upcoming("Team Vitality")
    print_upcoming(upcoming)

    # 3. Recent match results from team page
    print("\n[3] Recent Matches — Team Vitality")
    recent = client.get_team_matches("Team Vitality")
    print_matches(recent, "Team Vitality — Recent Matches")

    # 4. Tournament results (achievements)
    print("\n[4] Achievements — Team Vitality (S+A tier)")
    results = client.get_team_results("Team Vitality", limit=15, min_tier="A")
    print_results(results)

    # 5. Roster
    print("\n[5] Roster — Team Vitality")
    roster = client.get_team_roster("Team Vitality")
    print_roster(roster, "Team Vitality")

    # 6. Tournament list
    print("\n[6] S + A tier tournaments (10 most recent)")
    tourneys = client.get_tournaments(tiers=["S", "A"], limit=10)
    for t in tourneys:
        print(f"    [{t['tier']}] {t['name']}")

    # 7. Match scores from a specific tournament
    if tourneys:
        page = tourneys[0]["page"]
        print(f"\n[7] Matches from: {page}")
        tourn_matches = client.get_tournament_matches(page)
        print_matches(tourn_matches, f"{page} — matches", limit=15)

    # 8. H2H (from team match history — fast, single API call already done)
    print("\n[8] H2H — from Team Vitality's match history")
    h2h = client.get_head_to_head("Team Vitality", "Natus Vincere")
    print_h2h(h2h)

    # 9. Export all gathered data
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "upcoming_vitality": [asdict(u) for u in upcoming],
        "recent_matches_vitality": [asdict(m) for m in recent],
        "results_vitality": [asdict(r) for r in results],
        "tournament_matches": [asdict(m) for m in (tourn_matches if tourneys else [])],
        "h2h_vitality_navi": {
            "team1": h2h.team1, "team2": h2h.team2,
            "team1_wins": h2h.team1_wins, "team2_wins": h2h.team2_wins,
            "matches": [asdict(m) for m in h2h.matches],
        },
    }
    with open("liquipedia_data.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\n[9] Saved -> liquipedia_data.json")

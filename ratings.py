"""
CS2 Rating Engine
=================
Multi-layered team rating system for win probability estimation.
Reads historical data from maps.csv and lineups.csv.

Layers:
  1. Team Elo (map-agnostic) — tier-weighted K, margin-of-victory, time decay
  2. Map-specific Elo — per (team, map) ratings
  3. Round-level Beta-Binomial — per (team, map, side) round win rates + DP sim
  4. Lineup continuity — roster change detection + rating regression

Usage:
    python ratings.py backtest
    python ratings.py rankings [--top N]
    python ratings.py predict "Team A" "Team B" [--bo 3] [--maps Inferno,Mirage]
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAP_NAME_FIXES: dict[str, str] = {
    "Infenro": "Inferno",
    "Dust 2": "Dust II",
}
ACTIVE_MAPS = frozenset(
    {"Mirage", "Dust II", "Nuke", "Ancient", "Inferno", "Overpass", "Anubis"}
)


@dataclass
class Config:
    initial_rating: float = 1500.0
    k_base: dict[str, float] = field(
        default_factory=lambda: {"S": 100.0, "A":30.0, "B": 24.0, "C": 20.0, "Q": 16.0}
    )
    time_decay_halflife_days: float = 180.0
    mov_weight: float = 0.5
    mov_cap_series: float = 30.0
    mov_cap_map: float = 13.0
    roster_regression: float = 0.4
    roster_k_boost: float = 0.3
    roster_k_boost_games: int = 5
    beta_prior: float = 6.0
    round_decay_halflife_days: float = 120.0


# ---------------------------------------------------------------------------
# Step 0 — Data loading & cleaning
# ---------------------------------------------------------------------------

_NUMERIC_MAP_COLS = [
    "map_team1_score",
    "map_team2_score",
    "map_team1_t",
    "map_team1_ct",
    "map_team2_t",
    "map_team2_ct",
]


def load_data(
    maps_path: str = "maps.csv",
    lineups_path: str = "lineups.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load and clean maps + lineups CSVs.

    Returns
    -------
    valid_maps : per-map rows (only active pool, map_number>0), chrono-sorted.
    series_df  : one row per series with aggregated round totals, chrono-sorted.
    rosters    : {team -> [(timestamp, tournament_page, frozenset(players))]}
    """
    # ---- maps ----
    raw = pd.read_csv(maps_path)
    raw["map_name"] = raw["map_name"].replace(MAP_NAME_FIXES)

    for col in _NUMERIC_MAP_COLS:
        raw[col] = pd.to_numeric(raw[col], errors="coerce").fillna(0).astype(int)
    raw["map_number"] = pd.to_numeric(raw["map_number"], errors="coerce").fillna(0).astype(int)
    raw["timestamp"] = raw["timestamp"].astype(int)
    raw["series_score_team1"] = pd.to_numeric(raw["series_score_team1"], errors="coerce").fillna(0).astype(int)
    raw["series_score_team2"] = pd.to_numeric(raw["series_score_team2"], errors="coerce").fillna(0).astype(int)

    raw = raw.sort_values("timestamp").reset_index(drop=True)

    valid_maps = raw[raw["map_name"].isin(ACTIVE_MAPS) & (raw["map_number"] > 0)].copy()

    # ---- series aggregation ----
    series_records: list[dict] = []
    for (ts, t1, t2), grp in raw.groupby(["timestamp", "team1", "team2"]):
        first = grp.iloc[0]
        series_records.append(
            {
                "timestamp": int(ts),
                "date": first["date"],
                "team1": t1,
                "team2": t2,
                "score1": int(first["series_score_team1"]),
                "score2": int(first["series_score_team2"]),
                "winner": first["series_winner"],
                "tier": first["tier"],
                "best_of": first["best_of"],
                "tournament_page": first["tournament_page"],
                "stage": first.get("stage", ""),
                "total_rounds_team1": int(grp["map_team1_score"].sum()),
                "total_rounds_team2": int(grp["map_team2_score"].sum()),
                "n_maps_with_data": int((grp["map_number"] > 0).sum()),
            }
        )

    series_df = pd.DataFrame(series_records).sort_values("timestamp").reset_index(drop=True)

    # ---- lineups ----
    rosters = _build_rosters(lineups_path, raw)

    log.info(
        "Loaded %d map rows (%d valid), %d series, %d teams with rosters",
        len(raw),
        len(valid_maps),
        len(series_df),
        len(rosters),
    )
    return valid_maps, series_df, rosters


def _build_rosters(lineups_path: str, maps_df: pd.DataFrame) -> dict:
    lineups_df = pd.read_csv(lineups_path)
    players = lineups_df[lineups_df["is_coach"].astype(str).str.strip().str.lower() != "true"]

    tournament_first_ts = maps_df.groupby("tournament_page")["timestamp"].min()

    rosters: dict[str, list[tuple[int, str, frozenset[str]]]] = defaultdict(list)
    for (team, tourn), grp in players.groupby(["team", "tournament_page"]):
        roster_set = frozenset(grp["player_gamertag"].str.strip().str.lower())
        ts = int(tournament_first_ts.get(tourn, 0))
        rosters[team].append((ts, tourn, roster_set))

    for team in rosters:
        rosters[team].sort(key=lambda x: x[0])

    return dict(rosters)


# ---------------------------------------------------------------------------
# Layer 1 — Team Elo (map-agnostic)
# ---------------------------------------------------------------------------


@dataclass
class EloState:
    rating: float = 1500.0
    last_ts: int = 0
    games: int = 0
    k_boost_remaining: int = 0


class TeamElo:
    """Single scalar Elo per team with tier K-factor, MoV, and time decay."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ratings: dict[str, EloState] = defaultdict(
            lambda: EloState(cfg.initial_rating)
        )

    def _decay(self, team: str, current_ts: int):
        st = self.ratings[team]
        if st.last_ts <= 0 or current_ts <= st.last_ts:
            return
        days = (current_ts - st.last_ts) / 86400.0
        factor = 0.5 ** (days / self.cfg.time_decay_halflife_days)
        st.rating = self.cfg.initial_rating + (st.rating - self.cfg.initial_rating) * factor

    def expected(self, team_a: str, team_b: str) -> float:
        ra = self.ratings[team_a].rating
        rb = self.ratings[team_b].rating
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(
        self,
        team_a: str,
        team_b: str,
        outcome_a: float,
        tier: str,
        round_diff: int,
        current_ts: int,
    ):
        self._decay(team_a, current_ts)
        self._decay(team_b, current_ts)

        exp = self.expected(team_a, team_b)

        mov = math.log1p(abs(round_diff)) / math.log1p(self.cfg.mov_cap_series)
        k_tier = self.cfg.k_base.get(tier, 20.0)
        k = k_tier * (1.0 + self.cfg.mov_weight * mov)

        for team, sign in [(team_a, 1.0), (team_b, -1.0)]:
            st = self.ratings[team]
            k_eff = k * (1.0 + self.cfg.roster_k_boost) if st.k_boost_remaining > 0 else k
            st.rating += k_eff * sign * (outcome_a - exp)
            st.last_ts = current_ts
            st.games += 1
            if st.k_boost_remaining > 0:
                st.k_boost_remaining -= 1

    def apply_roster_regression(self, team: str, magnitude: float):
        st = self.ratings[team]
        regression = self.cfg.roster_regression * magnitude
        st.rating += (self.cfg.initial_rating - st.rating) * regression
        st.k_boost_remaining = self.cfg.roster_k_boost_games


# ---------------------------------------------------------------------------
# Layer 2 — Map-specific Elo
# ---------------------------------------------------------------------------


class MapElo:
    """Separate Elo per (team, map) pair."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ratings: dict[tuple[str, str], EloState] = defaultdict(
            lambda: EloState(cfg.initial_rating)
        )

    def _decay(self, team: str, map_name: str, current_ts: int):
        st = self.ratings[(team, map_name)]
        if st.last_ts <= 0 or current_ts <= st.last_ts:
            return
        days = (current_ts - st.last_ts) / 86400.0
        factor = 0.5 ** (days / self.cfg.time_decay_halflife_days)
        st.rating = self.cfg.initial_rating + (st.rating - self.cfg.initial_rating) * factor

    def expected_map(self, team_a: str, team_b: str, map_name: str) -> float:
        ra = self.ratings[(team_a, map_name)].rating
        rb = self.ratings[(team_b, map_name)].rating
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def expected_composite(self, team_a: str, team_b: str) -> float:
        return float(
            np.mean([self.expected_map(team_a, team_b, m) for m in ACTIVE_MAPS])
        )

    def update(
        self,
        team_a: str,
        team_b: str,
        map_name: str,
        outcome_a: float,
        tier: str,
        round_diff: int,
        current_ts: int,
    ):
        if map_name not in ACTIVE_MAPS:
            return
        self._decay(team_a, map_name, current_ts)
        self._decay(team_b, map_name, current_ts)

        exp = self.expected_map(team_a, team_b, map_name)
        mov = math.log1p(abs(round_diff)) / math.log1p(self.cfg.mov_cap_map)
        k_tier = self.cfg.k_base.get(tier, 20.0)
        k = k_tier * (1.0 + self.cfg.mov_weight * mov)

        sa = self.ratings[(team_a, map_name)]
        sb = self.ratings[(team_b, map_name)]
        sa.rating += k * (outcome_a - exp)
        sb.rating += k * ((1.0 - outcome_a) - (1.0 - exp))
        sa.last_ts = sb.last_ts = current_ts
        sa.games += 1
        sb.games += 1

    def apply_roster_regression(self, team: str, magnitude: float):
        regression = self.cfg.roster_regression * magnitude
        for map_name in ACTIVE_MAPS:
            key = (team, map_name)
            if key in self.ratings:
                st = self.ratings[key]
                st.rating += (self.cfg.initial_rating - st.rating) * regression


# ---------------------------------------------------------------------------
# Layer 3 — Round-level Beta-Binomial model
# ---------------------------------------------------------------------------


@dataclass
class BetaState:
    alpha: float = 6.0
    beta_: float = 6.0
    last_ts: int = 0


class RoundModel:
    """Beta-Binomial round win rates per (team, map, side).

    Combines ratings in log-odds space for head-to-head prediction, then uses
    exact DP for map win probability and analytical series probability.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._prior_a = cfg.beta_prior
        self._prior_b = cfg.beta_prior
        self.states: dict[tuple[str, str, str], BetaState] = defaultdict(
            lambda: BetaState(cfg.beta_prior, cfg.beta_prior)
        )

    # -- updates --

    def _decay(self, key: tuple, current_ts: int):
        st = self.states[key]
        if st.last_ts <= 0 or current_ts <= st.last_ts:
            return
        days = (current_ts - st.last_ts) / 86400.0
        factor = 0.5 ** (days / self.cfg.round_decay_halflife_days)
        st.alpha = self._prior_a + (st.alpha - self._prior_a) * factor
        st.beta_ = self._prior_b + (st.beta_ - self._prior_b) * factor

    def update(
        self,
        team: str,
        map_name: str,
        side: str,
        rounds_won: int,
        rounds_total: int,
        current_ts: int,
    ):
        if map_name not in ACTIVE_MAPS or rounds_total == 0:
            return
        key = (team, map_name, side)
        self._decay(key, current_ts)
        st = self.states[key]
        st.alpha += rounds_won
        st.beta_ += rounds_total - rounds_won
        st.last_ts = current_ts

    def apply_roster_regression(self, team: str, magnitude: float):
        factor = 1.0 - self.cfg.roster_regression * magnitude
        for key, st in list(self.states.items()):
            if key[0] == team:
                st.alpha = self._prior_a + (st.alpha - self._prior_a) * factor
                st.beta_ = self._prior_b + (st.beta_ - self._prior_b) * factor

    # -- prediction helpers --

    def round_win_rate(self, team: str, map_name: str, side: str) -> float:
        """Posterior mean round win rate."""
        st = self.states[(team, map_name, side)]
        return st.alpha / (st.alpha + st.beta_)

    def h2h_round_prob(
        self, team_a: str, team_b: str, map_name: str, a_side: str
    ) -> float:
        """P(A wins a round) when A is on *a_side* vs B on the opposite side.

        Uses log-odds combination: logit(P) = logit(p_a) - logit(p_b).
        Two average-strength teams (0.5 each) yield P = 0.5.
        """
        b_side = "ct" if a_side == "t" else "t"
        p_a = self.round_win_rate(team_a, map_name, a_side)
        p_b = self.round_win_rate(team_b, map_name, b_side)
        eps = 1e-9
        p_a = max(eps, min(1.0 - eps, p_a))
        p_b = max(eps, min(1.0 - eps, p_b))
        logit_diff = math.log(p_a / (1.0 - p_a)) - math.log(p_b / (1.0 - p_b))
        return 1.0 / (1.0 + math.exp(-logit_diff))

    # -- map / series win probability --

    def compute_map_win_prob(self, team_a: str, team_b: str, map_name: str) -> float:
        """P(A wins map) averaged over both possible starting sides."""
        p_a_ct = self.h2h_round_prob(team_a, team_b, map_name, "ct")
        p_a_t = self.h2h_round_prob(team_a, team_b, map_name, "t")
        p_start_ct = _regulation_and_ot(p_first=p_a_ct, p_second=p_a_t)
        p_start_t = _regulation_and_ot(p_first=p_a_t, p_second=p_a_ct)
        return 0.5 * (p_start_ct + p_start_t)

    def predict_series(
        self,
        team_a: str,
        team_b: str,
        maps: list[str] | None,
        best_of: int,
    ) -> float:
        wins_needed = best_of // 2 + 1
        if maps:
            probs = [self.compute_map_win_prob(team_a, team_b, m) for m in maps]
            return _series_win_prob(probs, wins_needed)
        avg_p = float(
            np.mean(
                [self.compute_map_win_prob(team_a, team_b, m) for m in ACTIVE_MAPS]
            )
        )
        return _series_win_prob([avg_p] * best_of, wins_needed)


# -- pure-function helpers for map / series DP --


def _regulation_and_ot(p_first: float, p_second: float) -> float:
    """P(team wins the map) given per-half round win probabilities.

    Regulation: MR12 (first to 13, 12 rounds per half, sides swap).
    Overtime: MR3 blocks (6 rounds each, need +4 to win, repeat if tied).
    """
    p_reg, p_tie = _regulation_dp(p_first, p_second)
    if p_tie < 1e-12:
        return p_reg
    p_ot = (p_first + p_second) / 2.0
    return p_reg + p_tie * _ot_win_prob(p_ot)


def _regulation_dp(
    p_first: float, p_second: float
) -> tuple[float, float]:
    """Returns (P(A wins in regulation), P(12-12 tie))."""
    memo: dict[tuple[int, int], tuple[float, float]] = {}

    def dp(a: int, b: int) -> tuple[float, float]:
        if a == 13:
            return (1.0, 0.0)
        if b == 13:
            return (0.0, 0.0)
        if a == 12 and b == 12:
            return (0.0, 1.0)
        key = (a, b)
        if key in memo:
            return memo[key]
        rnd = a + b  # 0-indexed round number
        p = p_first if rnd < 12 else p_second
        w_a, t_a = dp(a + 1, b)
        w_b, t_b = dp(a, b + 1)
        result = (p * w_a + (1.0 - p) * w_b, p * t_a + (1.0 - p) * t_b)
        memo[key] = result
        return result

    return dp(0, 0)


def _ot_win_prob(p: float) -> float:
    """P(A wins MR3 overtime) given constant round win probability *p*.

    Each OT period is 6 rounds. Win 4+ to take the period; 3-3 resets.
    P(win) = P(win period) / (1 - P(draw period)).
    """
    from math import comb

    p_win_period = sum(comb(6, k) * p**k * (1.0 - p) ** (6 - k) for k in range(4, 7))
    p_draw_period = comb(6, 3) * p**3 * (1.0 - p) ** 3
    denom = 1.0 - p_draw_period
    if denom < 1e-15:
        return 0.5
    return p_win_period / denom


def _series_win_prob(map_probs: list[float], wins_needed: int) -> float:
    """P(A wins a best-of series) via DP over ordered map probabilities."""
    n = len(map_probs)
    memo: dict[tuple[int, int, int], float] = {}

    def dp(idx: int, aw: int, bw: int) -> float:
        if aw == wins_needed:
            return 1.0
        if bw == wins_needed:
            return 0.0
        if idx >= n:
            return 0.0
        key = (idx, aw, bw)
        if key in memo:
            return memo[key]
        p = map_probs[idx]
        result = p * dp(idx + 1, aw + 1, bw) + (1.0 - p) * dp(idx + 1, aw, bw + 1)
        memo[key] = result
        return result

    return dp(0, 0, 0)


# ---------------------------------------------------------------------------
# Layer 4 — Lineup change detection
# ---------------------------------------------------------------------------


class LineupTracker:
    def __init__(self, rosters: dict):
        self._rosters = rosters
        self._last_seen: dict[str, frozenset[str]] = {}

    def check_and_update(self, team: str, tournament_page: str) -> float:
        """Return roster change magnitude in [0, 1]."""
        entries = self._rosters.get(team, [])
        current_roster: frozenset[str] | None = None
        for _ts, tp, roster in entries:
            if tp == tournament_page:
                current_roster = roster
                break

        if current_roster is None:
            return 0.0

        prev = self._last_seen.get(team)
        self._last_seen[team] = current_roster

        if prev is None or current_roster == prev:
            return 0.0

        changes = max(len(prev - current_roster), len(current_roster - prev))
        return min(changes / 5.0, 1.0)


# ---------------------------------------------------------------------------
# Rating engine — orchestrator
# ---------------------------------------------------------------------------


class RatingEngine:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.team_elo = TeamElo(self.cfg)
        self.map_elo = MapElo(self.cfg)
        self.round_model = RoundModel(self.cfg)
        self.lineup_tracker: LineupTracker | None = None
        self._seen_tournaments: dict[str, set[str]] = defaultdict(set)

    def set_rosters(self, rosters: dict):
        self.lineup_tracker = LineupTracker(rosters)

    def _check_lineup(self, team: str, tournament_page: str):
        if self.lineup_tracker is None:
            return
        if tournament_page in self._seen_tournaments[team]:
            return
        self._seen_tournaments[team].add(tournament_page)

        mag = self.lineup_tracker.check_and_update(team, tournament_page)
        if mag > 0:
            log.debug("Roster change detected: %s  (magnitude=%.2f)", team, mag)
            self.team_elo.apply_roster_regression(team, mag)
            self.map_elo.apply_roster_regression(team, mag)
            self.round_model.apply_roster_regression(team, mag)

    # -- prediction (read-only, no mutations) --

    def predict(
        self,
        team_a: str,
        team_b: str,
        best_of: int = 3,
        maps: list[str] | None = None,
    ) -> dict:
        p_elo = self.team_elo.expected(team_a, team_b)
        p_map_composite = self.map_elo.expected_composite(team_a, team_b)
        p_round = self.round_model.predict_series(team_a, team_b, maps, best_of)

        wins_needed = best_of // 2 + 1
        p_elo_series = _series_win_prob([p_elo] * best_of, wins_needed)

        if maps:
            map_probs = [self.map_elo.expected_map(team_a, team_b, m) for m in maps]
        else:
            map_probs = [p_map_composite] * best_of
        p_map_series = _series_win_prob(map_probs, wins_needed)

        return {
            "elo_raw": p_elo,
            "elo_series": p_elo_series,
            "map_composite": p_map_composite,
            "map_series": p_map_series,
            "round_series": p_round,
            "elo_team1": self.team_elo.ratings[team_a].rating,
            "elo_team2": self.team_elo.ratings[team_b].rating,
        }

    # -- process one series (mutates state) --

    def process_series(self, row: pd.Series, maps_for_series: pd.DataFrame):
        t1, t2 = row["team1"], row["team2"]
        ts = int(row["timestamp"])
        tier = str(row["tier"])
        tourn = str(row["tournament_page"])

        self._check_lineup(t1, tourn)
        self._check_lineup(t2, tourn)

        winner = row["winner"]
        outcome_a = 1.0 if winner == t1 else (0.0 if winner == t2 else 0.5)

        rd = row["total_rounds_team1"] - row["total_rounds_team2"]
        round_diff = rd if outcome_a == 1.0 else -rd

        self.team_elo.update(t1, t2, outcome_a, tier, round_diff, ts)

        if maps_for_series is None or maps_for_series.empty:
            return

        for _, mrow in maps_for_series.iterrows():
            map_name = mrow["map_name"]
            if map_name not in ACTIVE_MAPS or int(mrow["map_number"]) == 0:
                continue

            s1, s2 = int(mrow["map_team1_score"]), int(mrow["map_team2_score"])
            map_out = 1.0 if s1 > s2 else (0.0 if s2 > s1 else 0.5)

            self.map_elo.update(t1, t2, map_name, map_out, tier, abs(s1 - s2), ts)

            t1_t = int(mrow["map_team1_t"])
            t1_ct = int(mrow["map_team1_ct"])
            t2_t = int(mrow["map_team2_t"])
            t2_ct = int(mrow["map_team2_ct"])

            self.round_model.update(t1, map_name, "t", t1_t, t1_t + t2_ct, ts)
            self.round_model.update(t1, map_name, "ct", t1_ct, t1_ct + t2_t, ts)
            self.round_model.update(t2, map_name, "t", t2_t, t2_t + t1_ct, ts)
            self.round_model.update(t2, map_name, "ct", t2_ct, t2_ct + t1_t, ts)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------


def _parse_bo(bo_str: str) -> int:
    s = str(bo_str)
    if "5" in s:
        return 5
    if "1" in s:
        return 1
    return 3


def run_backtest(
    valid_maps: pd.DataFrame,
    series_df: pd.DataFrame,
    rosters: dict,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Walk-forward backtest: predict each series, then update ratings."""
    engine = RatingEngine(cfg)
    engine.set_rosters(rosters)

    maps_index: dict[tuple, pd.DataFrame] = {}
    for key, grp in valid_maps.groupby(["timestamp", "team1", "team2"]):
        maps_index[key] = grp

    records: list[dict] = []
    n = len(series_df)

    for idx, row in series_df.iterrows():
        t1, t2 = row["team1"], row["team2"]
        ts = row["timestamp"]

        pred = engine.predict(t1, t2, best_of=_parse_bo(row["best_of"]))
        actual = 1.0 if row["winner"] == t1 else (0.0 if row["winner"] == t2 else 0.5)

        records.append(
            {
                "date": row["date"],
                "timestamp": ts,
                "team1": t1,
                "team2": t2,
                "tier": row["tier"],
                "best_of": row["best_of"],
                "actual": actual,
                **{f"pred_{k}": v for k, v in pred.items()},
            }
        )

        series_maps = maps_index.get((ts, t1, t2), pd.DataFrame())
        engine.process_series(row, series_maps)

        if (len(records)) % 500 == 0:
            log.info("Backtest progress: %d / %d series", len(records), n)

    log.info("Backtest complete: %d series", len(records))
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(results: pd.DataFrame, pred_col: str) -> dict:
    valid = results.dropna(subset=[pred_col, "actual"])
    valid = valid[valid["actual"].isin([0.0, 1.0])]
    if valid.empty:
        return {}

    p = valid[pred_col].values.astype(float)
    y = valid["actual"].values.astype(float)
    eps = 1e-15
    pc = np.clip(p, eps, 1.0 - eps)

    log_loss = float(-np.mean(y * np.log(pc) + (1.0 - y) * np.log(1.0 - pc)))
    brier = float(np.mean((p - y) ** 2))
    accuracy = float(np.mean((p >= 0.5) == y))

    bins = np.linspace(0, 1, 11)
    indices = np.digitize(p, bins) - 1
    calibration: dict[str, dict] = {}
    for i in range(10):
        mask = indices == i
        if mask.sum() > 0:
            label = f"{bins[i]:.1f}-{bins[i + 1]:.1f}"
            calibration[label] = {
                "predicted_mean": float(p[mask].mean()),
                "actual_mean": float(y[mask].mean()),
                "count": int(mask.sum()),
            }

    return {
        "log_loss": log_loss,
        "brier_score": brier,
        "accuracy": accuracy,
        "n_samples": len(valid),
        "calibration": calibration,
    }


def print_metrics(metrics: dict, label: str = ""):
    if not metrics:
        print("  (no valid predictions)")
        return
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Log-loss : {metrics['log_loss']:.4f}")
    print(f"  Brier    : {metrics['brier_score']:.4f}")
    print(f"  Accuracy : {metrics['accuracy']:.2%}")
    print(f"  N        : {metrics['n_samples']}")
    cal = metrics.get("calibration", {})
    if cal:
        print(f"\n  {'Bucket':<12} {'Predicted':>10} {'Actual':>10} {'Count':>8}")
        print(f"  {'-' * 42}")
        for bucket, v in cal.items():
            print(
                f"  {bucket:<12} {v['predicted_mean']:>10.3f}"
                f" {v['actual_mean']:>10.3f} {v['count']:>8d}"
            )
    print()


def plot_calibration(metrics: dict, path: str = "calibration.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot %s", path)
        return

    cal = metrics.get("calibration", {})
    if not cal:
        return

    preds = [v["predicted_mean"] for v in cal.values()]
    actuals = [v["actual_mean"] for v in cal.values()]
    counts = [v["count"] for v in cal.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    ax1.scatter(preds, actuals, s=[c * 0.5 for c in counts], alpha=0.7, zorder=5)
    ax1.plot(preds, actuals, "o-", alpha=0.7, label="Model")
    ax1.set(
        xlabel="Predicted probability",
        ylabel="Observed frequency",
        title="Calibration curve",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(counts)), counts, tick_label=list(cal.keys()))
    ax2.set(xlabel="Probability bucket", ylabel="Count", title="Prediction distribution")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def _train_engine(args) -> tuple[RatingEngine, pd.DataFrame, pd.DataFrame]:
    """Load data and process all series. Returns trained engine + data."""
    valid_maps, series_df, rosters = load_data(args.maps, args.lineups)
    cfg = Config()
    engine = RatingEngine(cfg)
    engine.set_rosters(rosters)

    maps_index: dict[tuple, pd.DataFrame] = {}
    for key, grp in valid_maps.groupby(["timestamp", "team1", "team2"]):
        maps_index[key] = grp

    for _, row in series_df.iterrows():
        key = (row["timestamp"], row["team1"], row["team2"])
        engine.process_series(row, maps_index.get(key, pd.DataFrame()))

    return engine, valid_maps, series_df


def cmd_backtest(args):
    valid_maps, series_df, rosters = load_data(args.maps, args.lineups)
    cfg = Config()

    results = run_backtest(valid_maps, series_df, rosters, cfg)
    results.to_csv("backtest_results.csv", index=False)
    log.info("Saved backtest_results.csv")

    pred_cols = ["pred_elo_series", "pred_map_series", "pred_round_series"]
    for col in pred_cols:
        if col not in results.columns:
            continue
        m = compute_metrics(results, col)
        label = col.replace("pred_", "").replace("_", " ").title()
        print_metrics(m, label=label)
        plot_calibration(m, path=f"calibration_{col.replace('pred_', '')}.png")

    # Per-tier breakdown for the primary model
    primary = "pred_elo_series"
    for tier in sorted(results["tier"].dropna().unique()):
        subset = results[results["tier"] == tier]
        m = compute_metrics(subset, primary)
        if m:
            print_metrics(m, label=f"Elo Series — tier {tier}")


def cmd_rankings(args):
    engine, _, _ = _train_engine(args)

    teams = [
        (t, st.rating, st.games)
        for t, st in engine.team_elo.ratings.items()
    ]
    teams.sort(key=lambda x: x[1], reverse=True)

    n = args.top or 50
    print(f"\n{'Rank':<6} {'Team':<35} {'Elo':>8} {'Games':>6}")
    print("-" * 57)
    for i, (team, elo, games) in enumerate(teams[:n], 1):
        print(f"{i:<6} {team:<35} {elo:>8.1f} {games:>6}")
    print()


def cmd_predict(args):
    engine, _, _ = _train_engine(args)

    bo = args.bo or 3
    map_list = [m.strip() for m in args.map_names.split(",")] if args.map_names else None

    pred = engine.predict(args.team1, args.team2, best_of=bo, maps=map_list)

    print(f"\n{'=' * 55}")
    print(f"  {args.team1}  vs  {args.team2}  (Bo{bo})")
    if map_list:
        print(f"  Maps: {', '.join(map_list)}")
    print(f"{'=' * 55}")
    print(f"  Team Elo       : {pred['elo_team1']:.0f}  vs  {pred['elo_team2']:.0f}")
    print(f"  Elo P(win)     : {pred['elo_series']:.1%}")
    print(f"  Map-Elo P(win) : {pred['map_series']:.1%}")
    print(f"  Round P(win)   : {pred['round_series']:.1%}")
    print()


def main():
    p = argparse.ArgumentParser(description="CS2 Rating Engine")
    p.add_argument("--maps", default="maps.csv")
    p.add_argument("--lineups", default="lineups.csv")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("backtest", help="Walk-forward backtest on historical data")

    rk = sub.add_parser("rankings", help="Show current Elo rankings")
    rk.add_argument("--top", type=int, default=50)

    pr = sub.add_parser("predict", help="Predict a matchup")
    pr.add_argument("team1")
    pr.add_argument("team2")
    pr.add_argument("--bo", type=int, default=3)
    pr.add_argument("--map-names", default=None, help="Comma-separated map names")

    args = p.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "rankings":
        cmd_rankings(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()

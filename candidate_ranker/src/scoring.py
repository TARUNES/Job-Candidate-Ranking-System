"""
scoring.py
==========
Non-semantic sub-scores for the candidate ranking pipeline.

All sub-scores return a value in [0.0, 1.0].  The composite
non-semantic score is a weighted sum of five dimensions:

  Dimension          Weight
  ─────────────────  ──────
  Experience          15 %    Gaussian curve peaking at 7 yr (std=3.5)
  Notice period       10 %    Linear decay from 30 d (1.0) to 180 d (0.0)
  Location            10 %    Tier-1 India hubs preferred; min floor 0.35
  Platform signals    45 %    7 behavioural signals from the Redrob platform
  Trust signals       20 %    Identity verification + professional presence

The soft-penalty multiplier (from heuristics.compute_soft_penalty) is
applied on top of the composite score in rank.py — not here — so this
module is purely additive and side-effect free.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)

# Preferred Tier-1 Indian hiring hubs as stated in the JD.
_PREFERRED_HUBS: frozenset[str] = frozenset({
    "pune", "noida", "delhi", "ncr", "mumbai",
    "hyderabad", "bangalore", "bengaluru", "chennai", "gurgaon",
})


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def calculate_experience_score(years: float) -> float:
    """
    Maps years of experience to a [0, 1] score using a Gaussian curve
    that peaks at 7 years (the JD sweet spot).

    std_dev = 3.5 was chosen so that senior engineers (12–14 yr) receive
    a meaningful score (≥ 0.18) rather than being near-zeroed.  The
    symmetric bell shape also gently reduces scores for very junior
    candidates below 3 years.

    Example outputs:
      3 yr  ->  0.565
      7 yr  ->  1.000 (peak)
      10 yr ->  0.724
      14 yr ->  0.182
    """
    ideal_years = 7.0
    std_dev = 3.5
    score = math.exp(-((years - ideal_years) ** 2) / (2 * std_dev ** 2))
    log.debug("experience_score(%.1f yr) = %.4f", years, score)
    return score


def calculate_notice_period_score(notice_days: int) -> float:
    """
    Maps notice period length to a [0, 1] score.

    Scoring:
      ≤ 30 days  ->  1.0    (immediately or near-immediately available)
      30–180 d   ->  linear decay from 1.0 to 0.0
      ≥ 180 days ->  0.0    (schema maximum; full quarter notice)

    A 60-day notice (common in India) scores 0.80.
    A 90-day notice scores 0.60.
    """
    if notice_days <= 30:
        return 1.0
    if notice_days >= 180:
        return 0.0
    score = 1.0 - ((notice_days - 30) / 150.0)
    log.debug("notice_score(%d d) = %.4f", notice_days, score)
    return score


def calculate_location_score(location: str, signals: dict[str, Any]) -> float:
    """
    Maps a candidate's location and mobility preferences to a [0.35, 1.0] score.

    Tiers:
      1.0   — Already in a preferred Indian Tier-1 hub.
      0.60  — Outside preferred hubs but willing to relocate,
               OR preferred_work_mode is remote/hybrid/flexible.
               These candidates can be hired without physical relocation.
      0.35  — International, unwilling to relocate, requires onsite presence.
               Floor raised from 0.2 to 0.35 to avoid near-eliminating
               candidates who may be exceptional on other dimensions.

    Note: location carries only 10% of the composite score, so a 0.35
    floor means at worst a 0.065 reduction in the non-semantic score —
    not a disqualifying penalty.
    """
    location_lower = location.lower()

    if any(hub in location_lower for hub in _PREFERRED_HUBS):
        return 1.0

    willing_to_relocate: bool = signals.get("willing_to_relocate", False)
    work_mode: str = signals.get("preferred_work_mode", "onsite").lower()
    remote_compatible = work_mode in ("remote", "hybrid", "flexible")

    if willing_to_relocate or remote_compatible:
        log.debug("location_score('%s') = 0.60 (mobile/remote)", location)
        return 0.60

    log.debug("location_score('%s') = 0.35 (international, onsite-only)", location)
    return 0.35


def calculate_platform_signals_score(signals: dict[str, Any]) -> float:
    """
    Aggregates seven Redrob platform behavioural signals into a [0, 1] score.

    Signal weights (sum = 1.0):
      recruiter_response_rate   0.15  — responds to recruiter messages
      interview_completion_rate 0.15  — shows up for scheduled interviews
      github_activity_score     0.20  — open-source / side-project activity
      response_speed            0.15  — derived from avg_response_time_hours
      skill_assessment_avg      0.20  — mean score across completed assessments
      offer_acceptance_rate     0.10  — historical reliability on offers
      profile_completeness      0.05  — signals effort and seriousness

    Signal handling:
      github_activity_score = -1  (no GitHub linked)  ->  treated as 0.0
      offer_acceptance_rate = -1  (no offer history)   ->  treated as 0.0
      Empty skill_assessment_scores dict               ->  treated as 0.0
    """
    response_rate: float = signals["recruiter_response_rate"]
    interview_rate: float = signals["interview_completion_rate"]

    github_raw: float = signals["github_activity_score"]
    github_norm: float = github_raw / 100.0 if github_raw >= 0 else 0.0

    avg_response_hours: float = signals["avg_response_time_hours"]
    response_speed: float = max(0.0, 1.0 - (avg_response_hours / 200.0))

    skill_scores: dict[str, float] = signals["skill_assessment_scores"]
    if skill_scores:
        assessment_avg = sum(skill_scores.values()) / (len(skill_scores) * 100.0)
    else:
        assessment_avg = 0.0

    offer_rate_raw: float = signals["offer_acceptance_rate"]
    offer_rate: float = offer_rate_raw if offer_rate_raw >= 0 else 0.0

    completeness: float = signals["profile_completeness_score"] / 100.0

    score = (
        0.15 * response_rate
        + 0.15 * interview_rate
        + 0.20 * github_norm
        + 0.15 * response_speed
        + 0.20 * assessment_avg
        + 0.10 * offer_rate
        + 0.05 * completeness
    )
    log.debug(
        "platform_signals: rr=%.2f ir=%.2f gh=%.2f rs=%.2f aa=%.2f or=%.2f cp=%.2f -> %.4f",
        response_rate, interview_rate, github_norm, response_speed,
        assessment_avg, offer_rate, completeness, score,
    )
    return score


def calculate_trust_signals_score(signals: dict[str, Any]) -> float:
    """
    Scores identity verification and professional presence signals.

    Signal weights (sum = 1.0):
      verified_email         0.30  — basic identity confirmation
      verified_phone         0.30  — basic identity confirmation
      linkedin_connected     0.25  — professional identity linkage
      endorsements_received  0.15  — normalised to [0, 1] capped at 100

    A candidate with no verification at all scores 0.0 on this dimension.
    Trust carries 20% of the composite score, making it a meaningful but
    not dominant signal.
    """
    verified_email: bool = signals["verified_email"]
    verified_phone: bool = signals["verified_phone"]
    linkedin: bool = signals["linkedin_connected"]
    endorsements: int = signals["endorsements_received"]
    endorsements_norm: float = min(endorsements, 100) / 100.0

    score = (
        0.30 * float(verified_email)
        + 0.30 * float(verified_phone)
        + 0.25 * float(linkedin)
        + 0.15 * endorsements_norm
    )
    log.debug(
        "trust_signals: email=%s phone=%s linkedin=%s endorsements=%d -> %.4f",
        verified_email, verified_phone, linkedin, endorsements, score,
    )
    return score


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_non_semantic_score(candidate: dict[str, Any]) -> float:
    """
    Combines all five sub-scores into a single non-semantic score in [0, 1].

    Weight breakdown:
      Experience score      15 %
      Notice period score   10 %
      Location score        10 %
      Platform signals      45 %
      Trust signals         20 %

    The soft-penalty multiplier from heuristics.compute_soft_penalty() is
    applied to this value in rank.py.  This function is purely a weighted
    sum and introduces no additional clamping or fallback logic.
    """
    profile: dict[str, Any] = candidate["profile"]
    signals: dict[str, Any] = candidate["redrob_signals"]

    years: float = float(profile["years_of_experience"])
    notice: int = int(signals["notice_period_days"])
    location: str = profile["location"]

    exp_s = calculate_experience_score(years)
    notice_s = calculate_notice_period_score(notice)
    loc_s = calculate_location_score(location, signals)
    platform_s = calculate_platform_signals_score(signals)
    trust_s = calculate_trust_signals_score(signals)

    composite = (
        0.15 * exp_s
        + 0.10 * notice_s
        + 0.10 * loc_s
        + 0.45 * platform_s
        + 0.20 * trust_s
    )
    log.debug(
        "non_semantic[%s]: exp=%.3f not=%.3f loc=%.3f plat=%.3f trust=%.3f -> %.4f",
        candidate.get("candidate_id", "?"),
        exp_s, notice_s, loc_s, platform_s, trust_s, composite,
    )
    return composite

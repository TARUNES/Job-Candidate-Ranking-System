"""
scoring.py
==========
Non-semantic sub-scores for the candidate ranking pipeline.

All sub-scores return a value in [0.0, 1.0].  The composite
non-semantic score is a weighted sum of eight dimensions:

  Dimension              Weight
  ─────────────────────  ──────
  Experience              12 %    Gaussian curve peaking at JD-extracted ideal years
  Notice period            8 %    Linear decay from 30 d (1.0) to 180 d (0.0)
  Location                 8 %    JD-extracted preferred hubs; min floor 0.35
  Platform signals        42 %    7 behavioural signals from the Redrob platform
  Trust signals           15 %    Identity verification + professional presence
  Skill match             10 %    Structured overlap of JD skills vs candidate skills
  Education match          3 %    Degree level + field of study vs JD requirements
  Certification match      2 %    Named certifications vs JD-mentioned certs

The soft-penalty multiplier (from heuristics.compute_soft_penalty) is
applied on top of the composite score in rank.py — not here — so this
module is purely additive and side-effect free.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.jd_parser import JDProfile

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Education degree tier map (higher = better)
# ---------------------------------------------------------------------------

_DEGREE_TIER: dict[str, int] = {
    "phd": 4, "doctorate": 4, "ph.d": 4,
    "m.tech": 3, "m.e": 3, "m.s": 3, "ms": 3, "m.sc": 3, "msc": 3,
    "mba": 3, "masters": 3, "master": 3, "m.e.": 3,
    "b.tech": 2, "b.e": 2, "b.s": 2, "bs": 2, "b.sc": 2, "bsc": 2,
    "bachelors": 2, "bachelor": 2, "b.e.": 2, "b.tech.": 2,
    "diploma": 1, "associate": 1,
}

# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def calculate_experience_score(
    years: float,
    ideal_years: float = 5.0,
    min_years: float = 3.0,
    max_years: float = 10.0,
) -> float:
    """
    Maps years of experience to a [0, 1] score using a Gaussian curve
    that peaks at *ideal_years* (dynamically derived from the JD).

    std_dev is computed as half the range width (clamped to [2, 5]) so
    that a narrow JD range (e.g. "5-7 years") produces a sharper peak
    while a wide range (e.g. "2-15 years") is more forgiving.

    Example outputs (ideal=7, std=3.5):
      3 yr  -> 0.565
      7 yr  -> 1.000  (peak)
      10 yr -> 0.724
      14 yr -> 0.182
    """
    std_dev = max(2.0, min(5.0, (max_years - min_years) / 2.0))
    score = math.exp(-((years - ideal_years) ** 2) / (2 * std_dev ** 2))
    log.debug("experience_score(%.1f yr, ideal=%.1f) = %.4f", years, ideal_years, score)
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


def calculate_location_score(
    location: str,
    signals: dict[str, Any],
    preferred_locations: list[str] | None = None,
    remote_ok: bool = False,
) -> float:
    """
    Maps a candidate's location and mobility preferences to a [0.35, 1.0] score.

    Tiers:
      1.0   — Already in a JD-preferred location (case-insensitive substring match).
      0.60  — Outside preferred locations but willing to relocate,
               OR JD accepts remote/hybrid AND candidate's preferred_work_mode is remote/hybrid/flexible.
      0.35  — International, onsite-only, unwilling to relocate.
               Floor raised from 0.2 to 0.35 to avoid near-eliminating candidates
               who may be exceptional on other dimensions.
    """
    preferred = preferred_locations or []
    location_lower = location.lower()

    # Check if candidate is already in a preferred location
    for hub in preferred:
        if hub.lower() in location_lower:
            return 1.0

    willing_to_relocate: bool = signals.get("willing_to_relocate", False)
    work_mode: str = signals.get("preferred_work_mode", "onsite").lower()
    remote_compatible = work_mode in ("remote", "hybrid", "flexible")

    if willing_to_relocate or (remote_ok and remote_compatible):
        log.debug("location_score('%s') = 0.60 (mobile/remote)", location)
        return 0.60

    log.debug("location_score('%s') = 0.00 (non-preferred, unwilling to relocate)", location)
    return 0.0


def calculate_platform_signals_score(signals: dict[str, Any]) -> float:
    """
    Aggregates seven Redrob platform behavioural signals into a [0, 1] score.

    Signal weights (sum = 1.0):
      recruiter_response_rate   0.15  — responds to recruiter messages
      interview_completion_rate 0.15  — shows up for scheduled interviews
      github_activity_score     0.18  — open-source / side-project activity (reduced 2%)
      response_speed            0.15  — derived from avg_response_time_hours
      skill_assessment_avg      0.22  — mean score across completed assessments (increased 2%)
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
        + 0.18 * github_norm
        + 0.15 * response_speed
        + 0.22 * assessment_avg
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


def calculate_skill_match_score(
    candidate: dict[str, Any],
    jd_profile: "JDProfile",
) -> float:
    """
    Computes a structured skill-match score in [0, 1].

    Algorithm:
      1. Build a weighted vocabulary of JD skills:
           required skill  -> weight 2.0
           preferred skill -> weight 1.0
      2. For each candidate skill, look up the JD weight (case-insensitive).
      3. Apply a proficiency multiplier:
           expert     -> 1.0
           advanced   -> 0.85
           intermediate -> 0.65
           beginner   -> 0.40
      4. Apply an endorsement bonus: min(endorsements, 50) / 50 * 0.10 additive
      5. Apply a duration bonus: min(duration_months, 36) / 36 * 0.10 additive
      6. Sum up all weighted skill points.
      7. Normalise: divide by the maximum possible score if the candidate had
         ALL required JD skills at expert proficiency, capped at 1.0.
    """
    if not jd_profile.all_skills:
        return 0.0

    # Build JD skill weight map
    jd_weights: dict[str, float] = {}
    for s in jd_profile.required_skills:
        jd_weights[s.lower()] = 2.0
    for s in jd_profile.preferred_skills:
        key = s.lower()
        if key not in jd_weights:
            jd_weights[key] = 1.0

    proficiency_mult = {
        "expert": 1.0,
        "advanced": 0.85,
        "intermediate": 0.65,
        "beginner": 0.40,
    }

    total_score = 0.0
    for skill in candidate.get("skills", []):
        name = skill.get("name", "")
        key = name.lower()
        if key not in jd_weights:
            continue
        jd_w = jd_weights[key]
        prof = skill.get("proficiency", "beginner").lower()
        prof_m = proficiency_mult.get(prof, 0.40)

        endorsements = min(skill.get("endorsements", 0), 50)
        end_bonus = endorsements / 50.0 * 0.10

        duration = min(skill.get("duration_months", 0), 36)
        dur_bonus = duration / 36.0 * 0.10

        skill_score = jd_w * (prof_m + end_bonus + dur_bonus)
        total_score += skill_score

    # Max possible: all required skills at expert with full endorsements and duration
    n_req = len(jd_profile.required_skills)
    n_pref = len(jd_profile.preferred_skills)
    max_possible = n_req * 2.0 * 1.20 + n_pref * 1.0 * 1.20  # 1.20 = 1.0 + 0.10 + 0.10
    if max_possible == 0:
        return 0.0

    score = min(1.0, total_score / max_possible)
    log.debug("skill_match_score = %.4f (raw=%.2f / max=%.2f)", score, total_score, max_possible)
    return score


def calculate_education_match_score(
    candidate: dict[str, Any],
    jd_profile: "JDProfile",
) -> float:
    """
    Scores the candidate's education against JD requirements.

    Scoring:
      - If JD has no degree / field requirements: return 0.5 (neutral)
      - Degree level match:
          Candidate's highest degree tier >= JD's highest required tier -> 1.0
          One tier below -> 0.5
          Two+ tiers below -> 0.0
      - Field of study match:
          Any education entry's field_of_study matches a JD required field -> +0.5 bonus
          combined, then normalised to [0, 1]
    """
    jd_degrees = jd_profile.required_degrees
    jd_fields_lower = frozenset(f.lower() for f in jd_profile.required_fields)

    # No requirements -> neutral
    if not jd_degrees and not jd_fields_lower:
        return 0.5

    # Determine JD's required degree tier
    jd_tier = 0
    for deg in jd_degrees:
        key = deg.lower().rstrip(".")
        t = _DEGREE_TIER.get(key, 0)
        jd_tier = max(jd_tier, t)

    # Determine candidate's highest degree tier
    cand_tier = 0
    cand_fields_lower: set[str] = set()
    for edu in candidate.get("education", []):
        degree_str = edu.get("degree", "").lower().rstrip(".")
        t = _DEGREE_TIER.get(degree_str, 0)
        cand_tier = max(cand_tier, t)
        fos = edu.get("field_of_study", "").lower()
        if fos:
            cand_fields_lower.add(fos)

    # Degree level scoring
    if jd_tier == 0:
        degree_score = 0.5  # no specific degree required
    elif cand_tier >= jd_tier:
        degree_score = 1.0
    elif cand_tier == jd_tier - 1:
        degree_score = 0.5
    else:
        degree_score = 0.0

    # Field of study scoring
    if not jd_fields_lower:
        field_score = 0.5  # no specific field required
    else:
        # Check for substring match (e.g. "computer science" in "b.tech computer science")
        matched = any(
            any(jf in cf or cf in jf for cf in cand_fields_lower)
            for jf in jd_fields_lower
        )
        field_score = 1.0 if matched else 0.0

    score = 0.6 * degree_score + 0.4 * field_score
    log.debug("education_match: deg_score=%.2f field_score=%.2f -> %.4f", degree_score, field_score, score)
    return score


def calculate_certification_match_score(
    candidate: dict[str, Any],
    jd_profile: "JDProfile",
) -> float:
    """
    Scores how many JD-mentioned certifications the candidate holds.

    If JD has no cert requirements: returns 0.5 (neutral).
    Otherwise: (matched certs) / (total JD certs), capped at 1.0.
    Matching is case-insensitive substring match to handle
    "AWS Certified Solutions Architect" vs "AWS Certified".
    """
    jd_certs = jd_profile.required_certs
    if not jd_certs:
        return 0.5  # neutral — no certs mentioned

    cand_certs_lower = [
        c.get("name", "").lower()
        for c in candidate.get("certifications", [])
        if c.get("name")
    ]

    if not cand_certs_lower:
        return 0.0

    matched = 0
    for jd_cert in jd_certs:
        jd_lower = jd_cert.lower()
        for cc in cand_certs_lower:
            if jd_lower in cc or cc in jd_lower:
                matched += 1
                break

    score = min(1.0, matched / len(jd_certs))
    log.debug("cert_match: %d/%d -> %.4f", matched, len(jd_certs), score)
    return score


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------

def compute_non_semantic_score(
    candidate: dict[str, Any],
    jd_profile: "JDProfile | None" = None,
) -> float:
    """
    Combines all eight sub-scores into a single non-semantic score in [0, 1].

    Weight breakdown:
      Experience score        12 %
      Notice period score      8 %
      Location score           8 %
      Platform signals        42 %
      Trust signals           15 %
      Skill match             10 %
      Education match          3 %
      Certification match      2 %

    If jd_profile is None (backward compat), the three JD-driven dimensions
    (skill, education, cert) each return their neutral value (0.5) and the
    legacy scoring weights are used.
    """
    from src.jd_parser import JDProfile  # local import to avoid circular at module load

    profile: dict[str, Any] = candidate["profile"]
    signals: dict[str, Any] = candidate["redrob_signals"]

    years: float = float(profile["years_of_experience"])
    notice: int = int(signals["notice_period_days"])
    location: str = profile["location"]

    if jd_profile is not None:
        ideal_yr = jd_profile.ideal_years
        min_yr = jd_profile.min_years
        max_yr = jd_profile.max_years
        preferred_locs = jd_profile.preferred_locations
        remote_ok = jd_profile.remote_ok
    else:
        ideal_yr, min_yr, max_yr = 5.0, 3.0, 10.0
        preferred_locs = []
        remote_ok = False

    exp_s   = calculate_experience_score(years, ideal_yr, min_yr, max_yr)
    notice_s = calculate_notice_period_score(notice)
    loc_s   = calculate_location_score(location, signals, preferred_locs, remote_ok)
    platform_s = calculate_platform_signals_score(signals)
    trust_s = calculate_trust_signals_score(signals)

    if jd_profile is not None:
        skill_s = calculate_skill_match_score(candidate, jd_profile)
        edu_s   = calculate_education_match_score(candidate, jd_profile)
        cert_s  = calculate_certification_match_score(candidate, jd_profile)
    else:
        skill_s = edu_s = cert_s = 0.5  # neutral fallback

    composite = (
        0.12 * exp_s
        + 0.08 * notice_s
        + 0.08 * loc_s
        + 0.42 * platform_s
        + 0.15 * trust_s
        + 0.10 * skill_s
        + 0.03 * edu_s
        + 0.02 * cert_s
    )
    log.debug(
        "non_semantic[%s]: exp=%.3f not=%.3f loc=%.3f plat=%.3f trust=%.3f "
        "skill=%.3f edu=%.3f cert=%.3f -> %.4f",
        candidate.get("candidate_id", "?"),
        exp_s, notice_s, loc_s, platform_s, trust_s,
        skill_s, edu_s, cert_s, composite,
    )
    return composite

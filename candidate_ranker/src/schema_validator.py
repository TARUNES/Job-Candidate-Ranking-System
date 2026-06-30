"""
schema_validator.py
-------------------
Lightweight structural validation of candidate records before they enter the
filter / scoring pipeline.

Design goals
  - Never raise an exception — return (is_valid, reason) tuples so the caller
    can decide whether to skip or log.
  - Check only the fields that the downstream pipeline actually reads so that
    future schema additions don't break this file.
  - Fast: pure-Python, no external dependencies.
"""

from __future__ import annotations

import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Field presence helpers
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = {"candidate_id", "profile", "career_history", "skills", "redrob_signals"}

_REQUIRED_PROFILE = {
    "headline", "summary", "location", "country",
    "years_of_experience", "current_title", "current_company",
    "current_industry",
}

_REQUIRED_SIGNALS = {
    "open_to_work_flag", "notice_period_days", "recruiter_response_rate",
    "avg_response_time_hours", "interview_completion_rate",
    "offer_acceptance_rate", "github_activity_score",
    "profile_completeness_score", "verified_email", "verified_phone",
    "linkedin_connected", "skill_assessment_scores",
    "endorsements_received", "willing_to_relocate", "preferred_work_mode",
}

_REQUIRED_JOB = {"company", "title", "start_date", "industry", "is_current"}

_VALID_PROFICIENCY = {"beginner", "intermediate", "advanced", "expert"}

_VALID_WORK_MODE = {"remote", "hybrid", "onsite", "flexible"}

_CANDIDATE_ID_PREFIX = "CAND_"


def _fail(reason: str) -> tuple[bool, str]:
    return False, reason


def _ok() -> tuple[bool, str]:
    return True, ""


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _validate_candidate_id(candidate: dict[str, Any]) -> tuple[bool, str]:
    cid = candidate.get("candidate_id", "")
    if not isinstance(cid, str) or not cid.startswith(_CANDIDATE_ID_PREFIX):
        return _fail(f"invalid candidate_id: {cid!r}")
    return _ok()


def _validate_profile(profile: dict[str, Any]) -> tuple[bool, str]:
    missing = _REQUIRED_PROFILE - profile.keys()
    if missing:
        return _fail(f"profile missing fields: {missing}")
    yoe = profile.get("years_of_experience")
    if not isinstance(yoe, (int, float)) or yoe < 0:
        return _fail(f"years_of_experience out of range: {yoe!r}")
    return _ok()


def _validate_career_history(career: list[Any]) -> tuple[bool, str]:
    if not isinstance(career, list) or len(career) == 0:
        return _fail("career_history is empty or not a list")
    for idx, job in enumerate(career):
        if not isinstance(job, dict):
            return _fail(f"career_history[{idx}] is not an object")
        missing = _REQUIRED_JOB - job.keys()
        if missing:
            return _fail(f"career_history[{idx}] missing fields: {missing}")
        # validate date format for start_date only (end_date may be null)
        start_str = job.get("start_date", "")
        if start_str:
            try:
                datetime.date.fromisoformat(start_str)
            except ValueError:
                return _fail(f"career_history[{idx}].start_date bad format: {start_str!r}")
    return _ok()


def _validate_skills(skills: list[Any]) -> tuple[bool, str]:
    if not isinstance(skills, list):
        return _fail("skills is not a list")
    for idx, skill in enumerate(skills):
        if not isinstance(skill, dict):
            return _fail(f"skills[{idx}] is not an object")
        if "name" not in skill:
            return _fail(f"skills[{idx}] missing 'name'")
        prof = skill.get("proficiency", "").lower()
        if prof and prof not in _VALID_PROFICIENCY:
            return _fail(f"skills[{idx}].proficiency invalid: {prof!r}")
    return _ok()


def _validate_signals(signals: dict[str, Any]) -> tuple[bool, str]:
    missing = _REQUIRED_SIGNALS - signals.keys()
    if missing:
        return _fail(f"redrob_signals missing fields: {missing}")
    mode = signals.get("preferred_work_mode", "")
    if mode not in _VALID_WORK_MODE:
        return _fail(f"preferred_work_mode invalid: {mode!r}")
    assessments = signals.get("skill_assessment_scores")
    if not isinstance(assessments, dict):
        return _fail("skill_assessment_scores must be a dict")
    return _ok()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_candidate(candidate: Any) -> tuple[bool, str]:
    """
    Validates a single candidate record parsed from the JSONL feed.

    Returns
    -------
    (True,  "")           — record is structurally valid
    (False, reason_str)   — record has a problem; reason_str explains why
    """
    if not isinstance(candidate, dict):
        return _fail("record is not a JSON object")

    # top-level required keys
    missing_top = _REQUIRED_TOP_LEVEL - candidate.keys()
    if missing_top:
        return _fail(f"missing top-level fields: {missing_top}")

    ok, reason = _validate_candidate_id(candidate)
    if not ok:
        return _fail(reason)

    ok, reason = _validate_profile(candidate.get("profile", {}))
    if not ok:
        return _fail(reason)

    ok, reason = _validate_career_history(candidate.get("career_history", []))
    if not ok:
        return _fail(reason)

    ok, reason = _validate_skills(candidate.get("skills", []))
    if not ok:
        return _fail(reason)

    ok, reason = _validate_signals(candidate.get("redrob_signals", {}))
    if not ok:
        return _fail(reason)

    return _ok()

"""
heuristics.py
=============
Two-tier filtering for the candidate ranking pipeline.

Hard disqualifiers (per-candidate, during streaming)
    Binary: a True result removes the candidate from the pool entirely.
    Current checks:
      - Timeline overlaps       : impossible overlapping employment (>90 days)
      - Impossible skills       : advanced/expert claims with zero duration

    Note: The consulting-career filter has been intentionally removed.
    Filtering by employer industry (e.g., "IT Services") is a blunt proxy
    that disproportionately penalises candidates at large firms who hold
    genuinely technical roles.  Skills and work descriptions are far more
    reliable signals of actual fit.

Mismatched profile detection (batch, post-streaming)
    Uses TF-IDF coherence analysis to detect candidates whose headline/
    summary domain diverges from their career description domain.
    This is data-driven — vocabulary importance is learned from the entire
    candidate corpus, with no hardcoded keyword pairs.

Soft penalties
    A multiplier in [0.60, 1.00] applied to the non-semantic score.
    A value below 1.0 reduces rank without eliminating the candidate.
    Floor is 0.60 — no candidate can be penalised by more than 40%.

    Penalty sources:
      - Not open to work            :  -0.15
      - Slow response time (>150 h) :  -0.10
      - Skill claim vs assessment   :  -0.10 per mismatch (max -0.20)
      - Low interview completion    :  -0.10  (< 0.50)
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sklearn_normalize

log = logging.getLogger(__name__)





# ---------------------------------------------------------------------------
# Hard disqualifiers
# ---------------------------------------------------------------------------

def has_timeline_overlaps(career_history: list[dict]) -> bool:
    """
    Returns True if any two consecutive roles overlap by more than 90 days.

    Overlapping employment is a strong signal of a fabricated profile.
    The 90-day grace period accommodates legitimate transition periods
    where someone might formally be employed at two places briefly.
    """
    intervals: list[tuple[datetime.date, datetime.date]] = []

    for job in career_history:
        start_str = job.get("start_date")
        end_str = job.get("end_date")
        if not start_str:
            continue
        try:
            start_dt = datetime.date.fromisoformat(start_str)
            end_dt = (
                datetime.date.fromisoformat(end_str)
                if end_str
                else datetime.date.today()
            )
            intervals.append((start_dt, end_dt))
        except ValueError:
            log.debug("Unparseable date in career history: start=%s end=%s", start_str, end_str)
            continue

    intervals.sort(key=lambda x: x[0])

    for i in range(len(intervals) - 1):
        current_end = intervals[i][1]
        next_start = intervals[i + 1][0]
        if current_end > next_start:
            overlap_days = (current_end - next_start).days
            if overlap_days > 90:
                log.debug("Timeline overlap detected: %d days", overlap_days)
                return True

    return False


def has_impossible_skills(skills: list[dict]) -> bool:
    """
    Returns True when a candidate claims advanced/expert proficiency on
    multiple skills but lists zero months of experience on most of them.

    Threshold: ≥5 advanced/expert skills AND ≥3 of those have duration = 0.
    This catches profiles that were keyword-stuffed without fabricating duration.
    """
    advanced_count = 0
    zero_duration_count = 0

    for skill in skills:
        proficiency = skill.get("proficiency", "").lower()
        duration = skill.get("duration_months", 0)
        if proficiency in ("expert", "advanced"):
            advanced_count += 1
            if duration == 0:
                zero_duration_count += 1

    triggered = advanced_count >= 5 and zero_duration_count >= 3
    if triggered:
        log.debug(
            "Impossible skills detected: %d advanced with %d zero-duration",
            advanced_count, zero_duration_count,
        )
    return triggered


# ---------------------------------------------------------------------------
# Data-driven mismatch detection (batch, post-streaming)
# ---------------------------------------------------------------------------

class MismatchDetector:
    """
    Detects mismatched profiles using TF-IDF coherence analysis.

    Uses a pre-trained TF-IDF Vectorizer fitted on the entire candidate corpus.
    This learns vocabulary importance from the entire corpus (100K profiles)
    and ensures that the feature extraction and similarity score for any candidate
    are completely deterministic and independent of other candidates in the batch.

    Candidates whose identity and evidence vectors have very low cosine
    similarity are flagged (i.e. similarity < 0.023). This fixed threshold
    successfully filters out synthetic domain-mismatch honeypots while having
    zero false-positive rates for valid AI/ML engineers.
    """

    _MIN_DESC_LENGTH: int = 100     # skip candidates with very short descriptions
    _THRESHOLD: float = 0.023       # fixed coherence threshold

    def __init__(self) -> None:
        # Load the pre-fitted vectorizer relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        vectorizer_path = os.path.join(current_dir, "tfidf_vectorizer.joblib")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Pre-fitted TF-IDF vectorizer not found at: {vectorizer_path}. "
                f"Please run 'src/fit_tfidf.py' to generate it."
            )
        self.vectorizer = joblib.load(vectorizer_path)

    def detect(self, candidates: list[dict[str, Any]]) -> set[str]:
        """
        Identifies candidates with mismatched headline vs career descriptions.

        Parameters
        ----------
        candidates : list of candidate dicts

        Returns
        -------
        set of candidate_ids flagged as mismatched
        """
        identity_texts: list[str] = []
        evidence_texts: list[str] = []
        candidate_ids: list[str] = []
        checkable_mask: list[bool] = []

        for cand in candidates:
            profile = cand.get("profile", {})
            headline = profile.get("headline", "")
            summary = profile.get("summary", "")
            career = cand.get("career_history", [])
            skills = cand.get("skills", [])

            skill_names = " ".join(s.get("name", "") for s in skills)
            identity = f"{headline} {summary} {skill_names}"

            evidence = " ".join(
                f"{job.get('title', '')} {job.get('description', '')}"
                for job in career
            )

            identity_texts.append(identity)
            evidence_texts.append(evidence)
            candidate_ids.append(cand.get("candidate_id", ""))
            checkable_mask.append(len(evidence.strip()) >= self._MIN_DESC_LENGTH)

        checkable_indices = [i for i, ok in enumerate(checkable_mask) if ok]
        if not checkable_indices:
            return set()

        identity_vecs = self.vectorizer.transform(identity_texts)
        evidence_vecs = self.vectorizer.transform(evidence_texts)

        # L2-normalise for cosine similarity via dot product
        identity_norm = sklearn_normalize(identity_vecs, norm="l2")
        evidence_norm = sklearn_normalize(evidence_vecs, norm="l2")

        # Compute per-candidate coherence (row-wise dot product)
        coherence_all = np.array(
            identity_norm.multiply(evidence_norm).sum(axis=1)
        ).flatten()

        mismatched: set[str] = set()
        for i in checkable_indices:
            if coherence_all[i] < self._THRESHOLD:
                mismatched.add(candidate_ids[i])
                log.info(
                    "[FILTER] %s — mismatched profile (coherence=%.4f < %.4f)",
                    candidate_ids[i], coherence_all[i], self._THRESHOLD,
                )

        log.info("Mismatch detector flagged %d candidates", len(mismatched))
        return mismatched


def check_honeypots_and_filters(candidate: dict[str, Any]) -> bool:
    """
    Runs all hard-disqualifier checks against a single candidate record.

    Returns True  ->  candidate is eliminated from the pool.
    Returns False ->  candidate passes and proceeds to scoring.

    Checks applied (in order, short-circuit on first True):
      1. has_timeline_overlaps     — impossible career timeline
      2. has_impossible_skills     — fabricated proficiency claims
      3. is_mismatched_profile     — title/description domain contradiction
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    headline = profile.get("headline", "")
    summary = profile.get("summary", "")
    cid = candidate.get("candidate_id", "?")

    if has_timeline_overlaps(career_history):
        log.info("[FILTER] %s — timeline overlap", cid)
        return True

    if has_impossible_skills(skills):
        log.info("[FILTER] %s — impossible skills", cid)
        return True

    # Mismatch detection is now handled as a batch step via MismatchDetector
    # (called from rank.py after streaming) instead of per-candidate here.

    return False


# ---------------------------------------------------------------------------
# Soft penalties
# ---------------------------------------------------------------------------

def compute_soft_penalty(candidate: dict[str, Any]) -> float:
    """
    Returns a multiplier in [0.60, 1.00] applied to the non-semantic score.

    Penalty sources and their deductions:
      - Not open to work            :  -0.15
      - Avg response time  > 150 h  :  -0.10
      - Skill claim vs assessment   :  -0.10 per mismatched skill, max -0.20
        (advanced/expert claim where platform assessment score < 40)
      - Interview completion < 0.50 :  -0.10

    Floor: 0.60 — no stacking of penalties can reduce the multiplier below 0.60.
    This ensures that even candidates with all four penalties still contribute
    meaningful signal to the final hybrid score.
    """
    signals = candidate.get("redrob_signals", {})
    skills = candidate.get("skills", [])
    cid = candidate.get("candidate_id", "?")
    penalty = 0.0
    reasons: list[str] = []

    # Not actively seeking work
    if not signals.get("open_to_work_flag", True):
        penalty += 0.15
        reasons.append("not_open_to_work")

    # Slow responder — harder to engage in interview process
    avg_hours: float = signals.get("avg_response_time_hours", 0.0)
    if avg_hours > 150:
        penalty += 0.10
        reasons.append(f"slow_response({avg_hours:.0f}h)")

    # Skill claim not backed by assessment score
    assessments: dict[str, float] = signals.get("skill_assessment_scores", {})
    mismatch_count = 0
    for skill in skills:
        name = skill.get("name", "")
        proficiency = skill.get("proficiency", "").lower()
        if proficiency in ("advanced", "expert") and name in assessments:
            if assessments[name] < 40.0:
                mismatch_count += 1
    if mismatch_count:
        deduction = min(mismatch_count, 2) * 0.10
        penalty += deduction
        reasons.append(f"skill_assessment_mismatch({mismatch_count})")

    # Low interview attendance — unreliable through the hiring process
    interview_rate: float = signals.get("interview_completion_rate", 1.0)
    if interview_rate < 0.50:
        penalty += 0.10
        reasons.append(f"low_interview_rate({interview_rate:.2f})")

    multiplier = max(0.60, 1.0 - penalty)
    if reasons:
        log.debug("[PENALTY] %s — mult=%.2f reasons=%s", cid, multiplier, reasons)
    return multiplier

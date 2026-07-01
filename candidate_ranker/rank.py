"""
rank.py
=======
CLI entry-point for the Intelligent Candidate Discovery & Ranking System.

Fully dynamic two-stage pipeline — all scoring parameters are derived
from the uploaded Job Description at runtime.

    1.  Parse CLI arguments
    2.  Load and parse the Job Description (JD) → JDProfile
    3.  Embed the JD text for semantic scoring
    4.  Stream candidate records from JSONL or JSONL.GZ
    5.  Schema-validate each record  — skip malformed ones
    6.  Hard-filter via heuristics   — skip disqualified ones
    7.  Batch TF-IDF mismatch detection — data-driven honeypot filter
    8.  Pre-rank ALL valid candidates by non-semantic score (instant)
    9.  Shortlist top-N by non-semantic score (default 2000)
   10.  Batch-encode only the shortlisted candidates
   11.  Compute hybrid score per shortlisted candidate:
            final = 0.45 * semantic + 0.55 * (non_semantic * soft_penalty)
   12.  Sort descending by score, emit top 100 to submission.csv

Usage
    python rank.py \\
        --candidates path/to/candidates.jsonl(.gz) \\
        --jd         path/to/job_description.docx \\
        --out        path/to/submission.csv

Compute constraints (as per hackathon spec)
    Runtime  ≤ 5 min wall-clock
    RAM      ≤ 16 GB
    Compute  CPU only — no GPU
    Network  Off — no external API calls
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_docx_text, stream_candidates
from src.embeddings import get_sentence_embeddings
from src.heuristics import MismatchDetector, check_honeypots_and_filters, compute_soft_penalty
from src.jd_parser import JDParser, JDProfile
from src.schema_validator import validate_candidate
from src.scoring import compute_non_semantic_score


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hybrid score weights
# ---------------------------------------------------------------------------

_SEMANTIC_WEIGHT: float = 0.45
_NON_SEMANTIC_WEIGHT: float = 0.55
_SHORTLIST_SIZE: int = 2000          # Only embed the top-N by non-semantic score
_MAX_CANDIDATE_TEXT_LEN: int = 600   # Truncate candidate text for faster encoding


# ---------------------------------------------------------------------------
# Text extraction (full — no artificial caps)
# ---------------------------------------------------------------------------

def extract_candidate_text(candidate: dict[str, Any]) -> str:
    """
    Builds a rich text string from a candidate record for semantic embedding.

    Includes all available profile signals (not artificially capped):
      - Headline and professional summary
      - All career role titles + descriptions (not capped at 2)
      - All skills formatted as "SkillName (proficiency)"
      - All education entries (degree + field of study)
      - All certification names
      - Language proficiency entries

    The final text is truncated to _MAX_CANDIDATE_TEXT_LEN characters
    before passing to the sentence encoder. The model's tokeniser will
    further truncate, but shorter inputs encode faster.
    """
    profile = candidate["profile"]
    headline: str = profile["headline"]
    summary: str = profile["summary"]

    # All career roles (was: first 2 only)
    history_parts: list[str] = []
    for job in candidate.get("career_history", []):
        title = job.get("title", "")
        desc = job.get("description", "")
        if desc:
            history_parts.append(f"{title}: {desc}")

    # All skills (was: top 8 only)
    skill_parts: list[str] = [
        f"{s['name']} ({s['proficiency']})"
        for s in candidate.get("skills", [])
        if s.get("name")
    ]

    edu_parts: list[str] = [
        f"{e.get('degree', '')} in {e['field_of_study']}"
        for e in candidate.get("education", [])
        if e.get("field_of_study")
    ]

    cert_parts: list[str] = [
        c["name"]
        for c in candidate.get("certifications", [])
        if c.get("name")
    ]

    lang_parts: list[str] = [
        f"{l['language']} ({l['proficiency']})"
        for l in candidate.get("languages", [])
        if l.get("language")
    ]

    sections = [
        headline,
        summary,
        " ".join(history_parts),
        "Skills: " + ", ".join(skill_parts) if skill_parts else "",
        "Education: " + ". ".join(edu_parts) if edu_parts else "",
        "Certifications: " + ". ".join(cert_parts) if cert_parts else "",
        "Languages: " + ", ".join(lang_parts) if lang_parts else "",
    ]
    text = " ".join(s for s in sections if s).strip()
    return text[:_MAX_CANDIDATE_TEXT_LEN]


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------

def select_reasoning(
    candidate: dict[str, Any],
    semantic_score: float,
    total_score: float,
    jd_profile: JDProfile,
) -> str:
    """
    Generates a 1–2 sentence reasoning string grounded in the candidate's
    actual profile data and the JD's required/preferred skills.

    Logic:
      1.  Find which candidate skills appear in the JD's required skills first,
          then preferred skills (structured match, not just keyword overlap).
      2.  If none match, fall back to the top-3 skills by endorsement count.
      3.  Choose tone (strong / moderate / lower) based on total_score.
      4.  Note unavailability or relocation when relevant.
    """
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    years: float = profile["years_of_experience"]
    title: str = profile["current_title"]
    location: str = profile["location"]
    open_to_work: bool = signals["open_to_work_flag"]
    willing_to_relocate: bool = signals["willing_to_relocate"]

    # Identify JD-relevant skills — required first, then preferred
    candidate_skills = [
        s["name"] for s in candidate.get("skills", []) if s.get("name")
    ]
    candidate_skills_lower = {s.lower(): s for s in candidate_skills}

    matched_required = [
        candidate_skills_lower[s.lower()]
        for s in jd_profile.required_skills
        if s.lower() in candidate_skills_lower
    ]
    matched_preferred = [
        candidate_skills_lower[s.lower()]
        for s in jd_profile.preferred_skills
        if s.lower() in candidate_skills_lower
        and s.lower() not in {r.lower() for r in matched_required}
    ]
    matched_skills = matched_required + matched_preferred

    if not matched_skills:
        sorted_by_endorsements = sorted(
            candidate.get("skills", []),
            key=lambda x: x.get("endorsements", 0),
            reverse=True,
        )
        matched_skills = [
            s["name"] for s in sorted_by_endorsements[:3] if s.get("name")
        ]

    skills_str = ", ".join(matched_skills[:4]) if matched_skills else "general skills"

    location_note = f" based in {location}" if location else ""
    relocation_note = " (open to relocation)" if willing_to_relocate and location else ""
    availability_note = (
        " Note: currently not marked open to work."
        if not open_to_work else ""
    )

    if total_score >= 0.55:
        return (
            f"{title} with {years:.1f} YOE{location_note}{relocation_note}, "
            f"strong JD alignment via: {skills_str}.{availability_note}"
        )
    if total_score >= 0.38:
        return (
            f"{title} with {years:.1f} YOE{location_note}. "
            f"Partial JD overlap through: {skills_str}. "
            f"Moderate semantic similarity; may need assessment.{availability_note}"
        )
    return (
        f"{title} with {years:.1f} YOE{location_note}. "
        f"Limited skill overlap with the JD ({skills_str}). "
        f"Ranked lower due to weaker profile-to-JD alignment.{availability_note}"
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args():
    """
    Parses and validates CLI arguments.  Exits with a clear error message
    if required files are missing or arguments are malformed.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Rank candidates against a job description."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates.jsonl or candidates.jsonl.gz",
    )
    parser.add_argument(
        "--jd", required=True,
        help="Path to job_description.docx",
    )
    parser.add_argument(
        "--out", required=True,
        help="Path for output submission.csv",
    )
    args = parser.parse_args()

    if not os.path.exists(args.jd):
        log.error("Job description not found: %s", args.jd)
        sys.exit(1)
    if not os.path.exists(args.candidates):
        log.error("Candidates file not found: %s", args.candidates)
        sys.exit(1)

    return args


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrates the fully-dynamic two-stage ranking pipeline.

    Stage A — Parse JD + embed + stream + hard-filter:
        Parse .docx → extract JD skills, experience, locations, education,
        certs (JDParser). Embed JD. Stream 100K candidates, apply schema
        validation and hard heuristic filters.

    Stage B — Batch mismatch detection (data-driven):
        TF-IDF coherence analysis flags profiles where headline domain
        diverges from career description domain.

    Stage C — Non-semantic pre-rank + shortlist:
        Compute fast non-semantic scores for ALL valid candidates using the
        JDProfile (dynamic parameters), sort, take top _SHORTLIST_SIZE (2000).

    Stage D — Embed shortlist + hybrid score:
        Batch-encode shortlisted candidate texts (full text, no caps),
        compute cosine similarity vs JD, combine with non-semantic score,
        apply soft penalties. Uses vectorized numpy ops.

    Stage E — Output:
        Sort by score, write top 100 to submission.csv.
    """
    wall_start = time.perf_counter()
    args = _parse_args()

    # ── Stage A: JD ─────────────────────────────────────────────────────────
    log.info("Loading job description: %s", args.jd)
    jd_text = load_docx_text(args.jd)

    log.info("Parsing JD for dynamic signals...")
    jd_profile = JDParser.parse(jd_text)
    log.info(
        "JD parsed: %d required skills, %d preferred, ideal_years=%.1f, "
        "locations=%s, remote_ok=%s",
        len(jd_profile.required_skills),
        len(jd_profile.preferred_skills),
        jd_profile.ideal_years,
        jd_profile.preferred_locations,
        jd_profile.remote_ok,
    )

    log.info("Embedding job description...")
    jd_vector = get_sentence_embeddings([jd_text])[0]

    # ── Stage A (cont.): Stream + hard-filter ────────────────────────────────
    log.info("Streaming candidates from: %s", args.candidates)
    valid_candidates: list[dict[str, Any]] = []

    total_count = 0
    invalid_count = 0
    filtered_count = 0

    for cand in stream_candidates(args.candidates):
        total_count += 1

        is_valid, reason = validate_candidate(cand)
        if not is_valid:
            invalid_count += 1
            if invalid_count <= 10:
                cid = cand.get("candidate_id", "<unknown>") if isinstance(cand, dict) else "<non-dict>"
                log.warning("  [INVALID] %s — %s", cid, reason)
            continue

        if check_honeypots_and_filters(cand):
            filtered_count += 1
            continue

        valid_candidates.append(cand)

        if total_count % 10_000 == 0:
            log.info(
                "  Progress: %d processed | %d invalid | %d filtered | %d valid",
                total_count, invalid_count, filtered_count, len(valid_candidates),
            )

    log.info(
        "Streaming complete: %d total | %d invalid | %d filtered | %d valid",
        total_count, invalid_count, filtered_count, len(valid_candidates),
    )

    if not valid_candidates:
        log.error("No valid candidates after filtering. Exiting.")
        sys.exit(1)

    # ── Stage B: Batch mismatch detection (TF-IDF) ───────────────────────────
    log.info("Running data-driven mismatch detection on %d candidates...", len(valid_candidates))
    t0 = time.perf_counter()
    detector = MismatchDetector()
    mismatched_ids = detector.detect(valid_candidates)
    mismatch_count = len(mismatched_ids)
    filtered_count += mismatch_count

    if mismatched_ids:
        valid_candidates = [
            c for c in valid_candidates
            if c["candidate_id"] not in mismatched_ids
        ]
    log.info(
        "Mismatch detection complete in %.1f s: %d flagged, %d remaining",
        time.perf_counter() - t0, mismatch_count, len(valid_candidates),
    )

    # ── Stage C: Non-semantic pre-rank + shortlist ────────────────────────────
    log.info("Computing non-semantic scores for %d candidates...", len(valid_candidates))
    t0 = time.perf_counter()

    ns_scores: list[float] = []
    sp_scores: list[float] = []
    for cand in valid_candidates:
        ns_scores.append(float(compute_non_semantic_score(cand, jd_profile)))
        sp_scores.append(float(compute_soft_penalty(cand)))

    # Sort by penalised non-semantic score descending, take top N
    ns_penalised = [ns * sp for ns, sp in zip(ns_scores, sp_scores)]
    ranked_indices = sorted(
        range(len(valid_candidates)),
        key=lambda i: ns_penalised[i],
        reverse=True,
    )
    shortlist_indices = ranked_indices[:_SHORTLIST_SIZE]

    shortlist_candidates = [valid_candidates[i] for i in shortlist_indices]
    shortlist_ns = np.array([ns_scores[i] for i in shortlist_indices])
    shortlist_sp = np.array([sp_scores[i] for i in shortlist_indices])

    log.info(
        "Pre-rank complete in %.1f s: shortlisted %d / %d candidates",
        time.perf_counter() - t0, len(shortlist_candidates), len(valid_candidates),
    )

    # ── Stage D: Embed shortlist + hybrid score ──────────────────────────────
    log.info("Extracting text for %d shortlisted candidates...", len(shortlist_candidates))
    candidate_texts = [extract_candidate_text(c) for c in shortlist_candidates]

    log.info("Encoding %d candidate profiles (batched)...", len(candidate_texts))
    t0 = time.perf_counter()
    cand_vectors = get_sentence_embeddings(candidate_texts)
    log.info("Encoding complete in %.1f s", time.perf_counter() - t0)

    # Vectorized cosine similarity (all candidates at once)
    jd_norm = np.linalg.norm(jd_vector)
    cand_norms = np.linalg.norm(cand_vectors, axis=1)
    # Avoid division by zero
    safe_denom = jd_norm * cand_norms
    safe_denom[safe_denom == 0] = 1.0
    sem_scores = np.dot(cand_vectors, jd_vector) / safe_denom

    # Hybrid score: 0.45 * semantic + 0.55 * (non_semantic * soft_penalty)
    ns_penalised_arr = shortlist_ns * shortlist_sp
    total_scores = _SEMANTIC_WEIGHT * sem_scores + _NON_SEMANTIC_WEIGHT * ns_penalised_arr

    log.info("Computing reasoning for top candidates...")
    results: list[dict[str, Any]] = []
    for i, cand in enumerate(shortlist_candidates):
        reasoning = select_reasoning(
            cand, float(sem_scores[i]), float(total_scores[i]), jd_profile,
        )
        results.append({
            "candidate_id": cand["candidate_id"],
            "score": float(total_scores[i]),
            "reasoning": reasoning,
        })

    # ── Stage E: Sort + write ────────────────────────────────────────────────
    for entry in results:
        entry["score"] = round(entry["score"], 4)
    results.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    top_100 = results[:100]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    log.info("Writing top 100 to: %s", args.out)

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, entry in enumerate(top_100, start=1):
            writer.writerow([
                entry["candidate_id"],
                rank,
                entry["score"],
                entry["reasoning"],
            ])

    elapsed = time.perf_counter() - wall_start
    log.info("Done. Total wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()

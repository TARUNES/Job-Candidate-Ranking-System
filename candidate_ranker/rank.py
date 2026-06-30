"""
rank.py
=======
CLI entry-point for the Intelligent Candidate Discovery & Ranking System.

Optimized two-stage pipeline
    1.  Parse CLI arguments
    2.  Load and embed the Job Description (JD)
    3.  Stream candidate records from JSONL or JSONL.GZ
    4.  Schema-validate each record  — skip malformed ones
    5.  Hard-filter via heuristics   — skip disqualified ones
    6.  Batch TF-IDF mismatch detection — data-driven honeypot filter
    7.  Pre-rank ALL valid candidates by non-semantic score (instant)
    8.  Shortlist top-N by non-semantic score (default 2000)
    9.  Batch-encode only the shortlisted candidates
   10.  Compute hybrid score per shortlisted candidate:
            final = 0.45 * semantic + 0.55 * (non_semantic * soft_penalty)
   11.  Sort descending by score, emit top 100 to submission.csv

Usage
    python rank.py \
        --candidates path/to/candidates.jsonl(.gz) \
        --jd         path/to/job_description.docx \
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
import re
import sys
import time
from typing import Any

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_docx_text, stream_candidates
from src.embeddings import get_sentence_embeddings
from src.heuristics import MismatchDetector, check_honeypots_and_filters, compute_soft_penalty
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
_MAX_CANDIDATE_TEXT_LEN: int = 500   # Truncate candidate text for faster encoding


# ---------------------------------------------------------------------------
# JD stop-words (filtered out during keyword extraction)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with",
    "is", "are", "be", "on", "at", "by", "we", "you", "will", "our",
    "have", "has", "that", "this", "their", "from", "as", "not", "but",
    "can", "your", "it", "its", "about", "more", "than", "all", "any",
    "who", "what", "how", "when", "where", "which", "such", "also",
    "must", "should", "would", "may", "each", "both", "very", "just",
})

_MIN_KEYWORD_LEN: int = 3


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_candidate_text(candidate: dict[str, Any]) -> str:
    """
    Builds a compact text string from a candidate record for semantic embedding.

    Optimized for speed: limits to 2 career roles, 8 skills, and truncates
    the final text to _MAX_CANDIDATE_TEXT_LEN characters.  The model's
    tokenizer will further truncate, but shorter inputs encode faster.

    Includes (in order):
      - Headline and professional summary
      - Descriptions from the 2 most recent career roles (with title prefix)
      - Top 8 skills formatted as "SkillName (proficiency)"
      - Education field of study
      - Certification names
    """
    profile = candidate["profile"]
    headline: str = profile["headline"]
    summary: str = profile["summary"]

    history_parts: list[str] = []
    for job in candidate.get("career_history", [])[:2]:  # was 3
        title = job.get("title", "")
        desc = job.get("description", "")
        if desc:
            history_parts.append(f"{title}: {desc}")

    skill_parts: list[str] = [
        f"{s['name']} ({s['proficiency']})"
        for s in candidate.get("skills", [])[:8]  # was unlimited
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

    sections = [
        headline,
        summary,
        " ".join(history_parts),
        "Skills: " + ", ".join(skill_parts) if skill_parts else "",
        "Education: " + ". ".join(edu_parts) if edu_parts else "",
        "Certifications: " + ". ".join(cert_parts) if cert_parts else "",
    ]
    text = " ".join(s for s in sections if s).strip()
    return text[:_MAX_CANDIDATE_TEXT_LEN]


# ---------------------------------------------------------------------------
# JD keyword extraction (used only for reasoning generation)
# ---------------------------------------------------------------------------

def extract_jd_keywords(jd_text: str) -> frozenset[str]:
    """
    Extracts a deduplicated set of meaningful tokens from the JD text.

    Tokens shorter than _MIN_KEYWORD_LEN or in _STOP_WORDS are discarded.
    The returned set is lowercased for case-insensitive matching against
    candidate skill names in select_reasoning().
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#\-\.]*", jd_text)
    keywords: set[str] = set()
    for tok in tokens:
        lower = tok.lower()
        if len(lower) >= _MIN_KEYWORD_LEN and lower not in _STOP_WORDS:
            keywords.add(lower)
    log.info("JD keyword extraction: %d unique tokens", len(keywords))
    return frozenset(keywords)


# ---------------------------------------------------------------------------
# Reasoning generation
# ---------------------------------------------------------------------------

def select_reasoning(
    candidate: dict[str, Any],
    semantic_score: float,
    total_score: float,
    jd_keywords: frozenset[str],
) -> str:
    """
    Generates a 1–2 sentence reasoning string grounded in the candidate's
    actual profile data and the JD keyword set.

    Logic:
      1.  Find which candidate skills appear in the JD keyword set.
      2.  If none match, fall back to the top-3 skills by endorsement count.
      3.  Choose tone (strong / moderate / lower) based on total_score.
      4.  Note unavailability or relocation when relevant.

    The reasoning is intentionally data-driven so that:
      - Every claim corresponds to something in the profile (no hallucination).
      - Tone is consistent with the numeric rank.
      - Different candidates produce substantively different text.
    """
    profile = candidate["profile"]
    signals = candidate["redrob_signals"]

    years: float = profile["years_of_experience"]
    title: str = profile["current_title"]
    location: str = profile["location"]
    open_to_work: bool = signals["open_to_work_flag"]
    willing_to_relocate: bool = signals["willing_to_relocate"]

    # Identify JD-relevant skills
    candidate_skills = [
        s["name"] for s in candidate.get("skills", []) if s.get("name")
    ]
    matched_skills = [s for s in candidate_skills if s.lower() in jd_keywords]

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
    Orchestrates the optimized two-stage ranking pipeline.

    Stage A — Load JD + stream + hard-filter:
        Parse .docx, embed JD, stream 100K candidates, apply schema
        validation and hard heuristic filters (timeline, impossible skills).

    Stage B — Batch mismatch detection (data-driven):
        TF-IDF coherence analysis flags profiles where headline domain
        diverges from career description domain.  No hardcoded pairs.

    Stage C — Non-semantic pre-rank + shortlist:
        Compute fast non-semantic scores for ALL valid candidates,
        sort, and take the top _SHORTLIST_SIZE (2000).  This is the
        key optimization: we only embed 2000 candidates instead of 94K.

    Stage D — Embed shortlist + hybrid score:
        Batch-encode shortlisted candidate texts, compute cosine
        similarity vs JD, combine with non-semantic score, apply
        soft penalties.  Uses vectorized numpy ops.

    Stage E — Output:
        Sort by score, write top 100 to submission.csv.
    """
    wall_start = time.perf_counter()
    args = _parse_args()

    # ── Stage A: JD ─────────────────────────────────────────────────────────
    log.info("Loading job description: %s", args.jd)
    jd_text = load_docx_text(args.jd)
    jd_keywords = extract_jd_keywords(jd_text)

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
        ns_scores.append(float(compute_non_semantic_score(cand)))
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
            cand, float(sem_scores[i]), float(total_scores[i]), jd_keywords,
        )
        results.append({
            "candidate_id": cand["candidate_id"],
            "score": float(total_scores[i]),
            "reasoning": reasoning,
        })

    # ── Stage E: Sort + write ────────────────────────────────────────────────
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
                round(entry["score"], 4),
                entry["reasoning"],
            ])

    elapsed = time.perf_counter() - wall_start
    log.info("Done. Total wall time: %.1f s", elapsed)


if __name__ == "__main__":
    main()

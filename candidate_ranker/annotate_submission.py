import csv
import json
import os
import sys
import argparse
from pathlib import Path
from src.data_loader import stream_candidates

def main():
    parser = argparse.ArgumentParser(description="Annotate ranked candidates with detailed profiles.")
    parser.add_argument("--submission", default="submission.csv", help="Path to the input submission CSV.")
    parser.add_argument("--out", default="submission_detailed.csv", help="Path to the output detailed CSV.")
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parent
    submission_path = Path(args.submission)
    if not submission_path.is_absolute():
        submission_path = workspace_dir / submission_path

    candidates_path = workspace_dir / ".." / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "candidates.jsonl"
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = workspace_dir / output_path

    if not submission_path.exists():
        print(f"Error: submission.csv not found at {submission_path}")
        sys.exit(1)
    if not candidates_path.exists():
        # Check if the .gz version exists
        candidates_path_gz = candidates_path.with_name("candidates.jsonl.gz")
        if candidates_path_gz.exists():
            candidates_path = candidates_path_gz
        else:
            print(f"Error: candidates file not found at {candidates_path}")
            sys.exit(1)

    print(f"Reading candidates from: {submission_path}")
    ranked_candidates = []
    candidate_ids = set()
    with open(submission_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranked_candidates.append(row)
            candidate_ids.add(row["candidate_id"])

    print(f"Scanning candidates database ({candidates_path.name}) for matches...")
    candidate_details = {}
    for cand in stream_candidates(str(candidates_path)):
        cid = cand.get("candidate_id")
        if cid in candidate_ids:
            candidate_details[cid] = cand
            if len(candidate_details) == len(candidate_ids):
                break

    print(f"Matched {len(candidate_details)} candidates. Appending factors...")

    # Define the output columns
    # Start with the original columns
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    
    # Add profile fields
    profile_fields = [
        "anonymized_name", "headline", "summary", "location", "country", 
        "years_of_experience", "current_title", "current_company", 
        "current_company_size", "current_industry"
    ]
    fieldnames.extend([f"profile_{f}" for f in profile_fields])

    # Add simplified summary fields for lists
    fieldnames.extend(["skills_list", "career_history_summary", "education_summary", "certifications_summary"])

    # Add redrob signals fields
    signal_fields = [
        "profile_completeness_score", "signup_date", "last_active_date", 
        "open_to_work_flag", "profile_views_received_30d", "applications_submitted_30d", 
        "recruiter_response_rate", "avg_response_time_hours", "connection_count", 
        "endorsements_received", "notice_period_days", "expected_salary_min", 
        "expected_salary_max", "preferred_work_mode", "willing_to_relocate", 
        "github_activity_score", "search_appearance_30d", "saved_by_recruiters_30d", 
        "interview_completion_rate", "offer_acceptance_rate", "verified_email", 
        "verified_phone", "linkedin_connected"
    ]
    fieldnames.extend([f"signal_{f}" for f in signal_fields])

    detailed_rows = []
    for row in ranked_candidates:
        cid = row["candidate_id"]
        cand = candidate_details.get(cid, {})
        new_row = row.copy()

        if cand:
            # Profile
            profile = cand.get("profile", {})
            for f in profile_fields:
                new_row[f"profile_{f}"] = profile.get(f, "")

            # Skills
            skills = cand.get("skills", [])
            skills_str = ", ".join([f"{s.get('name')} ({s.get('proficiency')})" for s in skills if s.get("name")])
            new_row["skills_list"] = skills_str

            # Career history summary
            career = cand.get("career_history", [])
            career_summary = " | ".join([
                f"{j.get('title')} at {j.get('company')} ({j.get('duration_months', 0)} mos)" 
                for j in career
            ])
            new_row["career_history_summary"] = career_summary

            # Education summary
            education = cand.get("education", [])
            edu_summary = " | ".join([
                f"{e.get('degree')} in {e.get('field_of_study')} from {e.get('institution')}" 
                for e in education
            ])
            new_row["education_summary"] = edu_summary

            # Certifications summary
            certs = cand.get("certifications", [])
            certs_summary = ", ".join([c.get("name") for c in certs if c.get("name")])
            new_row["certifications_summary"] = certs_summary

            # Redrob signals
            signals = cand.get("redrob_signals", {})
            for f in signal_fields:
                if f == "expected_salary_min":
                    new_row[f"signal_{f}"] = signals.get("expected_salary_range_inr_lpa", {}).get("min", "")
                elif f == "expected_salary_max":
                    new_row[f"signal_{f}"] = signals.get("expected_salary_range_inr_lpa", {}).get("max", "")
                else:
                    new_row[f"signal_{f}"] = signals.get(f, "")
        
        detailed_rows.append(new_row)

    print(f"Writing detailed submission output to: {output_path}")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detailed_rows)

    print("Successfully completed!")

if __name__ == "__main__":
    main()

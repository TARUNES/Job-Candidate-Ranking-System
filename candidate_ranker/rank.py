import os
import sys
import csv
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_docx_text, stream_candidates
from src.heuristics import check_honeypots_and_filters
from src.scoring import compute_non_semantic_score
from src.embeddings import get_sentence_embeddings, compute_cosine_similarity

def extract_candidate_text(candidate):
    """
    Constructs a clean textual summary from a candidate's profile and recent job descriptions.
    """
    profile = candidate.get('profile', {})
    headline = profile.get('headline', '')
    summary = profile.get('summary', '')
    history_texts = []
    career_history = candidate.get('career_history', [])
    for job in career_history[:2]:
        desc = job.get('description', '')
        title = job.get('title', '')
        if desc:
            history_texts.append(f"{title}: {desc}")
    combined_history = " ".join(history_texts)
    return f"{headline}. {summary}. {combined_history}"

def select_reasoning(candidate, semantic_score, total_score):
    """
    Formulates a descriptive 1-2 sentence justification for why the candidate matches the JD parameters.
    """
    profile = candidate.get('profile', {})
    years = profile.get('years_of_experience', 0.0)
    title = profile.get('current_title', 'Engineer')
    location = profile.get('location', '')
    skills_list = [s.get('name', '') for s in candidate.get('skills', [])[:3]]
    skills_str = ", ".join(skills_list)
    if semantic_score > 0.5:
        return f"{title} with {years} YOE based in {location}. Possesses expertise in {skills_str} matching the production AI systems focus."
    else:
        return f"Software professional with {years} YOE working with {skills_str}. Alignment is weaker on the modern production ML stack."

def main():
    """
    Main orchestrator that handles argument parsing, runs the filtering, embedding comparison, hybrid scoring, and outputs the top 100 candidates.
    """
    parser = argparse.ArgumentParser(description="Rank candidates against a job description.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--jd", required=True, help="Path to job_description.docx")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    args = parser.parse_args()
    if not os.path.exists(args.jd):
        print(f"Error: Job description file {args.jd} not found.")
        sys.exit(1)
    if not os.path.exists(args.candidates):
        print(f"Error: Candidates file {args.candidates} not found.")
        sys.exit(1)
    print("Loading and parsing job description docx...")
    jd_text = load_docx_text(args.jd)
    print("Computing job description semantic vector...")
    jd_vector = get_sentence_embeddings([jd_text])[0]
    print("Filtering candidates and preparing batch embeddings...")
    valid_candidates = []
    candidate_texts = []
    processed_count = 0
    skipped_count = 0
    for cand in stream_candidates(args.candidates):
        processed_count += 1
        if check_honeypots_and_filters(cand):
            skipped_count += 1
            continue
        valid_candidates.append(cand)
        cand_text = extract_candidate_text(cand)
        candidate_texts.append(cand_text)
    print(f"Total processed: {processed_count}, Skipped/Filtered out: {skipped_count}, Valid: {len(valid_candidates)}")
    if not valid_candidates:
        print("No valid candidates after filtering. Exiting.")
        sys.exit(1)
    print("Calculating semantic vector representations for valid candidates in batches...")
    cand_vectors = get_sentence_embeddings(candidate_texts)
    print("Computing hybrid scores for all candidates...")
    final_list = []
    for i, cand in enumerate(valid_candidates):
        sem_score = float(compute_cosine_similarity(jd_vector, cand_vectors[i]))
        non_sem_score = float(compute_non_semantic_score(cand))
        total_score = float(0.4 * sem_score + 0.6 * non_sem_score)
        reasoning_str = select_reasoning(cand, sem_score, total_score)
        final_list.append({
            'candidate_id': cand['candidate_id'],
            'score': total_score,
            'semantic_score': sem_score,
            'reasoning': reasoning_str,
            'candidate_raw': cand
        })
    final_list.sort(key=lambda x: (-x['score'], x['candidate_id']))
    top_100 = final_list[:100]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"Writing top 100 candidates to {args.out}...")
    with open(args.out, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
        for idx, entry in enumerate(top_100):
            writer.writerow([entry['candidate_id'], idx + 1, round(entry['score'], 4), entry['reasoning']])
    print("Ranking and output generation complete.")

if __name__ == "__main__":
    main()

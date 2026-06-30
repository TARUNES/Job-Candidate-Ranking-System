import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

candidates_path = r"c:\candiateRanker\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
out_dir = r"c:\candiateRanker\candidate_ranker\src"
out_path = os.path.join(out_dir, "tfidf_vectorizer.joblib")

print("Loading candidates for TF-IDF fitting...")
candidates = []
with open(candidates_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            candidates.append(json.loads(line))

print(f"Loaded {len(candidates)} candidates. Extracting texts...")
identity_texts = []
evidence_texts = []

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

all_texts = identity_texts + evidence_texts

print("Fitting TfidfVectorizer...")
vectorizer = TfidfVectorizer(
    max_features=5_000,
    ngram_range=(1, 1),
    min_df=10,
    stop_words="english",
    sublinear_tf=True,
)
vectorizer.fit(all_texts)

print(f"Saving vectorizer to {out_path}...")
joblib.dump(vectorizer, out_path)
print("Done!")

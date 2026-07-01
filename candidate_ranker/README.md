# Intelligent Candidate Discovery and Ranking System

## Summary
An AI-powered candidate ranking pipeline designed to score and rank 100,000 synthetic candidate profiles against a job description. The system operates fully offline, running strictly on CPU constraints, and completes the entire process within 5 minutes. It utilizes a two-stage funnel approach, blending non-semantic heuristics with deep semantic embeddings to deliver highly accurate, justifiable rankings.

## Execution Guide

### Option 1: Docker Sandbox (Recommended)
The repository includes a self-contained Dockerfile that pre-caches all required model weights. This guarantees 100% offline functionality without manual environment setup.

```bash
docker-compose up -d --build
```
Once the container starts, access the interactive web sandbox at `http://localhost:8501`.

### Option 2: Local Execution
Requires Python 3.12+ and Poetry.

```bash
# 1. Install dependencies
poetry env use 3.12
poetry install

# 2. Run the interactive web UI
poetry run python app.py

# 3. Alternatively, run the CLI pipeline directly
poetry run python rank.py \
    --candidates path/to/candidates.jsonl \
    --jd path/to/job_description.docx \
    --out submission.csv
```

## Architecture and Pipeline Flow

1. **Document Parsing:** Unstructured job descriptions are parsed to extract core requirements.
2. **Data Streaming and Validation:** Candidate records are streamed sequentially and validated for structural integrity.
3. **Hard Filtering:** Profiles triggering heuristic honeypots (e.g., overlapping timelines, impossible skills) are dropped instantly.
4. **Mismatch Detection:** Coherence analysis filters profiles demonstrating domain mismatches between their headline and career history.
5. **Heuristic Pre-Rank:** A fast composite score is calculated based on experience curves, location hubs, and trust signals to shortlist the top 2,000 candidates.
6. **Semantic Encoding:** Text from the shortlisted profiles is batched and encoded into dense vectors.
7. **Hybrid Scoring:** Cosine similarity is computed against the job description and merged with the heuristic score.
8. **Justification Generation:** Grounded reasoning strings are generated for the final top 100 profiles based on extracted data.

## Design Decisions and Model Choices

### Model: all-MiniLM-L6-v2 (SentenceTransformer)
**Why:** Chosen for its optimal balance of speed and semantic accuracy on CPU hardware. It produces 384-dimensional vectors rapidly, allowing the system to embed 2,000 profiles in seconds and adhere strictly to the 5-minute limit.

### Model: Google Flan-T5-base
**Why:** Selected for zero-shot text structure parsing. It reliably extracts strict constraints (like minimum years of experience and core skills) from unstructured job descriptions without requiring external network API calls.

### Architecture: Two-Stage Funnel
**Why:** Running deep semantic encoding on 100,000 text profiles using purely CPU compute heavily exceeds the time limit. The first stage uses ultra-fast heuristic pre-ranking to discard 98% of the dataset, reserving computationally expensive transformer inference solely for the top 2,000 candidates.

### Algorithm: TF-IDF Mismatch Detection
**Why:** Provides a robust, data-driven approach to detect logically inconsistent profiles. It identifies vocabulary divergence between a candidate's stated headline and their actual career history descriptions without relying on brittle, hardcoded keyword rules.

### Implementation: Vectorized Cosine Similarity
**Why:** Implemented using NumPy matrix operations rather than iterative loops, drastically reducing the time required to calculate semantic alignment scores across the shortlisted candidate pool.

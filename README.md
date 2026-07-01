# Intelligent Candidate Discovery and Ranking System

This repository contains the candidate ranking pipeline developed for the Redrob Hackathon. It is designed to score and rank 100,000 synthetic candidate profiles against an AI/ML Engineer job description, running fully offline within 5 minutes on CPU constraints.

## Repository Structure

* **`submission_metadata.yaml`**: The filled metadata file specifying team identity, reproducibility parameters, and methodology declarations.
* **`candidate_ranker/`**: The codebase folder containing the pipeline script (`rank.py`), web server UI (`app.py`), configuration (`config.py`), and helper modules (`src/`).
* **`[PUB] India_runs_data_and_ai_challenge/`**: Folder containing the official datasets (`candidates.jsonl`), job description file, and validation tools provided by the organizers.

## Execution and Setup

To run, test, or build the Docker sandbox, navigate to the `candidate_ranker` folder:

```bash
cd candidate_ranker
```

Please refer to [candidate_ranker/README.md](file:///c:/D/Projects/Job-Candidate-Ranking-System/candidate_ranker/README.md) for the complete instructions on local execution, Docker container setup, design decisions, and scoring weights.

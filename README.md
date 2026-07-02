# Intelligent Candidate Discovery and Ranking System

This repository contains the candidate ranking pipeline developed for the Redrob Hackathon. It is designed to score and rank 100,000 synthetic candidate profiles against an AI/ML Engineer job description, running fully offline within 5 minutes on CPU constraints.

## Repository Structure

* **`submission_metadata.yaml`**: The filled metadata file specifying team identity, reproducibility parameters, and methodology declarations.
* **`candidate_ranker/`**: The codebase folder containing the pipeline script (`rank.py`), web server UI (`app.py`), configuration (`config.py`), and helper modules (`src/`).
* **`[PUB] India_runs_data_and_ai_challenge/`**: Folder containing the official datasets (`candidates.jsonl`), job description file, and validation tools provided by the organizers.

## Execution and Setup

You can run the pipeline or the Docker sandbox directly from the repository root (no directory change required).

### 1. Generate the Output CSV (CLI Pipeline)

Using **Poetry**:
```bash
# Install dependencies
poetry install --directory candidate_ranker

# Run the pipeline (single command)
poetry run --directory candidate_ranker python rank.py --candidates "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --jd "../[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx" --out submission.csv
```

Using **pip**:
```bash
# Install dependencies
pip install -r candidate_ranker/requirements.txt

# Run the pipeline (single command)
python candidate_ranker/rank.py --candidates "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl" --jd "[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/job_description.docx" --out candidate_ranker/submission.csv
```

### 2. Run the Interactive Web UI (Locally)

Using **Poetry**:
```bash
poetry run --directory candidate_ranker python app.py
```

Using **pip**:
```bash
python candidate_ranker/app.py
```

Once started, access the UI at `http://localhost:8501`.

### 3. Run with Docker (FastAPI Web UI Sandbox)

Using **Docker Compose**:
```bash
docker compose -f candidate_ranker/docker-compose.yml up -d --build
```

Using **Docker CLI**:
```bash
docker build -t candidate-ranker-sandbox:latest candidate_ranker
docker run -d -p 8501:8501 --name candidate_ranker_sandbox candidate-ranker-sandbox:latest
```

Once running, access the interactive web sandbox at `http://localhost:8501`.

Please refer to [candidate_ranker/README.md](file:///c:/D/Projects/Job-Candidate-Ranking-System/candidate_ranker/README.md) for more details on local execution, Docker container setup, design decisions, and scoring weights.

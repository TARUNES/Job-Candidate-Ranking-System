# Redrob Hackathon — Full Reference Doc

> **Challenge**: Intelligent Candidate Discovery & Ranking  
> **Dataset**: 100,000 candidate profiles (`.jsonl.gz`) + 1 Job Description (`.docx`)  
> **Goal**: Submit a ranked CSV of the top 100 candidates with reasoning

---

## 1. Problem Statement

You are given a pool of **100,000 synthetic candidate profiles** and a **single job description**. Your task is to build a ranking system that:

- Identifies the top 100 candidates best suited for the JD
- Ranks them best-fit first (rank 1 = best)
- Provides a 1–2 sentence reasoning per candidate
- Runs fully **offline** — no GPU, no external API calls, within 5 minutes, ≤16 GB RAM

The dataset contains **deliberate traps**:
- ~80 honeypot candidates with subtly impossible profiles
- Keyword stuffers (profiles padded with JD buzzwords but no depth)
- Behavioral twins (identical profiles with different IDs)
- Plain-language Tier 5s (look qualified on surface, fail on behavioral signals)

> **Disqualification rule**: If >10% of your top-100 are honeypots → auto-disqualified at Stage 3.

---

## 2. Job Description Summary

**Role**: AI/ML Engineer — Production Systems  
**Focus**: Building, deploying and maintaining production-grade AI/ML systems

Key requirements extracted from `job_description.docx`:

| Category | What the JD Wants |
|---|---|
| **Core Skills** | Python, PyTorch / TensorFlow, model serving (FastAPI, BentoML, TorchServe) |
| **ML Ops** | MLflow, Weights & Biases, Kubeflow, Airflow |
| **Data Infra** | Spark, Kafka, Airflow, dbt, Snowflake, BigQuery |
| **Vector / Search** | FAISS, Pinecone, Milvus, Weaviate, OpenSearch |
| **LLM / GenAI** | Fine-tuning LLMs, LoRA, PEFT, LangChain, Hugging Face |
| **Cloud** | AWS / GCP / Azure |
| **Experience** | ~5–10 years preferred, strong data-engineering-to-ML transition welcomed |
| **Location** | India (Tier-1 hubs preferred); relocation/remote candidates considered |

> **Participant note from JD**: "The evaluation is designed so that AI-assisted work where you did real engineering succeeds, while AI-only submissions fail at Stages 3–5."

---

## 3. Evaluation Pipeline (5 Stages)

```
Stage 1 ──► Format Validation      Auto-validator on every submission
Stage 2 ──► Scoring                NDCG@10, NDCG@50, MAP, P@10 computed on hidden ground truth
Stage 3 ──► Code Reproduction      Full repo + 5min/16GB/no-GPU/no-network sandbox run + honeypot check
Stage 4 ──► Manual Review          Reasoning quality, methodology, Git history authenticity
Stage 5 ──► Defend-Your-Work       30-min video call with Redrob engineering team
```

### Scoring Metrics

| Metric | Weight | Measures |
|---|---|---|
| **NDCG@10** | 50% | Quality of your top-10 picks |
| **NDCG@50** | 30% | Quality of your top-50 picks |
| **MAP** | 15% | Precision across all relevance levels |
| **P@10** | 5% | Fraction of top-10 that are "relevant" (tier 3+) |

### Reasoning Quality Checks (Stage 4)

| Check | Description |
|---|---|
| Specific facts | References actual data from the candidate profile |
| JD connection | Connects to specific JD requirements, not generic praise |
| Honest concerns | Acknowledges gaps where they exist |
| No hallucination | Every claim exists in the actual profile |
| Variation | 10 sampled reasonings are substantively different |
| Rank consistency | Tone matches the rank (rank-5 shouldn't have critical reasoning) |

---

## 4. Dataset Files

| File | Description |
|---|---|
| `candidates.jsonl` | Full 100,000-candidate pool (~487 MB uncompressed) |
| `sample_candidates.json` | First 50 candidates as pretty-printed JSON |
| `job_description.docx` | The JD you're ranking against |
| `candidate_schema.json` | JSON Schema for every field |
| `redrob_signals_doc.docx` | Reference for the 23 behavioral signals |
| `sample_submission.csv` | Format reference — **not** a quality ranking |
| `submission_metadata_template.yaml` | Metadata to fill and submit |
| `validate_submission.py` | Format validator — run before uploading |

---

## 5. Sample Candidates Dataset (All 50)

These are `CAND_0000001` through `CAND_0000050` from `sample_candidates.json`.

| ID | Title | YOE | Location | Country | Open? | Notice | Top Skills (first 5) |
|---|---|---|---|---|---|---|---|
| CAND_0000001 | Backend Engineer | 6.9 | Toronto | Canada | ✅ | 60d | Tailwind, NLP, Image Classification, Fine-tuning LLMs, Weights & Biases |
| CAND_0000002 | Operations Manager | 12.5 | Chennai | India | ✅ | 60d | Project Management, React, Photoshop, TypeScript, Marketing |
| CAND_0000003 | Customer Support | 1.1 | Austin | USA | ❌ | 150d | Angular, SEO, Excel, Accounting, Kubernetes |
| CAND_0000004 | Marketing Manager | 3.8 | Sydney | Australia | ❌ | 120d | Node.js, Content Writing, Redux, Airflow, GraphQL |
| CAND_0000005 | Accountant | 11.0 | Gurgaon | India | ✅ | 30d | SQL, PowerPoint, Photoshop, Tailwind, Apache Flink |
| CAND_0000006 | Business Analyst | 6.0 | Austin | USA | ❌ | 150d | Content Writing, SEO, Redux, SQL, Sales |
| CAND_0000007 | Civil Engineer | 5.5 | Gurgaon | India | ❌ | 30d | Content Writing, MongoDB, Sales, Spark, Scrum |
| CAND_0000008 | Operations Manager | 3.6 | Noida | India | ❌ | 90d | Java, BigQuery, Spark, Accounting, Kubernetes |
| CAND_0000009 | Mechanical Engineer | 11.0 | New York | USA | ❌ | 150d | Snowflake, gRPC, JavaScript, OpenCV, Go |
| CAND_0000010 | Data Engineer | 4.6 | London | UK | ❌ | 120d | GCP, Spring Boot, Kubeflow, Java, GANs |
| CAND_0000011 | QA Engineer | 2.0 | Hyderabad | India | ❌ | 90d | Recommendation Systems, Scrum, FastAPI, Hugging Face, AWS |
| CAND_0000012 | Operations Manager | 1.1 | Chandigarh | India | ❌ | 60d | Azure, Airflow, AWS, gRPC, Vue.js |
| CAND_0000013 | Civil Engineer | 1.1 | Dubai | UAE | ✅ | 30d | React, Redux, Vue.js, Six Sigma, Spring Boot |
| CAND_0000014 | Frontend Engineer | 8.4 | Hyderabad | India | ❌ | 90d | FAISS, BigQuery, React, OpenSearch, OpenCV |
| CAND_0000015 | Software Engineer | 5.4 | Trivandrum | India | ✅ | 90d | PyTorch, Content Writing, Weights & Biases, Qdrant, Sales |
| CAND_0000016 | Accountant | 5.3 | Gurgaon | India | ✅ | 60d | Node.js, Figma, Data Pipelines, Go, Photoshop |
| CAND_0000017 | Accountant | 12.3 | Bangalore | India | ❌ | 90d | Next.js, Java, Apache Flink, Sales, Tally |
| CAND_0000018 | Frontend Engineer | 6.6 | Bhubaneswar | India | ❌ | 120d | CNN, Java, Accounting, Data Pipelines, Node.js |
| CAND_0000019 | Project Manager | 6.5 | Trivandrum | India | ❌ | 60d | Figma, GraphQL, Six Sigma, Scrum, YOLO |
| CAND_0000020 | Mechanical Engineer | 6.3 | Ahmedabad | India | ❌ | 30d | GraphQL, TypeScript, Flask, Weights & Biases, GCP |
| CAND_0000021 | Project Manager | 14.5 | Bhubaneswar | India | ❌ | 60d | Hadoop, PostgreSQL, Kafka, Microservices, AWS |
| CAND_0000022 | Mechanical Engineer | 1.1 | Sydney | Australia | ✅ | 150d | OpenCV, Django, Terraform, Scrum, SQL |
| CAND_0000023 | Software Engineer | 3.7 | New York | USA | ❌ | 30d | BigQuery, Marketing, Node.js, Django, Salesforce CRM |
| CAND_0000024 | HR Manager | 7.5 | Trivandrum | India | ❌ | 60d | Figma, Kubernetes, Forecasting, ETL, Node.js |
| CAND_0000025 | Frontend Engineer | 7.3 | Vizag | India | ✅ | 120d | JavaScript, Spark, GCP, TypeScript, LangChain |
| CAND_0000026 | Graphic Designer | 6.8 | Kochi | India | ❌ | 30d | Apache Beam, Kubeflow, Scrum, ETL, Django |
| CAND_0000027 | DevOps Engineer | 3.9 | Kolkata | India | ✅ | 90d | Docker, YOLO, PEFT, Webpack, Data Science |
| CAND_0000028 | Operations Manager | 1.1 | Dubai | UAE | ❌ | 60d | Snowflake, React, JavaScript, Tailwind, REST APIs |
| CAND_0000029 | Business Analyst | 7.2 | Noida | India | ❌ | 60d | Node.js, Scrum, Tailwind, Hadoop, Spring Boot |
| CAND_0000030 | Marketing Manager | 10.0 | Kochi | India | ❌ | 60d | gRPC, Apache Beam, GraphQL, Java, Spring Boot |
| CAND_0000031 | Recommendation Systems Engineer | 6.0 | Hyderabad | India | ✅ | 60d | Go, MLflow, FAISS, Pinecone, Angular |
| CAND_0000032 | .NET Developer | 8.1 | Gurgaon | India | ❌ | 150d | Speech Recognition, Project Management, REST APIs, CSS, Embeddings |
| CAND_0000033 | Graphic Designer | 8.6 | Pune | India | ✅ | 30d | Kubernetes, Data Pipelines, Snowflake, CI/CD, SEO |
| CAND_0000034 | Business Analyst | 14.5 | Ahmedabad | India | ❌ | 90d | GraphQL, Excel, Node.js, Terraform, Salesforce CRM |
| CAND_0000035 | Full Stack Developer | 4.3 | Hyderabad | India | ❌ | 60d | Snowflake, BigQuery, Recommendation Systems, Data Pipelines, Docker |
| CAND_0000036 | Project Manager | 11.3 | Trivandrum | India | ✅ | 60d | Figma, MongoDB, PowerPoint, CSS, Excel |
| CAND_0000037 | Business Analyst | 14.3 | Dubai | UAE | ❌ | 30d | Databricks, Docker, Flask, AWS, Terraform |
| CAND_0000038 | Java Developer | 6.7 | Coimbatore | India | ✅ | 90d | Kubeflow, Django, Redux, Weaviate, PowerPoint |
| CAND_0000039 | Marketing Manager | 3.9 | Bhubaneswar | India | ❌ | 30d | Spark, Tailwind, Sales, CI/CD, Illustrator |
| CAND_0000040 | Customer Support | 1.6 | Kochi | India | ❌ | 90d | SQL, Spring Boot, Accounting, Rust, Redux |
| CAND_0000041 | Operations Manager | 13.7 | Delhi | India | ✅ | 90d | Airflow, SQL, Go, GCP, Figma |
| CAND_0000042 | HR Manager | 5.0 | Berlin | Germany | ❌ | 30d | Project Management, gRPC, Marketing, SAP, Illustrator |
| CAND_0000043 | Cloud Engineer | 8.3 | Chandigarh | India | ❌ | 120d | Elasticsearch, OpenSearch, Airflow, Kubeflow, Fine-tuning LLMs |
| CAND_0000044 | Frontend Engineer | 5.7 | Indore | India | ❌ | 90d | Hadoop, JavaScript, Databricks, Python, dbt |
| CAND_0000045 | Project Manager | 12.2 | Indore | India | ✅ | 60d | GCP, Sales, Redux, PostgreSQL, Airflow |
| CAND_0000046 | Mechanical Engineer | 7.8 | London | UK | ❌ | 30d | Agile, Scrum, SAP, React, Azure |
| CAND_0000047 | Project Manager | 2.4 | Kochi | India | ❌ | 90d | FastAPI, Java, Excel, Tally, SQL |
| CAND_0000048 | Mobile Developer | 9.7 | Hyderabad | India | ✅ | 120d | Hadoop, Terraform, Vue.js, Content Writing, AWS |
| CAND_0000049 | Mechanical Engineer | 11.8 | Berlin | Germany | ❌ | 120d | TypeScript, Rust, Data Pipelines, Apache Beam, GraphQL |
| CAND_0000050 | Business Analyst | 13.5 | Gurgaon | India | ❌ | 90d | gRPC, SEO, Feature Engineering, Marketing, Data Pipelines |

> **Observation**: The sample set is deliberately diverse — most candidates are NOT strong ML fits (Operations Managers, Civil Engineers, Accountants). Only a handful (CAND_0000001, 0000011, 0000015, 0000027, 0000031, 0000043) have meaningful ML/AI skills. This is intentional — your ranker must filter noise.

---

## 6. Sample Submission Output (format reference)

The provided `sample_submission.csv` has 100 rows. The **top 10 ranked candidates** in it are:

| Rank | Candidate ID | Score | Reasoning |
|---|---|---|---|
| 1 | CAND_0004989 | 0.9920 | HR Manager with 6.1 yrs; 9 AI core skills; response rate 0.76. |
| 2 | CAND_0001195 | 0.9840 | HR Manager with 8.7 yrs; 9 AI core skills; response rate 0.20. |
| 3 | CAND_0003114 | 0.9760 | ML Engineer with 6.4 yrs; 4 AI core skills; response rate 0.88. |
| 4 | CAND_0000339 | 0.9680 | Content Writer with 8.3 yrs; 8 AI core skills; response rate 0.72. |
| 5 | CAND_0001082 | 0.9600 | HR Manager with 5.0 yrs; 8 AI core skills; response rate 0.62. |
| 6 | CAND_0001218 | 0.9520 | Graphic Designer with 10.4 yrs; 9 AI core skills; response rate 0.56. |
| 7 | CAND_0004558 | 0.9440 | Business Analyst with 5.1 yrs; 8 AI core skills; response rate 0.54. |
| 8 | CAND_0001753 | 0.9360 | Content Writer with 8.3 yrs; 8 AI core skills; response rate 0.53. |
| 9 | CAND_0001503 | 0.9280 | Marketing Manager with 8.0 yrs; 8 AI core skills; response rate 0.32. |
| 10 | CAND_0004548 | 0.9200 | HR Manager with 7.3 yrs; 8 AI core skills; response rate 0.30. |

> [!WARNING]
> The sample submission is a **format reference only**, not a quality ranking. Notice it ranks an HR Manager #1 and a Content Writer #4 for an AI/ML role. Your ranker must do significantly better. The sample reasoning is also templated and identical in format — Stage 4 will flag this.

### Cross-reference: Sample Candidates vs Sample Submission

Only **2 of the 50 sample candidates** appear in the sample submission's top 100:

| ID | Title | YOE | Location | Rank in Sample | Score |
|---|---|---|---|---|---|
| CAND_0000002 | Operations Manager | 12.5 | Chennai | 14 | 0.8880 |
| CAND_0000007 | Civil Engineer | 5.5 | Gurgaon | 31 | 0.7520 |

---

## 7. Submission Format Specification

### CSV format (mandatory)
```csv
candidate_id,rank,score,reasoning
CAND_0001234,1,0.9312,"ML Engineer with 7 YOE in Bangalore. Strong match on PyTorch, fine-tuning LLMs, and MLflow; notice period of 30 days."
CAND_0005678,2,0.9101,"Data scientist with production MLOps background..."
```

### Rules
| Rule | Requirement |
|---|---|
| Rows | Exactly 100 rows (ranks 1–100, each used exactly once) |
| `score` | Float, monotonically non-increasing as rank increases |
| `candidate_id` | Must exist in `candidates.jsonl` |
| `reasoning` | Optional but **strongly recommended** (used at Stage 4) |
| Runtime | ≤ 5 min wall-clock, CPU only, no network |
| Memory | ≤ 16 GB RAM |
| Disk | ≤ 5 GB intermediate state |

---

## 8. Submission Metadata Checklist

Fill out `submission_metadata_template.yaml`. All ✅ fields are mandatory:

| Field | Required | Notes |
|---|---|---|
| `team_name` | ✅ | Used in leaderboard |
| `primary_contact.name` | ✅ | Point of contact |
| `primary_contact.email` | ✅ | All organizer communication |
| `primary_contact.phone` | ✅ | Used for top-N outreach |
| `team_members[]` | ✅ | Name + email for each member |
| `github_repo` | ✅ | Must be reachable (private OK if access granted at Stage 3) |
| `sandbox_link` | ✅ | Working hosted demo (HuggingFace Spaces, Streamlit Cloud, Replit, Colab, Docker, Binder) |
| `reproduce_command` | ✅ | Single command to produce submission.csv |
| `compute.platform` | ✅ | E.g. "MacBook Pro M2, 16GB RAM, Python 3.11" |
| `compute.uses_gpu_for_inference` | ✅ | Must be `false` |
| `compute.has_network_during_ranking` | ✅ | Must be `false` |
| `ai_tools_used[]` | ✅ | Declare honestly — not penalized |
| `declarations.read_submission_spec` | ✅ | Must be `true` |
| `declarations.code_is_original_work` | ✅ | Must be `true` |
| `declarations.no_collusion` | ✅ | Must be `true` |
| `declarations.reproduction_tested` | ✅ | Must be `true` |
| `methodology_summary` | ⚠️ Optional | ≤200 words — strongly recommended for Stage 4 |

### Sandbox Requirements
The sandbox must:
- Accept ≤100 candidates as input (upload or pre-loaded)
- Run the ranker end-to-end and produce a ranked CSV
- Complete within ≤5 min on CPU

---

## 9. Our Implementation Status

### Architecture (candidate_ranker)

```
candidates.jsonl
    │
    ▼
schema_validator.py   ← [NEW] validates all fields, skips malformed records
    │
    ▼
heuristics.py         ← Hard filters (consulting-only, timeline overlap, impossible skills, mismatch)
                         + compute_soft_penalty() for behavioural signals
    │
    ▼
rank.py               ← extract_candidate_text() with skills + edu + certs
                         extract_jd_keywords() for JD-aware reasoning
    │
    ▼
embeddings.py         ← SentenceTransformer('all-MiniLM-L6-v2'), singleton, batched
    │
    ▼
scoring.py            ← experience (15%) + notice (10%) + location (10%)
                         + platform_signals (45%) + trust_signals (20%)
    │
    ▼
rank.py               ← final = 0.45*semantic + 0.55*(non_semantic * soft_penalty)
                         Top 100 → submission.csv
```

### What We Have Built

| Component | Status | Notes |
|---|---|---|
| `src/schema_validator.py` | ✅ Done | Validates 5 subsections, graceful skip |
| `src/heuristics.py` | ✅ Done | Industry-field consulting filter, 5 mismatch archetypes, soft penalty |
| `src/scoring.py` | ✅ Done | 5 sub-scores, trust signals, relocation-aware location |
| `src/embeddings.py` | ✅ Done | Singleton SentenceTransformer, batch encode |
| `src/data_loader.py` | ✅ Done | Streaming JSONL + .gz support |
| `rank.py` | ✅ Done | Full pipeline, JD keyword extraction, JD-aware reasoning |

### What Is Still Needed (Submission Blockers)

| Item | Status | Action Required |
|---|---|---|
| `submission.csv` | ❌ Not generated | Run `rank.py` on full `candidates.jsonl` |
| `submission_metadata.yaml` | ❌ Not filled | Fill team name, contact, compute env |
| `github_repo` | ❌ Needed | Push code to GitHub |
| `sandbox_link` | ❌ Needed | Deploy to HuggingFace Spaces / Streamlit Cloud |
| `validate_submission.py` run | ❌ Not run | Run after generating submission.csv |
| `methodology_summary` | ❌ Not written | Write ≤200 word summary |
| AI tools declaration | ❌ Not filled | Declare: Antigravity / Gemini, used for arch + code |

---

## 10. Run Commands

### Install dependencies
```bash
cd candidate_ranker
pip install sentence-transformers pandas numpy jsonlines python-docx
```
> First run downloads `all-MiniLM-L6-v2` (~85MB) to `~/.cache/huggingface/` automatically.

### Generate submission (PowerShell)
```powershell
cd c:\D\Projects\Job-Candidate-Ranking-System\candidate_ranker

python rank.py `
  --candidates "..\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" `
  --jd "..\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\job_description.docx" `
  --out ".\output\submission.csv"
```

### Validate submission
```powershell
python "..\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py" `
  --submission ".\output\submission.csv"
```

### Expected console output
```
Loading and parsing job description (.docx)...
  JD keywords extracted: ~350
Computing job description semantic vector...
Streaming, validating and filtering candidates...
  Total records : 100000
  Schema invalid: N
  Hard-filtered : ~XXXX
  Valid / scored: ~XXXXX
Encoding candidate profiles (batched)...
Computing hybrid scores...
Writing top 100 candidates to .\output\submission.csv...
Done. Ranking and output generation complete.
```

---

## 11. Three-Submission Budget

| Submission | Status | Notes |
|---|---|---|
| #1 | — | First run — validate format, check NDCG |
| #2 | — | Tuning run after reviewing scores |
| #3 | — | Final |

> **Only your last valid submission counts.**

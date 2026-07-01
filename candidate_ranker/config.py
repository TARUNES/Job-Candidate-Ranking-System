# Uvicorn Server config
API_HOST = "0.0.0.0"
API_PORT = 8501

# Model Names
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PARSING_MODEL_NAME = "google/flan-t5-base"

# Pipeline Weights and Sizes
SHORTLIST_SIZE = 2000
SEMANTIC_WEIGHT = 0.45
NON_SEMANTIC_WEIGHT = 0.55
MAX_CANDIDATE_TEXT_LEN = 600

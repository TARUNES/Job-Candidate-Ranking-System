"""
jd_parser.py
============
Dynamically parses a Job Description (JD) text document and extracts all
structured signals needed to score candidates against it.

Primary extraction strategy — Local LLM (flan-t5-small)
    Uses Google's Flan-T5-small model (~300 MB, CPU-only) via the
    HuggingFace Transformers library.  The model is loaded once as a
    singleton and cached for the lifetime of the process, so it adds
    roughly 1–2 seconds of one-time load overhead and <1 second per
    prompt at inference time.

    Why flan-t5-small?
      - Instruction-tuned: responds accurately to "Extract X from Y" prompts.
      - Tiny footprint: well within the hackathon's 16 GB RAM constraint.
      - CPU-only inference: no GPU required.
      - Zero network calls at inference time: only an initial one-time model
        download to the HuggingFace cache (~/.cache/huggingface/).

Fallback strategy — Regex heuristics
    If the LLM produces an empty or implausible result for any field, the
    parser automatically falls back to a regex/vocabulary-based heuristic.
    This makes the parser robust against edge-case JD formats.

Extracted signals (returned as a JDProfile dataclass)
------------------------------------------------------
  required_skills     list[str]   Explicit required skill names
  preferred_skills    list[str]   Nice-to-have skill names
  min_years           float       Minimum years of experience
  max_years           float       Maximum years of experience
  ideal_years         float       Midpoint / target years of experience
  preferred_locations list[str]   City / country names from the JD
  remote_ok           bool        True if JD allows remote or hybrid work
  required_degrees    list[str]   Degree abbreviations (B.Tech, M.Tech …)
  required_fields     list[str]   Fields of study (Computer Science, AI …)
  required_certs      list[str]   Named certifications mentioned in the JD
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM singleton — loaded once, reused across all extraction calls
# ---------------------------------------------------------------------------

class _LLMSingleton:
    """
    Lazy-loading singleton wrapper around flan-t5-small.

    The model and tokeniser are loaded on first use and kept alive in
    process memory for all subsequent calls.  Using a class-level cache
    avoids the ~1 s load penalty for every extraction call.
    """

    _model = None
    _tokeniser = None
    _available: Optional[bool] = None  # None = not yet checked

    @classmethod
    def _load(cls) -> bool:
        """
        Attempts to load flan-t5-small.  Returns True on success, False
        if the model is unavailable (e.g. missing cache + no network).
        Sets cls._available so subsequent calls short-circuit immediately.
        """
        if cls._available is not None:
            return cls._available

        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            import config
            model_name = config.PARSING_MODEL_NAME
            log.info("Loading local LLM for JD parsing: %s", model_name)
            cls._tokeniser = T5Tokenizer.from_pretrained(model_name)
            cls._model = T5ForConditionalGeneration.from_pretrained(model_name)
            cls._model.eval()
            cls._available = True
            log.info("LLM loaded successfully.")
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "flan-t5-small unavailable (%s). JD parsing will use regex fallback.",
                exc,
            )
            cls._available = False

        return cls._available

    @classmethod
    def generate(cls, prompt: str, max_new_tokens: int = 128) -> str:
        """
        Runs inference on the loaded model.

        Parameters
        ----------
        prompt          : The instruction prompt to send to the model.
        max_new_tokens  : Maximum tokens to generate in the response.

        Returns
        -------
        The decoded model output string, or an empty string if the model
        is unavailable.
        """
        if not cls._load():
            return ""

        try:
            import torch
            inputs = cls._tokeniser(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                output_ids = cls._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=2,
                    early_stopping=True,
                )
            return cls._tokeniser.decode(output_ids[0], skip_special_tokens=True).strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM inference error: %s. Returning empty string.", exc)
            return ""


# ---------------------------------------------------------------------------
# Vocabulary sets used by the regex fallback
# ---------------------------------------------------------------------------

# Comprehensive city / metro list (India-focused + global tech hubs)
_CITIES: frozenset[str] = frozenset({
    # India — Tier 1
    "bangalore", "bengaluru", "mumbai", "delhi", "ncr", "noida", "gurgaon",
    "gurugram", "hyderabad", "chennai", "pune", "kolkata",
    # India — Tier 2
    "ahmedabad", "jaipur", "indore", "bhopal", "nagpur", "surat", "vadodara",
    "coimbatore", "kochi", "cochin", "trivandrum", "thiruvananthapuram",
    "chandigarh", "lucknow", "kanpur", "patna", "bhubaneswar", "vizag",
    "visakhapatnam", "mangalore", "mysore", "mysuru", "rajkot", "nashik",
    "aurangabad",
    # Global major tech hubs
    "san francisco", "new york", "london", "berlin", "singapore", "dubai",
    "amsterdam", "toronto", "sydney", "melbourne", "tokyo", "seattle",
    "austin", "boston", "chicago", "paris", "zurich", "stockholm",
    # Countries
    "india", "usa", "uk", "germany", "canada", "australia",
})

# Known tech skill tokens for the regex fallback
_TECH_VOCAB: frozenset[str] = frozenset({
    "python", "java", "scala", "go", "rust", "c", "c++", "c#", "typescript",
    "javascript", "sql", "nosql", "bash", "r",
    "pytorch", "tensorflow", "keras", "jax", "huggingface", "transformers",
    "bert", "gpt", "llm", "llms", "rag", "langchain", "llamaindex",
    "openai", "anthropic", "gemini", "mistral", "llama",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "chroma",
    "opensearch", "elasticsearch", "bm25",
    "lora", "qlora", "peft", "rlhf", "dpo", "sft",
    "mlflow", "wandb", "dvc",
    "docker", "kubernetes", "k8s", "airflow", "kubeflow",
    "aws", "gcp", "azure", "s3", "ec2", "lambda",
    "spark", "kafka", "flink", "databricks", "snowflake", "redshift",
    "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "react", "node", "flask", "fastapi", "django",
    "nlp", "cv", "embeddings", "embedding",
    "recommendation", "ranking", "retrieval", "inference", "fine-tuning",
    "finetuning", "pretraining", "training", "serving",
    "ray", "triton", "onnx", "tensorrt",
    "git", "ci", "cd", "github", "gitlab", "jenkins",
    "sklearn", "scikit", "scikit-learn", "xgboost", "lightgbm", "catboost",
    "pandas", "numpy", "scipy", "matplotlib", "plotly",
    "bentoml", "seldon", "torchserve", "sagemaker",
    "dbt", "prefect", "luigi",
    "re-ranking", "embeddings-based", "vector-search",
    "learning-to-rank", "hybrid-search", "text-to-speech", "speech-to-text",
})

# Common stop-words filtered during regex skill extraction
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for", "with",
    "is", "are", "be", "on", "at", "by", "we", "you", "will", "our",
    "have", "has", "that", "this", "their", "from", "as", "not", "but",
    "can", "your", "it", "its", "about", "more", "than", "all", "any",
    "who", "what", "how", "when", "where", "which", "such", "also",
    "must", "should", "would", "may", "each", "both", "very", "just",
    "using", "experience", "knowledge", "understanding", "ability",
    "strong", "good", "excellent", "work", "team", "role", "position",
    "skills", "skill", "tools", "tool", "system", "systems", "platform",
    "candidate", "candidates", "job", "description", "apply",
})

# Degree patterns for the regex fallback
_DEGREE_PATTERNS: list[str] = [
    r"\bPh\.?D\.?\b", r"\bPhD\b", r"\bDoctor(?:ate)?\b",
    r"\bM\.?S\.?\b", r"\bM\.?E\.?\b", r"\bM\.?Tech\.?\b", r"\bMasters?\b",
    r"\bM\.?Sc\.?\b", r"\bM\.?B\.?A\.?\b",
    r"\bB\.?Tech\.?\b", r"\bB\.?E\.?\b", r"\bB\.?S\.?\b", r"\bB\.?Sc\.?\b",
    r"\bBachelors?\b",
]

# Preferred / nice-to-have section header patterns
_PREFERRED_SECTION_RE = re.compile(
    r"preferred|nice.to.have|good.to.have|bonus|plus|advantageous|desirable",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# JD Profile dataclass
# ---------------------------------------------------------------------------

@dataclass
class JDProfile:
    """
    Structured representation of a parsed Job Description.

    All fields are populated by JDParser.parse() and consumed by
    scoring.py and rank.py.  The class exposes several convenience
    properties that downstream code uses for matching.
    """

    raw_text: str = ""
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_years: float = 0.0
    max_years: float = 15.0
    ideal_years: float = 5.0
    preferred_locations: list[str] = field(default_factory=list)
    remote_ok: bool = False
    required_degrees: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    required_certs: list[str] = field(default_factory=list)

    @property
    def all_skills(self) -> list[str]:
        """Combined required + preferred skills (required first, deduplicated)."""
        seen: set[str] = set()
        out: list[str] = []
        for s in self.required_skills + self.preferred_skills:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    @property
    def all_skills_lower(self) -> frozenset[str]:
        """Lowercase frozenset of all skills for O(1) lookup."""
        return frozenset(s.lower() for s in self.all_skills)

    @property
    def required_skills_lower(self) -> frozenset[str]:
        """Lowercase frozenset of required skills for O(1) lookup."""
        return frozenset(s.lower() for s in self.required_skills)

    @property
    def preferred_locations_lower(self) -> frozenset[str]:
        """Lowercase frozenset of preferred locations for O(1) lookup."""
        return frozenset(loc.lower() for loc in self.preferred_locations)


# ---------------------------------------------------------------------------
# JD Parser
# ---------------------------------------------------------------------------

class JDParser:
    """
    Parses a raw JD text string into a JDProfile.

    Extraction hierarchy for each field:
      1. LLM (flan-t5-small) — primary, semantically aware
      2. Regex / vocabulary heuristics — fallback if LLM result is empty
         or implausible

    Usage
    -----
      jd_text = load_docx_text(path)
      jd_profile = JDParser.parse(jd_text)
    """

    # Minimum character length for a skill token to be considered valid
    _MIN_SKILL_LEN: int = 2

    # Fields of study vocabulary for the regex fallback
    _FIELD_VOCAB: list[str] = [
        "Computer Science", "Computer Engineering", "Software Engineering",
        "Information Technology", "Information Systems",
        "Electrical Engineering", "Electronics",
        "Data Science", "Artificial Intelligence", "Machine Learning",
        "Statistics", "Mathematics", "Applied Mathematics",
        "Physics", "Operations Research", "Cognitive Science",
    ]

    # ---------------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------------

    @classmethod
    def parse(cls, jd_text: str) -> JDProfile:
        """
        Main entry point.  Parses the full JD text and returns a JDProfile.

        Each field is attempted with the LLM first; if the result is
        falsy or implausible the regex fallback is used transparently.

        Parameters
        ----------
        jd_text : Raw text extracted from the JD document.

        Returns
        -------
        JDProfile populated with all extractable signals.
        """
        profile = JDProfile(raw_text=jd_text)

        # ── Skills ────────────────────────────────────────────────────────────
        llm_req  = cls._llm_extract_skills(jd_text, required=True)
        llm_pref = cls._llm_extract_skills(jd_text, required=False)

        req_text, pref_text = cls._split_sections(jd_text)
        reg_req  = cls._regex_extract_skills(req_text)
        reg_pref = cls._regex_extract_skills(
            pref_text, exclude={s.lower() for s in reg_req}
        )

        log.info("Skills (LLM): Required: %s | Preferred: %s", llm_req, llm_pref)
        log.info("Skills (Regex): Required: %s | Preferred: %s", reg_req, reg_pref)

        # Union of LLM and Regex skills to ensure maximum tech vocabulary coverage
        required_union = list(dict.fromkeys(llm_req + reg_req))
        preferred_union = list(dict.fromkeys(llm_pref + reg_pref))
        # Deduplicate preferred skills from required skills
        required_lower = {s.lower() for s in required_union}
        preferred_union = [s for s in preferred_union if s.lower() not in required_lower]

        profile.required_skills  = required_union
        profile.preferred_skills = preferred_union

        # ── Experience ────────────────────────────────────────────────────────
        llm_min_yr, llm_max_yr, llm_ideal_yr = cls._llm_extract_experience(jd_text)
        reg_min_yr, reg_max_yr, reg_ideal_yr = cls._regex_extract_experience(jd_text)

        log.info("Experience (LLM): min=%.1f, max=%.1f, ideal=%.1f", llm_min_yr, llm_max_yr, llm_ideal_yr)
        log.info("Experience (Regex): min=%.1f, max=%.1f, ideal=%.1f", reg_min_yr, reg_max_yr, reg_ideal_yr)

        if llm_ideal_yr > 0:
            profile.min_years   = llm_min_yr
            profile.max_years   = llm_max_yr
            profile.ideal_years = llm_ideal_yr
        else:
            profile.min_years   = reg_min_yr
            profile.max_years   = reg_max_yr
            profile.ideal_years = reg_ideal_yr

        # ── Location + Remote ────────────────────────────────────────────────
        llm_locs = cls._llm_extract_locations(jd_text)
        reg_locs = cls._regex_extract_locations(jd_text)

        log.info("Locations (LLM): %s", llm_locs)
        log.info("Locations (Regex): %s", reg_locs)

        profile.preferred_locations = llm_locs if llm_locs else reg_locs
        profile.remote_ok = cls._extract_remote(jd_text)
        log.info("Remote OK: %s", profile.remote_ok)

        # ── Education ────────────────────────────────────────────────────────
        profile.required_degrees = cls._regex_extract_degrees(jd_text)
        log.info("Degrees (Regex): %s", profile.required_degrees)

        llm_fields = cls._llm_extract_fields(jd_text)
        reg_fields = cls._regex_extract_fields(jd_text)

        log.info("Fields of study (LLM): %s", llm_fields)
        log.info("Fields of study (Regex): %s", reg_fields)

        profile.required_fields = llm_fields if llm_fields else reg_fields

        profile.required_certs = cls._regex_extract_certs(jd_text)
        log.info("Certifications (Regex): %s", profile.required_certs)

        log.info(
            "JDParser Final Output: %d required skills, %d preferred, "
            "exp=%.0f–%.0f yr (ideal %.1f), locations=%s, remote=%s, "
            "degrees=%s, certs=%d",
            len(profile.required_skills),
            len(profile.preferred_skills),
            profile.min_years,
            profile.max_years,
            profile.ideal_years,
            profile.preferred_locations,
            profile.remote_ok,
            profile.required_degrees,
            len(profile.required_certs),
        )
        return profile

    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # LLM-based extractors
    # ---------------------------------------------------------------------------

    @classmethod
    def _get_section_text(cls, text: str, keywords: list[str], max_chars: int = 1500) -> str:
        """
        Finds a section in the text that starts with any of the keywords and returns it.
        """
        lines = text.splitlines()
        start_idx = -1
        for i, line in enumerate(lines):
            stripped = line.lower()
            if any(kw in stripped for kw in keywords) and len(stripped) < 100:
                start_idx = i
                break
        if start_idx != -1:
            return "\n".join(lines[start_idx:])[:max_chars]
        return text[:max_chars]

    @classmethod
    def _llm_extract_skills(cls, jd_text: str, *, required: bool) -> list[str]:
        """
        Uses flan-t5-base to extract required or preferred skills from the JD.
        """
        if required:
            keywords = ["absolutely need", "must-have", "required skills", "essential", "requirements", "skills inventory"]
            skill_type = "required"
        else:
            keywords = ["like you to have", "nice-to-have", "preferred", "bonus", "plus", "desirable"]
            skill_type = "preferred"

        section = cls._get_section_text(jd_text, keywords=keywords)
        prompt = (
            f"Identify all the {skill_type} technical skills and tools mentioned in this text. "
            f"Return only a comma-separated list of names. "
            f"Text:\n{section}"
        )
        raw = _LLMSingleton.generate(prompt, max_new_tokens=100)
        # Tokenise the LLM's raw output using regex to split into words
        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#\-\.]*", raw)
        jd_lower = jd_text.lower()
        skills: list[str] = []
        seen_lower: set[str] = set()

        for tok in raw_tokens:
            tok = tok.strip(".-")
            lower = tok.lower()
            if not tok or lower in seen_lower or lower in _STOP_WORDS:
                continue
            if len(tok) < cls._MIN_SKILL_LEN:
                continue
            if lower not in jd_lower:
                continue

            is_known_tech = lower in _TECH_VOCAB
            # camelCase (contains lowercase followed by uppercase, e.g. PyTorch, LangChain, LlamaIndex)
            is_camel_case = (
                not tok.islower() 
                and not tok.isupper() 
                and any(c.isupper() for c in tok[1:])
            )
            # contains symbols like +, #, . (e.g. C++, C#, Next.js)
            has_special_chars = any(c in tok for c in ["+", "#", "."])

            if is_known_tech or is_camel_case or has_special_chars:
                seen_lower.add(lower)
                skills.append(tok)

        log.debug("LLM %s skills: %s", skill_type, skills)
        return skills

    @classmethod
    def _llm_extract_experience(cls, jd_text: str) -> tuple[float, float, float]:
        """
        Uses flan-t5-base to extract the experience range from the JD.
        """
        # Focus on the experience section or the beginning metadata
        keywords = ["experience required", "experience", "years of experience", "what we mean"]
        section = cls._get_section_text(jd_text, keywords=keywords, max_chars=800)
        
        prompt = (
            "From this text, extract the required years of experience range. "
            "Reply with only two numbers separated by a comma (e.g. 5,9). "
            f"Text:\n{section}"
        )
        raw = _LLMSingleton.generate(prompt, max_new_tokens=16)
        if not raw:
            return 0.0, 0.0, 0.0

        # Parse "min,max" response
        nums = re.findall(r"\d+(?:\.\d+)?", raw)
        if len(nums) >= 2:
            lo, hi = float(nums[0]), float(nums[1])
            if 0 < lo <= hi <= 40:
                ideal = round((lo + hi) / 2, 1)
                log.debug("LLM experience: %.0f–%.0f (ideal %.1f)", lo, hi, ideal)
                return lo, hi, ideal
        elif len(nums) == 1:
            lo = float(nums[0])
            if 0 < lo <= 40:
                ideal = lo + 2.0
                log.debug("LLM experience (single): %.0f+ (ideal %.1f)", lo, ideal)
                return lo, lo + 5.0, ideal

        return 0.0, 0.0, 0.0

    @classmethod
    def _llm_extract_locations(cls, jd_text: str) -> list[str]:
        """
        Uses flan-t5-base to extract preferred work locations from the JD.
        """
        # Location metadata is always at the top of the file
        section = jd_text[:800]
        prompt = (
            "From this text, extract the preferred work locations (cities or countries). "
            "Return only a comma-separated list of location names. "
            f"Text:\n{section}"
        )
        raw = _LLMSingleton.generate(prompt, max_new_tokens=64)
        if not raw:
            return []

        # Split raw by comma, slash, "or", or "and"
        tokens = []
        for part in re.split(r",|/|\s+or\s+|\s+and\s+", raw):
            part = part.strip().strip(".")
            if part:
                tokens.append(part)

        jd_lower = jd_text.lower()
        locs: list[str] = []
        seen: set[str] = set()

        for token in tokens:
            lower = token.lower()
            # Hallucination guard: location must appear in the JD text
            if (
                token
                and len(token) >= 2
                and lower not in seen
                and lower in jd_lower
            ):
                seen.add(lower)
                locs.append(token.title())

        log.debug("LLM locations: %s", locs)
        return locs

    @classmethod
    def _llm_extract_fields(cls, jd_text: str) -> list[str]:
        """
        Uses flan-t5-base to extract required fields of study from the JD.
        """
        keywords = ["absolutely need", "must-have", "required skills", "essential", "requirements", "skills inventory", "education"]
        section = cls._get_section_text(jd_text, keywords=keywords)
        prompt = (
            "From this text, extract the required fields of study or academic disciplines. "
            "Return only a comma-separated list of names. "
            f"Text:\n{section}"
        )
        raw = _LLMSingleton.generate(prompt, max_new_tokens=64)
        if not raw:
            return []

        jd_lower = jd_text.lower()
        fields: list[str] = []
        seen: set[str] = set()

        for token in raw.split(","):
            token = token.strip().strip(".")
            lower = token.lower()
            if (
                token
                and len(token) >= 3
                and lower not in seen
                and lower not in _STOP_WORDS
                and lower in jd_lower
            ):
                seen.add(lower)
                fields.append(token)

        log.debug("LLM fields: %s", fields)
        return fields

    # ---------------------------------------------------------------------------
    # Regex / heuristic fallback extractors
    # ---------------------------------------------------------------------------

    @classmethod
    def _split_sections(cls, text: str) -> tuple[str, str]:
        """
        Splits JD text into (required_text, preferred_text) by finding the
        first section header that matches a "preferred / nice-to-have" pattern.

        Only lines shorter than 80 characters are treated as section headers
        to avoid false matches in paragraph body text.

        Returns
        -------
        Tuple of (required_text, preferred_text).  If no boundary is found
        the entire text is returned as required_text with an empty preferred.
        """
        lines = text.splitlines()
        preferred_start: Optional[int] = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and len(stripped) < 80 and _PREFERRED_SECTION_RE.search(stripped):
                preferred_start = i
                break

        if preferred_start is None:
            return text, ""

        return "\n".join(lines[:preferred_start]), "\n".join(lines[preferred_start:])

    @classmethod
    def _regex_extract_skills(
        cls,
        text: str,
        exclude: set[str] | None = None,
    ) -> list[str]:
        """
        Extracts skill tokens from a text block using regex and vocabulary lookup.

        Strategy:
          1. Tokenise using a regex that captures tech-style tokens (allows
             +, #, ., - as inner characters).
          2. Keep tokens that are in _TECH_VOCAB OR start with an uppercase
             letter and are long enough (likely a proper noun / product name).
          3. Remove stop-words and very short tokens.
          4. Deduplicate while preserving original casing of first occurrence.

        Parameters
        ----------
        text    : Text block to extract skills from.
        exclude : Set of lowercase tokens to skip (already found in required).

        Returns
        -------
        Deduplicated list of skill name strings.
        """
        if not text:
            return []

        if exclude is None:
            exclude = set()

        raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#\-\.]*", text)
        seen_lower: set[str] = set(exclude)
        skills: list[str] = []

        for tok in raw_tokens:
            tok = tok.strip(".-")
            lower = tok.lower()
            if not tok or lower in seen_lower or lower in _STOP_WORDS:
                continue
            if len(tok) < cls._MIN_SKILL_LEN:
                continue

            is_known_tech = lower in _TECH_VOCAB
            # camelCase (contains lowercase followed by uppercase, e.g. PyTorch, LangChain, LlamaIndex)
            is_camel_case = (
                not tok.islower() 
                and not tok.isupper() 
                and any(c.isupper() for c in tok[1:])
            )
            # contains symbols like +, #, . (e.g. C++, C#, Next.js)
            has_special_chars = any(c in tok for c in ["+", "#", "."])

            if is_known_tech or is_camel_case or has_special_chars:
                seen_lower.add(lower)
                skills.append(tok)

        return skills

    @classmethod
    def _regex_extract_experience(cls, text: str) -> tuple[float, float, float]:
        """
        Extracts experience range from JD text using regex pattern matching.

        Patterns checked in priority order:
          "5-8 years", "5 to 8 years", "5–8 yrs"
          "5+ years", "5 or more years"
          "minimum 5 years", "at least 5 years"

        Returns
        -------
        Tuple of (min_years, max_years, ideal_years).
        Falls back to (3.0, 10.0, 5.0) if no pattern matches.
        """
        lower = text.lower()

        range_patterns = [
            r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        ]
        plus_patterns = [
            r"(\d+(?:\.\d+)?)\+\s*(?:years?|yrs?)",
            r"(\d+(?:\.\d+)?)\s+or\s+more\s+(?:years?|yrs?)",
        ]
        min_patterns = [
            r"(?:minimum|at\s+least|minimum\s+of)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
        ]

        for pat in range_patterns:
            m = re.search(pat, lower)
            if m:
                lo, hi = float(m.group(1)), float(m.group(2))
                ideal = round((lo + hi) / 2, 1)
                log.debug("Regex experience range: %.0f–%.0f (ideal %.1f)", lo, hi, ideal)
                return lo, hi, ideal

        for pat in plus_patterns:
            m = re.search(pat, lower)
            if m:
                lo = float(m.group(1))
                log.debug("Regex experience X+ pattern: %.0f+ (ideal %.1f)", lo, lo + 2.0)
                return lo, lo + 5.0, lo + 2.0

        for pat in min_patterns:
            m = re.search(pat, lower)
            if m:
                lo = float(m.group(1))
                log.debug("Regex experience minimum: %.0f (ideal %.1f)", lo, lo + 2.0)
                return lo, lo + 5.0, lo + 2.0

        log.debug("Regex experience: no pattern found, using defaults (3-10, ideal 5)")
        return 3.0, 10.0, 5.0

    @classmethod
    def _regex_extract_locations(cls, text: str) -> list[str]:
        """
        Extracts preferred city / country names from the JD text by matching
        against the _CITIES vocabulary.

        Prioritises matches near explicit location context words (e.g.
        "Location:", "Office:", "Hub:") before scanning the full text.

        Returns
        -------
        Deduplicated list of location strings (title-cased).
        """
        found: list[str] = []
        seen: set[str] = set()

        context_matches = re.findall(
            r"(?:location|office|hub|city|cities|work\s+location)[^\n]{0,120}",
            text,
            re.IGNORECASE,
        )
        context_text = " ".join(context_matches)

        # Search longest city names first to avoid sub-string false matches
        for city in sorted(_CITIES, key=len, reverse=True):
            pattern = r"\b" + re.escape(city) + r"\b"
            if re.search(pattern, context_text, re.IGNORECASE):
                if city not in seen:
                    seen.add(city)
                    found.append(city.title())

        # Widen to full text if context search found nothing
        if not found:
            for city in sorted(_CITIES, key=len, reverse=True):
                pattern = r"\b" + re.escape(city) + r"\b"
                if re.search(pattern, text, re.IGNORECASE):
                    if city not in seen:
                        seen.add(city)
                        found.append(city.title())

        log.debug("Regex locations: %s", found)
        return found

    @classmethod
    def _extract_remote(cls, text: str) -> bool:
        """
        Returns True if the JD explicitly mentions remote or hybrid work.

        Checks for: "remote", "hybrid", "work from home", "WFH",
        "flexible work".
        """
        return bool(re.search(
            r"\b(?:remote|hybrid|work.from.home|wfh|flexible.work)\b",
            text,
            re.IGNORECASE,
        ))

    @classmethod
    def _regex_extract_degrees(cls, text: str) -> list[str]:
        """
        Extracts degree abbreviations from the JD text using regex patterns.

        Returns
        -------
        Deduplicated list of degree strings (e.g. ["B.Tech", "M.Tech"]).
        """
        found: list[str] = []
        seen: set[str] = set()

        for pat in _DEGREE_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                normalized = m.group(0).strip().rstrip(".")
                key = normalized.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(normalized)

        log.debug("Regex degrees: %s", found)
        return found

    @classmethod
    def _regex_extract_fields(cls, text: str) -> list[str]:
        """
        Extracts fields of study from the JD text using a curated vocabulary.

        Returns
        -------
        List of matching field of study strings from _FIELD_VOCAB.
        """
        found = [
            fld for fld in cls._FIELD_VOCAB
            if re.search(r"\b" + re.escape(fld) + r"\b", text, re.IGNORECASE)
        ]
        log.debug("Regex fields: %s", found)
        return found

    @classmethod
    def _regex_extract_certs(cls, text: str) -> list[str]:
        """
        Extracts certification names from the JD text by matching known
        certification provider / keyword patterns.

        Returns
        -------
        Deduplicated list of certification name strings.
        """
        cert_re = re.compile(
            r"(?:AWS|GCP|Google|Azure|Microsoft|Kubernetes|CKA|CKS"
            r"|Databricks|Snowflake|dbt|Scrum|PMP|CISSP|TensorFlow)"
            r"[A-Za-z0-9 \-]{0,50}"
            r"(?:Certified|Certification|Certificate|Professional"
            r"|Associate|Practitioner|Engineer|Expert)?",
            re.IGNORECASE,
        )
        certs: list[str] = []
        seen: set[str] = set()

        for m in cert_re.finditer(text):
            cert = m.group(0).strip()
            key = cert.lower()
            if len(cert) > 4 and key not in seen:
                seen.add(key)
                certs.append(cert)

        log.debug("Regex certs: %s", certs)
        return certs

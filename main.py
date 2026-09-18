import csv
import os
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
import requests  # To interface with llama.cpp server


DEFAULT_MAP_PROMPT = (
    "<start_of_turn>user\nYou are a data analysis assistant.\n"
    "Perform the following task on this dataset subset: {goal_prompt}\n"
    "Data:\n{chunk}\n"
    "Provide structured metrics, key trends, and numerical insights."
    "<end_of_turn>\n<start_of_turn>model\n"
)
DEFAULT_REDUCE_PROMPT = (
    "<start_of_turn>user\nYou are a lead data scientist.\n"
    "Below are regional summaries of a larger dataset. "
    "Combine these insights into a single unified executive summary for the "
    "overall goal: {goal_prompt}\n\nIntermediate Summaries:\n"
    "{combined_summaries}<end_of_turn>\n<start_of_turn>model\n"
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean value")
    return normalized in {"true", "1", "yes"}


def _env_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    csv_file_path: str
    llm_endpoint: str
    sanitize_pii: bool
    sensitive_columns: List[str]
    presidio_language: str
    max_context_tokens: int
    chunk_max_tokens: int
    max_llm_output_tokens: int
    token_estimate_chars_per_token: int
    request_timeout_seconds: int
    analysis_goal: str
    map_prompt_template: str
    reduce_prompt_template: str

    @classmethod
    def from_environment(cls) -> "Settings":
        settings = cls(
            csv_file_path=os.getenv("CSV_FILE_PATH", "large_sales_data.csv"),
            llm_endpoint=os.getenv(
                "LLM_ENDPOINT", "http://localhost:8080/completion"
            ),
            sanitize_pii=_env_bool("SANITIZE_PII", False),
            sensitive_columns=[
                column.strip().lower()
                for column in os.getenv(
                    "SENSITIVE_COLUMNS", "customer_name,credit_card,ssn"
                ).split(",")
                if column.strip()
            ],
            presidio_language=os.getenv("PRESIDIO_LANGUAGE", "en"),
            max_context_tokens=_env_int("MAX_CONTEXT_TOKENS", 128000),
            chunk_max_tokens=_env_int("CHUNK_MAX_TOKENS", 30000),
            max_llm_output_tokens=_env_int("MAX_LLM_OUTPUT_TOKENS", 1024),
            token_estimate_chars_per_token=_env_int(
                "TOKEN_ESTIMATE_CHARS_PER_TOKEN", 4
            ),
            request_timeout_seconds=_env_int("REQUEST_TIMEOUT_SECONDS", 120),
            analysis_goal=os.getenv(
                "ANALYSIS_GOAL",
                "Identify top revenue categories, average order values, and notable outliers.",
            ),
            map_prompt_template=os.getenv("MAP_PROMPT_TEMPLATE", DEFAULT_MAP_PROMPT).replace(
                r"\n", "\n"
            ),
            reduce_prompt_template=os.getenv(
                "REDUCE_PROMPT_TEMPLATE", DEFAULT_REDUCE_PROMPT
            ).replace(r"\n", "\n"),
        )
        if (
            settings.chunk_max_tokens + settings.max_llm_output_tokens
            >= settings.max_context_tokens
        ):
            raise ValueError(
                "CHUNK_MAX_TOKENS + MAX_LLM_OUTPUT_TOKENS must be less than "
                "MAX_CONTEXT_TOKENS"
            )
        return settings


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    return max(1, len(text) // chars_per_token)

class DataSanitizer:
    """Removes PII patterns and targeted sensitive columns."""
    
    def __init__(
        self,
        target_columns: List[str] = None,
        enabled: bool = True,
        language: str = "en",
    ):
        self.enabled = enabled
        self.target_columns = [col.lower() for col in (target_columns or [])]
        self.language = language
        if enabled:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
        else:
            self.analyzer = None
            self.anonymizer = None

    def sanitize_row(self, row: Dict[str, str]) -> Dict[str, str]:
        if not self.enabled:
            return row

        sanitized = {}
        for key, value in row.items():
            if key.lower() in self.target_columns:
                sanitized[key] = "[REDACTED_COLUMN]"
                continue

            text = str(value)
            results = self.analyzer.analyze(text=text, language=self.language)
            sanitized[key] = self.anonymizer.anonymize(
                text=text, analyzer_results=results
            ).text
        return sanitized


class CSVChunker:
    """Chunks CSV rows into Markdown blocks fitting context constraints."""
    
    def __init__(self, max_tokens: int = 30000, chars_per_token: int = 4):
        self.max_tokens = max_tokens
        self.chars_per_token = chars_per_token

    def chunk_csv(self, filepath: str, sanitizer: DataSanitizer) -> List[str]:
        chunks = []
        current_chunk_rows = []
        current_token_count = 0

        with open(filepath, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            header_str = " | ".join(headers) + "\n" + "|".join(["---"] * len(headers)) + "\n"
            
            current_token_count += estimate_tokens(header_str, self.chars_per_token)

            for row in reader:
                clean_row = sanitizer.sanitize_row(row)
                row_str = " | ".join(clean_row.values()) + "\n"
                row_tokens = estimate_tokens(row_str, self.chars_per_token)

                if current_token_count + row_tokens > self.max_tokens:
                    # Finalize current chunk
                    chunks.append(header_str + "".join(current_chunk_rows))
                    current_chunk_rows = [row_str]
                    current_token_count = estimate_tokens(
                        header_str, self.chars_per_token
                    ) + row_tokens
                else:
                    current_chunk_rows.append(row_str)
                    current_token_count += row_tokens

            if current_chunk_rows:
                chunks.append(header_str + "".join(current_chunk_rows))

        return chunks


class LlamaCppAnalyzer:
    """Interfaces with local llama.cpp server for Map-Reduce processing."""
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8080/completion",
        max_output_tokens: int = 1024,
        timeout_seconds: int = 120,
        map_prompt_template: str = DEFAULT_MAP_PROMPT,
        reduce_prompt_template: str = DEFAULT_REDUCE_PROMPT,
    ):
        self.endpoint = endpoint
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.map_prompt_template = map_prompt_template
        self.reduce_prompt_template = reduce_prompt_template

    def query_llm(self, prompt: str) -> str:
        payload = {
            "prompt": prompt,
            "temperature": 0.2,
            "n_predict": self.max_output_tokens,
        }
        response = requests.post(
            self.endpoint, json=payload, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        return response.json().get("content", "")

    def map_reduce_analysis(self, chunks: List[str], goal_prompt: str) -> str:
        intermediate_summaries = []

        print(f"--- MAP PHASE: Processing {len(chunks)} Chunks ---")
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            map_prompt = self.map_prompt_template.format(
                goal_prompt=goal_prompt,
                chunk=chunk,
                chunk_number=i + 1,
                chunk_count=len(chunks),
            )
            summary = self.query_llm(map_prompt)
            intermediate_summaries.append(f"### Chunk {i+1} Summary:\n{summary}")

        print("\n--- REDUCE PHASE: Generating Final Analysis ---")
        combined_summaries = "\n\n".join(intermediate_summaries)
        reduce_prompt = self.reduce_prompt_template.format(
            goal_prompt=goal_prompt,
            combined_summaries=combined_summaries,
            chunk_count=len(chunks),
        )
        return self.query_llm(reduce_prompt)


# --- Example Usage ---
if __name__ == "__main__":
    load_dotenv()
    settings = Settings.from_environment()

    sanitizer = DataSanitizer(
        target_columns=settings.sensitive_columns,
        enabled=settings.sanitize_pii,
        language=settings.presidio_language,
    )

    chunker = CSVChunker(
        max_tokens=min(
            settings.chunk_max_tokens,
            settings.max_context_tokens - settings.max_llm_output_tokens,
        ),
        chars_per_token=settings.token_estimate_chars_per_token,
    )
    chunks = chunker.chunk_csv(settings.csv_file_path, sanitizer)
    print(f"Created {len(chunks)} chunks.")

    analyzer = LlamaCppAnalyzer(
        endpoint=settings.llm_endpoint,
        max_output_tokens=settings.max_llm_output_tokens,
        timeout_seconds=settings.request_timeout_seconds,
        map_prompt_template=settings.map_prompt_template,
        reduce_prompt_template=settings.reduce_prompt_template,
    )

    final_report = analyzer.map_reduce_analysis(chunks, settings.analysis_goal)
    print("\n=== FINAL DATASET ANALYSIS ===")
    print(final_report)
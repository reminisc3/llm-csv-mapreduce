# Overview

A scalable, privacy-focused Python pipeline for running full-dataset analysis on large CSV files using local LLMs (such as Gemma 4).

By combining automated PII sanitization with a Map-Reduce chunking architecture, this tool enables high-accuracy aggregation over massive datasets without exposing sensitive client data or overflowing model context windows.

# Key Features

- Automated PII Redaction: Removes target columns and automatically detects and anonymizes sensitive data with Presidio prior to LLM processing.

- Context-Aware Chunking: Slices raw CSV data into Markdown-formatted blocks calibrated to fit cleanly within large context windows (up to 128k tokens).

- Map-Reduce Aggregation: Slices processing into two phases—generating structured chunk-level summaries (Map), then synthesizing them into a single executive analysis (Reduce).

- 100% Local Inference: Designed to run seamlessly against local llama.cpp server instances, ensuring complete data privacy and zero API dependency.

## Configuration

Copy `.env.sample` to `.env` and adjust the values for your environment. Install dependencies with:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

PII sanitization is disabled by default. Set `SANITIZE_PII=true` to enable Presidio detection and anonymization. `SENSITIVE_COLUMNS` is a comma-separated list of columns that should always be redacted when sanitization is enabled.

The CSV path, CSV encoding (`CSV_ENCODING`, default `cp1252`), LLM endpoint, API format, context window, chunk limit, output limit, token estimation ratio, and request timeout (`REQUEST_TIMEOUT_SECONDS`, default 600 seconds per request) are all configurable through `.env`. Use `CSV_ENCODING=utf-8` for UTF-8 files. `LLM_API_FORMAT=auto` detects OpenAI chat endpoints ending in `/chat/completions`; use `openai` or `llama_cpp` to override detection. `CHUNK_MAX_TOKENS` plus `MAX_LLM_OUTPUT_TOKENS` must remain below `MAX_CONTEXT_TOKENS`.

The analysis goal is configured with `ANALYSIS_GOAL`. `MAP_PROMPT_TEMPLATE` and `REDUCE_PROMPT_TEMPLATE` customize the prompts while retaining the placeholders shown in `.env.sample`. Use `\n` for line breaks in `.env` values.

LLM conversations are written as one JSON object per line to `CONVERSATION_LOG_PATH` (default `logs/conversations.jsonl`). Each request records its phase, chunk number, prompt, response, and timestamp; failed requests record the error and are re-raised. The file is updated after every request, so completed chunks remain available if a later request fails. Logs can contain dataset content and should be handled as sensitive data.

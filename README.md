# langchain-rag-kb

A chatbot over your own knowledge base, **100% open source and self-hosted**: LangGraph + LangChain for the logic, **llama.cpp** to run the model on CPU (no GPU, no paid APIs), **BM25** search over your documents and, optionally, **Langfuse** for observability.

One `docker compose up` starts everything. The model (Qwen3-1.7B, Apache-2.0) is downloaded from HuggingFace on first start.

```
 browser ──▶ chatbot (FastAPI + LangGraph) ──▶ llama (llama.cpp, Qwen3-1.7B)
                │        │
                │        └─▶ knowledge base (BM25 over ./knowledge, a sitemap or your own source)
                ├─▶ SQLite (conversations, LangGraph checkpointer)
                └─▶ Langfuse (optional: a trace for every request)
```

## Features

- **Simple, lightweight RAG**: for each question, the most relevant excerpts from the knowledge base are sent to the model. Search is BM25 (keywords): no embedding model and no extra RAM.
- **Per-conversation memory** with LangGraph's SQLite checkpointer. Recent history is trimmed to fit the context window.
- **Streaming answers** in a simple chat page, with configurable suggested questions. Links and bold text are rendered.
- **Answers in the question's language**: Portuguese or English questions are detected, and the search prefers excerpts in that language.
- **Tuned for CPU**:
  - **Stable prompt order**, so llama.cpp can reuse its cache across requests.
  - **Answer cache** for repeated first questions, which are answered instantly.
  - **Short answers**, and **"thinking out loud" turned off** for Qwen3 models.
- **Doesn't make things up**: it only answers from the knowledge base. When the answer isn't there, it says so and points to a contact.
- **Optional observability**: each conversation is a session in Langfuse, with the search, prompt, answer, tokens, cached tokens and timing.

## Stack and licenses

| Component | Role | License |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://github.com/langchain-ai/langchain) | Graph, memory, model client | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | CPU inference, OpenAI-compatible API | MIT |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (GGUF by [unsloth](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)) | Model | Apache-2.0 |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Knowledge base search | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | API and chat page | MIT |
| [Langfuse](https://github.com/langfuse/langfuse) (optional) | Observability | MIT (core) |

It doesn't use LangGraph Server or LangSmith, which aren't open source. LangGraph runs as a library inside the app.

## Requirements

- Docker with Docker Compose v2.24 or newer.
- **Chatbot only**: 2 vCPUs and ~2 GB of free RAM.
- **With Langfuse**: ~2 GB more RAM.
- ~1.5 GB of disk for the model and images, plus ~3 GB with Langfuse.

## Getting started

```bash
git clone https://github.com/staminna/langchain-rag-kb.git
cd langchain-rag-kb
./scripts/init-env.sh          # creates .env with random secrets
docker compose up -d
```

Open http://localhost:8000. On first start, llama.cpp downloads the model (~1.1 GB). Follow it with `docker compose logs -f llama`.

The default texts (chat page, system prompt and the example knowledge base) are in European Portuguese. Change them in `.env` and `./knowledge`, as described below.

### With Langfuse

```bash
./scripts/init-env.sh --langfuse
docker compose up -d
```

The dashboard is at http://localhost:3000. The admin email and password are in `.env` (`LANGFUSE_ADMIN_EMAIL`, `LANGFUSE_ADMIN_PASSWORD`). Sign-up of new accounts is disabled, and so is telemetry to Langfuse.

Already have a `.env`? To turn Langfuse on, set `COMPOSE_PROFILES=langfuse` and `LANGFUSE_HOST=http://langfuse-web:3000` in it.

## Your knowledge base

Choose the source with `KB_SOURCE` in `.env`. The knowledge base is rebuilt every time the chatbot starts (`docker compose restart chatbot`).

| `KB_SOURCE` | Where the content comes from |
|---|---|
| `folder` (default) | `.md`/`.txt` files in `./knowledge`. Each `#` section is one document. An example is included. |
| `sitemap` | The pages listed in `KB_SITEMAP_URL` (the visible text of each page). |
| `module:function` | Your own Python function, e.g. to read from a CMS or a database. |

A custom source is a `.py` file next to `app.py` that returns a list of documents:

```python
# my_source.py  →  KB_SOURCE=my_source:build_docs
from kb import doc

def build_docs():
    return [
        doc("faq:1", "en", "Opening hours", "https://example.com/contact", ["We're open Monday to Friday, 9am-6pm."]),
    ]
```

Each document has a language (`"pt"`, `"en"` or `"any"`). Versions of the same document in different languages share the first argument (the key). Only one version goes into the prompt, and the one in the question's language is preferred.

To see what the search returns for a question:

```bash
docker compose exec chatbot python kb.py "what are your opening hours?"
```

If the content has addresses or text that must not reach the public, set `KB_STRIP_PATTERN` to a regular expression. Anything that matches is removed from the documents.

## Configuration

Everything goes in `.env` (see [.env.example](.env.example)):

| Variable | Default | Purpose |
|---|---|---|
| `BOT_NAME`, `BOT_CONTACT` | `este projeto`, empty | Used in the system prompt and in the "I don't know" answer |
| `SYSTEM_PROMPT` | generated | Replaces the whole system prompt |
| `UI_TITLE`, `UI_INTRO`, `UI_SUGGESTIONS` | | Page title, intro text and suggested questions (separated by `\|`) |
| `LLM_HF_MODEL` | `unsloth/Qwen3-1.7B-GGUF:Q4_K_M` | Any GGUF on HuggingFace (`repo:quantization`) |
| `LLM_CONTEXT`, `LLM_PARALLEL` | `6144`, `2` | Total context and parallel requests: each request gets `LLM_CONTEXT / LLM_PARALLEL` tokens |
| `MAX_ANSWER_TOKENS` | `250` | Maximum answer length |
| `MAX_HISTORY_TOKENS` | `800` | How much history goes into the prompt |
| `KB_MAX_CHARS` | `1600` | How much knowledge base text goes into the prompt (less = faster) |
| `LLM_TEMPERATURE` | `0.1` | Low, so answers stick to the excerpts |
| `CHAT_BIND`, `CHAT_PORT` | `127.0.0.1`, `8000` | Where the chatbot listens |

## Speed

Measured on a 4 vCPU server (AMD EPYC, no GPU) with Qwen3-1.7B:

| | |
|---|---|
| Prompt processing | ~127 tokens/s |
| Generation | ~22 tokens/s |
| Typical answer | 2 to 9 s |
| Repeated question (cache) | instant |

What matters most is how much knowledge base text goes into each question (`KB_MAX_CHARS`). A larger model, such as `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`, answers better but is ~2.5× slower on CPU.

## Running in production

- The chatbot **has no login**. To publish it, put it behind a reverse proxy (nginx, Caddy, Traefik) with HTTPS and authentication. With nginx, use `proxy_buffering off;` so streaming works.
- The Langfuse dashboard only listens on `127.0.0.1`. Reach it through an SSH tunnel: `ssh -N -L 3000:127.0.0.1:3000 your-server`.
- Conversations are stored in `./data/chat.sqlite`. With Langfuse on, they are stored there too.
- The chatbot container receives the variables from `.env` (`env_file`), including Langfuse passwords it doesn't use. If that bothers you, move the Langfuse secrets to a separate file.

## Developing without Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# a llama-server running on localhost:8080, e.g.:
#   llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --jinja --reasoning-budget 0 -c 6144 --parallel 2
.venv/bin/uvicorn app:app --port 8000
```

## Limitations

- **Small model**: the 1.7B sometimes mixes up information from different excerpts, e.g. attributing a piece of equipment to the wrong room. For more accuracy, use a larger model.
- **Keyword search**: it doesn't understand synonyms. If people ask using words that aren't in your documents (e.g. "where are you?" vs "address"), add those words to the section title.
- **Language detection** only distinguishes Portuguese and English. Questions in other languages are treated as Portuguese.
- **Limited context** (3072 tokens per request by default): only recent history goes into the prompt.

## License

[MIT](LICENSE)

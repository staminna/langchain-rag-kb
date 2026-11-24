# langchain

Chatbot com knowledge base própria, **100% open source e self-hosted**: LangGraph + LangChain para a lógica, **llama.cpp** para correr o modelo em CPU (sem GPU, sem APIs pagas), pesquisa **BM25** sobre os teus documentos e, opcionalmente, **Langfuse** para observabilidade.

Um `docker compose up` arranca tudo. O modelo (Qwen3-1.7B, Apache-2.0) é descarregado do HuggingFace no primeiro arranque.

```
 browser ──▶ chatbot (FastAPI + LangGraph) ──▶ llama (llama.cpp, Qwen3-1.7B)
                │        │
                │        └─▶ knowledge base (BM25 sobre ./knowledge, um sitemap ou a tua fonte)
                ├─▶ SQLite (conversas, checkpointer do LangGraph)
                └─▶ Langfuse (opcional: traces de cada pedido)
```

## Funcionalidades

- **RAG simples e leve**: a cada pergunta, os excertos mais relevantes da knowledge base vão para o modelo. A pesquisa é BM25 (palavras-chave), sem modelo de embeddings e sem RAM extra.
- **Memória por conversa** com o checkpointer SQLite do LangGraph. O histórico recente é cortado para caber no contexto.
- **Respostas em streaming** numa página de chat simples, com sugestões de perguntas configuráveis.
- **Afinado para CPU**:
  - **Prompt em ordem estável**, para o llama.cpp reaproveitar a cache entre pedidos.
  - **Cache de respostas** para primeiras perguntas repetidas, respondidas na hora.
  - **Respostas curtas** e **"pensar em voz alta" desligado** nos modelos Qwen3.
- **Não inventa**: só responde com o que está na knowledge base. Quando a resposta não está lá, diz que não sabe e indica um contacto.
- **Observabilidade opcional**: cada conversa é uma sessão no Langfuse, com a pesquisa, o prompt, a resposta, os tokens, os tokens vindos da cache e o tempo.

## Stack e licenças

| Componente | Função | Licença |
|---|---|---|
| [LangGraph](https://github.com/langchain-ai/langgraph) / [LangChain](https://github.com/langchain-ai/langchain) | Grafo, memória, cliente do modelo | MIT |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | Inferência em CPU, API compatível com OpenAI | MIT |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (GGUF da [unsloth](https://huggingface.co/unsloth/Qwen3-1.7B-GGUF)) | Modelo | Apache-2.0 |
| [rank-bm25](https://github.com/dorianbrown/rank_bm25) | Pesquisa na knowledge base | Apache-2.0 |
| [FastAPI](https://github.com/fastapi/fastapi) | API e página de chat | MIT |
| [Langfuse](https://github.com/langfuse/langfuse) (opcional) | Observabilidade | MIT (núcleo) |

Não usa o LangGraph Server nem o LangSmith, que não são open source. O LangGraph corre como biblioteca dentro da app.

## Requisitos

- Docker com Docker Compose v2.24 ou mais recente.
- **Só o chatbot**: 2 vCPU e ~2 GB de RAM livres.
- **Com o Langfuse**: mais ~2 GB de RAM.
- ~1,5 GB de disco para o modelo e as imagens, mais ~3 GB com o Langfuse.

## Começar

```bash
git clone https://github.com/staminna/langchain.git
cd langchain
./scripts/init-env.sh          # cria o .env com segredos aleatórios
docker compose up -d
```

Abre http://localhost:8000. No primeiro arranque o llama.cpp descarrega o modelo, ~1,1 GB. Acompanha com `docker compose logs -f llama`.

### Com Langfuse

```bash
./scripts/init-env.sh --langfuse
docker compose up -d
```

O painel fica em http://localhost:3000. O email e a password do admin estão no `.env` (`LANGFUSE_ADMIN_EMAIL`, `LANGFUSE_ADMIN_PASSWORD`). O registo de contas novas está desligado e a telemetria para a Langfuse também.

Já tens o `.env`? Para ligar o Langfuse, define nele `COMPOSE_PROFILES=langfuse` e `LANGFUSE_HOST=http://langfuse-web:3000`.

## A tua knowledge base

Escolhe a fonte com `KB_SOURCE` no `.env`. A knowledge base é reconstruída sempre que o chatbot arranca (`docker compose restart chatbot`).

| `KB_SOURCE` | De onde vem o conteúdo |
|---|---|
| `folder` (por defeito) | Ficheiros `.md`/`.txt` em `./knowledge`. Cada secção `#` é um documento. Vem com um exemplo. |
| `sitemap` | As páginas listadas em `KB_SITEMAP_URL` (o texto visível de cada uma). |
| `modulo:funcao` | A tua própria função Python, p.ex. para ler um CMS ou uma base de dados. |

Uma fonte própria é um ficheiro `.py` ao lado do `app.py`, que devolve uma lista de documentos:

```python
# minha_fonte.py  →  KB_SOURCE=minha_fonte:build_docs
from kb import doc

def build_docs():
    return [
        doc("faq:1", "pt", "Horário", "https://exemplo.pt/contactos", ["Abrimos de segunda a sexta, 9h-18h."]),
    ]
```

Para ver o que a pesquisa devolve para uma pergunta:

```bash
docker compose exec chatbot python kb.py "qual é o horário?"
```

Se o conteúdo tiver endereços ou textos que não devem chegar ao público, define `KB_STRIP_PATTERN` com uma expressão regular. Tudo o que corresponder é retirado dos documentos.

## Configuração

Tudo no `.env` (ver [.env.example](.env.example)):

| Variável | Por defeito | Para quê |
|---|---|---|
| `BOT_NAME`, `BOT_CONTACT` | `este projeto`, vazio | Usados no prompt de sistema e na resposta "não sei" |
| `SYSTEM_PROMPT` | gerado | Substitui o prompt por completo |
| `UI_TITLE`, `UI_INTRO`, `UI_SUGGESTIONS` | | Título, introdução e sugestões da página (separadas por `\|`) |
| `LLM_HF_MODEL` | `unsloth/Qwen3-1.7B-GGUF:Q4_K_M` | Qualquer GGUF no HuggingFace (`repo:quantização`) |
| `LLM_CONTEXT`, `LLM_PARALLEL` | `6144`, `2` | Contexto total e pedidos em paralelo: cada pedido tem `LLM_CONTEXT / LLM_PARALLEL` tokens |
| `MAX_ANSWER_TOKENS` | `250` | Tamanho máximo das respostas |
| `MAX_HISTORY_TOKENS` | `800` | Quanto histórico entra no prompt |
| `KB_MAX_CHARS` | `1600` | Quanto texto da knowledge base entra no prompt (menos = mais rápido) |
| `LLM_TEMPERATURE` | `0.1` | Baixa, para seguir os excertos |
| `CHAT_BIND`, `CHAT_PORT` | `127.0.0.1`, `8000` | Onde o chatbot escuta |

## Velocidade

Medido num servidor de 4 vCPU (AMD EPYC, sem GPU) com o Qwen3-1.7B:

| | |
|---|---|
| Leitura do prompt | ~127 tokens/s |
| Geração | ~22 tokens/s |
| Resposta típica | 2 a 9 s |
| Pergunta repetida (cache) | imediata |

O que mais pesa é a quantidade de texto da knowledge base em cada pergunta (`KB_MAX_CHARS`). Um modelo maior, como o `unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_K_M`, responde melhor mas é ~2,5× mais lento em CPU.

## Pôr em produção

- O chatbot **não tem login**. Para o publicar, põe-no atrás de um reverse proxy (nginx, Caddy, Traefik) com HTTPS e autenticação. No nginx, usa `proxy_buffering off;` para o streaming funcionar.
- O painel do Langfuse só escuta em `127.0.0.1`. Acede-lhe por túnel SSH: `ssh -N -L 3000:127.0.0.1:3000 servidor`.
- As conversas ficam em `./data/chat.sqlite`. Com o Langfuse ligado, ficam também lá.
- O container do chatbot recebe as variáveis do `.env` (`env_file`), incluindo as passwords do Langfuse que não usa. Se isso te incomodar, separa os segredos do Langfuse num ficheiro próprio.

## Desenvolver sem Docker

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# um llama-server a correr em localhost:8080, p.ex.:
#   llama-server -hf unsloth/Qwen3-1.7B-GGUF:Q4_K_M --jinja --reasoning-budget 0 -c 6144 --parallel 2
.venv/bin/uvicorn app:app --port 8000
```

## Limitações

- **Modelo pequeno**: o 1.7B às vezes mistura informação de excertos diferentes. Por exemplo, pode atribuir um equipamento à sala errada. Para mais rigor, usa um modelo maior.
- **Pesquisa por palavras-chave**: não percebe sinónimos. Se as pessoas perguntam com palavras que não estão nos documentos (p.ex. "onde ficam?" vs "morada"), acrescenta essas palavras ao título da secção.
- **Contexto limitado** (3072 tokens por pedido por defeito): só o histórico recente entra no prompt.

## Licença

[MIT](LICENSE)

"""Chatbot com knowledge base: FastAPI + LangGraph + llama.cpp (llama-server).

Tudo open source. O llama-server expõe uma API compatível com a da OpenAI,
por isso usamos o cliente ChatOpenAI só como protocolo: não há nenhuma
chamada à OpenAI. Configuração por variáveis de ambiente (ver .env.example).
"""

import asyncio
import os
import re
import unicodedata
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from kb import KnowledgeBase

# --- Configuração (variáveis de ambiente) ---------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
# Quem é o bot e para onde encaminhar quando não sabe a resposta.
BOT_NAME = os.getenv("BOT_NAME", "este projeto")
BOT_CONTACT = os.getenv("BOT_CONTACT", "")  # p.ex. "geral@exemplo.pt"
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    f"És o assistente de {BOT_NAME}. "
    "Responde na língua do utilizador (por defeito português de Portugal). "
    "Sê breve: no máximo 3 frases ou uma lista curta, sem introduções nem repetir a pergunta. "
    "Usa apenas a INFORMAÇÃO DO SITE que vem com cada pergunta. Se a resposta não estiver lá, "
    f"responde: \"Não tenho essa informação.{f' Pode contactar-nos em {BOT_CONTACT}.' if BOT_CONTACT else ''}\" "
    "Não inventes preços, datas nem factos.",
)
# Textos da página de chat (servidos em /config).
UI_TITLE = os.getenv("UI_TITLE", "Chatbot")
UI_INTRO = os.getenv("UI_INTRO", f"Pergunta sobre {BOT_NAME}.")
UI_SUGGESTIONS = [s.strip() for s in os.getenv(
    "UI_SUGGESTIONS", "Que modelo usa?|Que hardware é preciso?|Como funciona a resposta?"
).split("|") if s.strip()]
# O llama-server tem 3072 tokens por pedido (-c 6144 / --parallel 2).
# Prompt de sistema + histórico + excertos do site + resposta têm de caber aí.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "250"))
MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "800"))
KB_MAX_CHARS = int(os.getenv("KB_MAX_CHARS", "1600"))
ANSWER_CACHE_SIZE = int(os.getenv("ANSWER_CACHE_SIZE", "500"))
DB_PATH = os.getenv("DB_PATH", "data/chat.sqlite")

STATIC_DIR = Path(__file__).parent / "static"
kb = KnowledgeBase([])  # preenchida no arranque (lifespan)

# Observabilidade com Langfuse (open source, self-hosted): ativa só se houver
# LANGFUSE_HOST e as chaves (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY). Sem
# elas a app funciona igual, sem traces.
langfuse_handler = None
if all(os.getenv(v) for v in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
    from langfuse.langchain import CallbackHandler

    langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key="sem-chave",  # o llama-server ignora; o cliente exige um valor
    # O llama-server ignora o nome; serve só para identificar o modelo no Langfuse.
    model=os.getenv("LLM_MODEL_NAME", "qwen3-1.7b"),
    # Baixa: as respostas devem seguir os excertos, não ser criativas.
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
    max_tokens=MAX_ANSWER_TOKENS,
    streaming=True,
    stream_usage=True,  # pede a contagem de tokens no fim do streaming (vai para o Langfuse)
    # Desliga o "pensar em voz alta" dos modelos híbridos (Qwen3): alguns, como o
    # Qwen3-0.6B, ignoram o --reasoning-budget 0 do servidor. Os outros ignoram isto.
    extra_body=(
        {"chat_template_kwargs": {"enable_thinking": False}}
        if os.getenv("LLM_DISABLE_THINKING", "true").lower() == "true" else None
    ),
    timeout=180,
)


# --- Cache de respostas ----------------------------------------------------
# A primeira pergunta de uma conversa não depende de histórico: se alguém já
# fez a mesma pergunta (normalizada), devolvemos a resposta guardada sem chamar
# o modelo. Fica em memória: reiniciar o container (que também atualiza a
# knowledge base) limpa-a, por isso nunca serve respostas de conteúdo antigo.
answer_cache: OrderedDict[str, str] = OrderedDict()
NO_ANSWER = re.compile(r"não (sei|tenho essa informação)|don.t (know|have that information)", re.I)


def cache_key(question: str) -> str:
    text = unicodedata.normalize("NFKD", question.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.findall(r"[a-z0-9]+", text))


# --- Grafo LangGraph: um único nó que chama o modelo ----------------------
search_site = RunnableLambda(
    lambda query: kb.search(query, max_chars=KB_MAX_CHARS), name="pesquisa_site"
)


async def chatbot(state: MessagesState, config: RunnableConfig):
    *earlier, last = state["messages"]
    question = last.content
    key = cache_key(question) if not earlier else None
    if key and key in answer_cache:
        answer_cache.move_to_end(key)
        return {"messages": [AIMessage(answer_cache[key])]}

    # Histórico recente que cabe no contexto (estimativa de ~4 caracteres/token).
    history = trim_messages(
        earlier,
        max_tokens=MAX_HISTORY_TOKENS,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
    )
    # Pesquisa no site com esta e a pergunta anterior, para que um seguimento
    # curto ("e o estúdio B?") herde o contexto.
    previous = [m.content for m in earlier if m.type == "human"][-1:]
    # Corre como Runnable para aparecer como passo próprio no Langfuse,
    # com a consulta e os excertos encontrados.
    found = await search_site.ainvoke(" ".join(previous + [question]), config)
    context = "\n\n".join(f"[{d['url']}]\n{d['text']}" for d in found) or "(nada encontrado)"

    # Ordem pensada para a cache do llama-server: a parte que não muda
    # (prompt de sistema, depois o histórico) vem primeiro e é reaproveitada
    # entre pedidos; só os excertos do site e a pergunta nova são processados.
    prompt = [
        SystemMessage(SYSTEM_PROMPT),
        *history,
        HumanMessage(f"INFORMAÇÃO DO SITE:\n{context}\n\nPERGUNTA: {question}"),
    ]
    reply = await llm.ainvoke(prompt, config)

    # Não guarda respostas "não sei": a pergunta pode ter ficado sem excertos
    # por acaso, e uma correção na knowledge base não deve ficar tapada pela cache.
    if key and reply.content and not NO_ANSWER.search(reply.content):
        answer_cache[key] = reply.content
        if len(answer_cache) > ANSWER_CACHE_SIZE:
            answer_cache.popitem(last=False)
    return {"messages": [reply]}


def build_graph(checkpointer):
    builder = StateGraph(MessagesState)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")
    return builder.compile(checkpointer=checkpointer)


# --- API ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    # Lê a fonte da knowledge base no arranque; reiniciar o container atualiza-a.
    kb = await asyncio.to_thread(KnowledgeBase.load)
    print(f"[kb] {len(kb.docs)} documentos", flush=True)
    # O checkpointer guarda cada conversa (thread_id) em SQLite.
    print(f"[langfuse] {'ligado a ' + os.getenv('LANGFUSE_HOST', '?') if langfuse_handler else 'desligado'}", flush=True)
    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as saver:
        app.state.graph = build_graph(saver)
        yield
    if langfuse_handler:
        from langfuse import get_client

        get_client().shutdown()  # envia os traces que ainda estejam em memória


app = FastAPI(title="mdblangchain", lifespan=lifespan)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/config")
async def ui_config():
    """Título, introdução e sugestões da página de chat."""
    return {"title": UI_TITLE, "intro": UI_INTRO, "suggestions": UI_SUGGESTIONS}


@app.post("/chat")
async def chat(body: ChatIn):
    """Responde em streaming (texto simples, token a token)."""
    thread_id = body.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    if langfuse_handler:
        # Cada conversa do chat aparece como uma sessão no Langfuse.
        config["callbacks"] = [langfuse_handler]
        config["metadata"] = {"langfuse_session_id": thread_id}
    graph = app.state.graph

    async def tokens():
        streamed = False
        try:
            async for mode, payload in graph.astream(
                {"messages": [HumanMessage(body.message)]},
                config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    chunk, _meta = payload
                    if isinstance(chunk, AIMessageChunk) and chunk.content:
                        streamed = True
                        yield chunk.content
                elif not streamed:
                    # Resposta vinda da cache: não houve tokens, envia-a inteira.
                    for update in payload.values():
                        for message in (update or {}).get("messages", []):
                            if message.type == "ai" and message.content:
                                yield message.content
        except Exception as exc:  # mostra o erro no chat em vez de cortar a ligação
            yield f"\n\n[erro: {type(exc).__name__}: {exc}]"

    return StreamingResponse(
        tokens(),
        media_type="text/plain; charset=utf-8",
        headers={"X-Thread-Id": thread_id, "X-Accel-Buffering": "no"},
    )


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    """Mensagens guardadas de uma conversa (para a página recarregar)."""
    snapshot = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = snapshot.values.get("messages", []) if snapshot else []
    return [
        {"role": "user" if m.type == "human" else "assistant", "content": m.content}
        for m in messages
        if m.type in ("human", "ai") and m.content
    ]

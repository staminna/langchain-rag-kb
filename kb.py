"""Knowledge base: documentos de texto + pesquisa BM25 (palavras-chave).

Sem modelo de embeddings nem RAM extra. A cada pergunta, a app pede aqui os
excertos mais relevantes e envia-os ao modelo (RAG).

Fontes (variável KB_SOURCE):
  folder            (por defeito) ficheiros .md/.txt em KB_DIR (./knowledge)
  sitemap           páginas de um site listadas em KB_SITEMAP_URL
  modulo:funcao     a tua própria função Python que devolve documentos,
                    p.ex. "minha_fonte:build_docs" (ver doc() abaixo)

    python kb.py            # reconstrói data/kb.json
    python kb.py "pergunta" # mostra o que a pesquisa devolve
"""

import importlib
import json
import os
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

from rank_bm25 import BM25Okapi

KB_SOURCE = os.getenv("KB_SOURCE", "folder")
KB_DIR = Path(os.getenv("KB_DIR", "knowledge"))
KB_SITEMAP_URL = os.getenv("KB_SITEMAP_URL", "")
KB_PATH = Path(os.getenv("KB_PATH", "data/kb.json"))
# Expressão regular opcional: o que corresponder é retirado de todos os
# documentos (p.ex. endereços internos que não devem chegar ao público).
KB_STRIP_PATTERN = os.getenv("KB_STRIP_PATTERN", "")
CHUNK_CHARS = int(os.getenv("KB_CHUNK_CHARS", "1200"))


def doc(key, lang, title, url, lines):
    """Um documento da knowledge base.

    key   identifica a entidade; versões em várias línguas partilham a key e só
          a que pontuar mais entra no prompt.
    lang  "pt", "en", ... ou "any".
    """
    body = "\n".join(line for line in lines if line and not line.endswith(": "))
    return {"key": key, "lang": lang, "title": title, "url": url, "text": f"{title}\n{body}".strip()}


def chunks(text: str, size: int = CHUNK_CHARS):
    """Parte um texto em pedaços de até `size` caracteres, por parágrafos."""
    out, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if cur and len(cur) + len(para) > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}".strip() if cur else para[: size * 2]
    if cur:
        out.append(cur)
    return out


# --- Fontes ----------------------------------------------------------------
def folder_docs():
    """Ficheiros .md/.txt: cada secção (# título) é um documento, partido em pedaços."""
    docs = []
    for path in sorted(KB_DIR.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt"):
            continue
        text = path.read_text(encoding="utf-8")
        # Divide por cabeçalhos Markdown; o texto antes do primeiro fica com o nome do ficheiro.
        sections = re.split(r"(?m)^(#{1,3} .+)$", text)
        title = path.stem.replace("-", " ").replace("_", " ")
        pending = [(title, sections[0])]
        for i in range(1, len(sections), 2):
            pending.append((sections[i].lstrip("# ").strip(), sections[i + 1]))
        for n, (sec_title, body) in enumerate(pending):
            for p, piece in enumerate(chunks(body)):
                docs.append(doc(f"{path.name}:{n}:{p}", "any", sec_title, str(path.relative_to(KB_DIR)), [piece]))
    return docs


def sitemap_docs():
    """Páginas de um site (sitemap.xml): o texto visível de cada página, em pedaços."""
    from bs4 import BeautifulSoup

    with urllib.request.urlopen(KB_SITEMAP_URL, timeout=30) as res:
        urls = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", res.read().decode())
    docs = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as res:
                soup = BeautifulSoup(res.read(), "html.parser")
        except Exception as exc:
            print(f"[kb] ignorei {url}: {exc}", file=sys.stderr)
            continue
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
            tag.decompose()
        title = (soup.title.string or url).strip() if soup.title else url
        main = soup.find("main") or soup.body or soup
        text = re.sub(r"\n\s*\n+", "\n\n", main.get_text("\n", strip=True))
        for p, piece in enumerate(chunks(text)):
            docs.append(doc(f"{url}:{p}", "any", title, url, [piece]))
    return docs


def build_docs():
    if KB_SOURCE == "folder":
        docs = folder_docs()
    elif KB_SOURCE == "sitemap":
        docs = sitemap_docs()
    else:
        module, _, func = KB_SOURCE.partition(":")
        docs = getattr(importlib.import_module(module), func or "build_docs")()
    if KB_STRIP_PATTERN:
        strip = re.compile(KB_STRIP_PATTERN, re.I)
        for d in docs:
            d["text"] = strip.sub("", d["text"])
    return docs


# --- Pesquisa --------------------------------------------------------------
STOPWORDS = set("""
a o os as um uma uns umas de do da dos das em no na nos nas por para com sem que e ou se
ao aos à às é são ser ter tem têm há qual quais quanto quantos como onde quando vocês voces
eu tu ele ela nos me te lhe vos meu minha the of and or to in on at for with is are do does
have has what which how where when you your i my it this that an be can
""".split())


PT_WORDS = set("que o a os as de do da em no na para com como onde quando qual quais quanto é são têm tem fazem vocês voces não sim um uma".split())
EN_WORDS = set("the what which who where when how do does you your is are can have has of in for with and a an any".split())


def detect_lang(text: str, default: str = "pt") -> str:
    """'en' ou 'pt', contando palavras frequentes de cada língua.

    Num empate (p.ex. "phone number", sem nenhuma dessas palavras) devolve
    `default`: a língua da página de onde veio a pergunta, quando se sabe.
    """
    words = re.findall(r"[a-zà-ú]+", text.lower())
    en = sum(w in EN_WORDS for w in words)
    pt = sum(w in PT_WORDS for w in words)
    return "en" if en > pt else "pt" if pt > en else default


def tokens(text: str):
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # "Stemming" rudimentar: as 6 primeiras letras (estúdio/estúdios, microfone/microfones).
    return [w[:6] for w in re.findall(r"[a-z0-9]+", text) if w not in STOPWORDS]


class KnowledgeBase:
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = BM25Okapi([tokens(d["text"]) for d in docs]) if docs else None

    @classmethod
    def load(cls, refresh: bool = True):
        """Reconstrói a partir da fonte; se falhar, usa a última cópia guardada."""
        if refresh:
            try:
                docs = build_docs()
                KB_PATH.parent.mkdir(parents=True, exist_ok=True)
                KB_PATH.write_text(json.dumps(docs, ensure_ascii=False, indent=1))
                return cls(docs)
            except Exception as exc:
                print(f"[kb] não consegui ler a fonte {KB_SOURCE!r} ({exc}); uso {KB_PATH}", file=sys.stderr)
        return cls(json.loads(KB_PATH.read_text()) if KB_PATH.exists() else [])

    def search(self, query: str, max_chars: int = 2400, k: int = 6, lang: str | None = None):
        """Melhores documentos até max_chars, um por entidade.

        Com `lang`, os documentos nessa língua sobem e os de outra língua descem;
        entre versões da mesma entidade fica a da língua pedida. Os modelos
        pequenos tendem a responder na língua dos excertos.
        """
        q = tokens(query)
        if not self.bm25 or not q:
            return []
        scores = list(self.bm25.get_scores(q))
        if lang:
            for i, d in enumerate(self.docs):
                if d["lang"] == lang:
                    scores[i] *= 1.5
                elif d["lang"] != "any":
                    scores[i] *= 0.7
        ranked = sorted(range(len(self.docs)), key=lambda i: scores[i], reverse=True)
        out, seen, used = [], set(), 0
        for i in ranked:
            d = self.docs[i]
            if scores[i] <= 0 or len(out) >= k:
                break
            if d["key"] in seen or used + len(d["text"]) > max_chars:
                continue
            seen.add(d["key"])
            used += len(d["text"])
            out.append(d)
        return out


if __name__ == "__main__":
    kb = KnowledgeBase.load()
    print(f"{len(kb.docs)} documentos em {KB_PATH} (fonte: {KB_SOURCE})")
    if len(sys.argv) > 1:
        for d in kb.search(" ".join(sys.argv[1:])):
            print(f"\n--- {d['title']} [{d['lang']}] {d['url']}\n{d['text'][:400]}")

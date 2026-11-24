# Sobre este chatbot

Este chatbot é um exemplo de uma stack 100% open source: LangGraph e LangChain para a lógica,
llama.cpp para correr o modelo localmente, BM25 para pesquisar a knowledge base e,
opcionalmente, Langfuse para observabilidade. Não usa nenhuma API paga.

Substitui os ficheiros desta pasta (`knowledge/`) pelo conteúdo do teu negócio:
um ficheiro Markdown por tema, com um título `#` por secção.

# Que modelo usa?

Por defeito usa o Qwen3-1.7B (licença Apache-2.0), quantizado em Q4_K_M, que corre em CPU.
O modelo é descarregado automaticamente do HuggingFace no primeiro arranque.
Podes trocar por outro modelo GGUF com a variável `LLM_HF_MODEL`.

# Que hardware é preciso?

Corre em CPU, sem GPU. Para o Qwen3-1.7B chegam 2 vCPU e 2 GB de RAM livres.
Com o Langfuse ligado são precisos mais cerca de 2 GB de RAM.
Num servidor de 4 vCPU, uma resposta típica demora entre 2 e 9 segundos.

# Como funciona a resposta?

1. A pergunta é pesquisada na knowledge base com BM25 (palavras-chave).
2. Os excertos mais relevantes são enviados ao modelo, junto com o histórico recente da conversa.
3. O modelo responde apenas com base nesses excertos; se a informação não existir, diz que não sabe.
4. As primeiras perguntas repetidas são respondidas de uma cache, sem chamar o modelo.

# Onde ficam guardadas as conversas?

Numa base de dados SQLite no volume `data/`, através do checkpointer do LangGraph.
Se o Langfuse estiver ligado, cada pedido também fica registado lá, para análise.

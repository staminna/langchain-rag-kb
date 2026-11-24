#!/usr/bin/env bash
# Cria o .env a partir do .env.example, com segredos aleatórios.
#   ./scripts/init-env.sh             # só llama.cpp + chatbot
#   ./scripts/init-env.sh --langfuse  # também o Langfuse
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then echo ".env já existe; não o substituo." >&2; exit 1; fi

hex() { openssl rand -hex "${1:-24}"; }
uuid() { python3 -c 'import uuid; print(uuid.uuid4())'; }
set_var() {  # set_var NOME valor
  python3 - "$1" "$2" <<'PY'
import re, sys
name, value = sys.argv[1], sys.argv[2]
text = open(".env").read()
text = re.sub(rf"(?m)^{name}=.*$", lambda _: f"{name}={value}", text)
open(".env", "w").write(text)
PY
}

umask 077
cp .env.example .env
for v in POSTGRES_PASSWORD CLICKHOUSE_PASSWORD MINIO_ROOT_PASSWORD REDIS_AUTH SALT; do set_var "$v" "$(hex)"; done
set_var ENCRYPTION_KEY "$(hex 32)"
set_var NEXTAUTH_SECRET "$(hex 32)"
set_var LANGFUSE_PUBLIC_KEY "pk-lf-$(uuid)"
set_var LANGFUSE_SECRET_KEY "sk-lf-$(uuid)"
set_var LANGFUSE_ADMIN_PASSWORD "$(hex 12)"

if [ "${1:-}" = "--langfuse" ]; then
  set_var COMPOSE_PROFILES langfuse
  set_var LANGFUSE_HOST http://langfuse-web:3000
  echo "Langfuse ligado: http://localhost:3000 (email e password do admin no .env)"
fi
echo ".env criado (chmod 600). Agora: docker compose up -d"

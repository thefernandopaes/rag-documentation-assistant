# 🚀 Guia Rápido - Executar em Localhost

## Pré-requisitos

- **Python 3.11+** instalado
- **Chave da API OpenAI** (obter em https://platform.openai.com/api-keys)
- **Git** (para clonar o repositório)
- **PostgreSQL** (opcional - pode usar SQLite para desenvolvimento)

---

## Passo 1: Preparar o Ambiente

### 1.1. Ativar o ambiente virtual (se já existe)

```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 1.2. OU criar novo ambiente virtual

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 1.3. Instalar dependências

```bash
# Usando pip com pyproject.toml
pip install -e .

# OU usando uv (mais rápido)
uv pip install -e .
```

---

## Passo 2: Configurar Variáveis de Ambiente

### 2.1. Copiar arquivo de exemplo

```bash
cp .env.example .env
```

### 2.2. Editar `.env` com suas configurações

**Configuração MÍNIMA para desenvolvimento local:**

```env
# === OBRIGATÓRIO ===
OPENAI_API_KEY=sk-proj-...sua-chave-aqui...

# === SECRETS (gerar valores aleatórios) ===
SESSION_SECRET=seu-secret-aleatorio-64-caracteres-hex
ADMIN_API_KEY=seu-admin-key-aleatorio-64-caracteres-hex

# === DATABASE (SQLite para desenvolvimento) ===
DATABASE_URL=sqlite:///./data/docrag.db

# === ChromaDB (armazenamento local) ===
CHROMA_DB_PATH=./data/chroma_db

# === Ambiente ===
ENV=development

# === FastAPI ===
FASTAPI_ENABLED=true
PORT=8000

# === Opcional: usar documentação de exemplo ===
DOC_USE_SAMPLE=true
```

**Para gerar secrets aleatórios:**

```bash
# Windows PowerShell
-join ((48..57) + (65..70) | Get-Random -Count 64 | % {[char]$_})

# Linux/Mac
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## Passo 3: Inicializar o Banco de Dados

### 3.1. Criar diretório de dados

```bash
mkdir -p data
```

### 3.2. Executar migrations do Alembic

```bash
# Criar o banco e tabelas
alembic upgrade head
```

### 3.3. Verificar que o banco foi criado

```bash
# Windows
dir data

# Linux/Mac
ls -lh data
```

Você deve ver `docrag.db` (SQLite) ou conexão PostgreSQL funcionando.

---

## Passo 4: Inicializar Documentação (Opcional)

### 4.1. Usar documentação de exemplo

Se configurou `DOC_USE_SAMPLE=true`, a documentação será carregada automaticamente na primeira execução.

### 4.2. OU carregar documentação real

Edite `config.py` e adicione suas fontes em `DOC_SOURCES`:

```python
DOC_SOURCES = [
    "https://fastapi.tiangolo.com",
    "https://docs.python.org/3/",
    # suas URLs aqui
]
```

Depois execute o endpoint de inicialização (após iniciar o servidor):

```bash
curl -X POST http://localhost:8000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"force_reload": false}'
```

---

## Passo 5: Executar o Servidor

### Opção A: FastAPI com Uvicorn (Recomendado - Assíncrono)

```bash
# Desenvolvimento (com auto-reload)
uvicorn fastapi_app:app --reload --host 0.0.0.0 --port 8000

# OU produção local
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 2
```

### Opção B: Flask com Gunicorn (Legado - Síncrono)

```bash
# Desenvolvimento
flask --app app run --debug --port 5000

# OU produção local
gunicorn -c gunicorn.conf.py app:app
```

### Verificar que está rodando

```bash
# Health check
curl http://localhost:8000/health

# Resposta esperada:
# {"status": "healthy", "timestamp": "..."}
```

---

## Passo 6: Acessar a Aplicação

### Interface Web

Abra seu navegador em:

- **FastAPI**: http://localhost:8000
- **Flask**: http://localhost:5000

### Documentação Interativa da API (FastAPI)

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Testar Chat

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is FastAPI?",
    "session_id": "test-session"
  }'
```

---

## Comandos Úteis

### Verificar logs

```bash
# Uvicorn mostra logs no console automaticamente
# Para salvar em arquivo:
uvicorn fastapi_app:app --log-config logging.conf > logs/server.log 2>&1
```

### Resetar banco de dados

```bash
# CUIDADO: Apaga todos os dados!
rm data/docrag.db data/chroma_db -rf
alembic upgrade head
```

### Rodar testes

```bash
# Todos os testes
pytest -v

# Testes específicos
pytest test_agent.py -v

# Com cobertura
pytest --cov=. --cov-report=html
```

### Verificar estatísticas

```bash
curl http://localhost:8000/api/stats
```

### Limpar cache

```bash
curl -X POST http://localhost:8000/api/cache/clear \
  -H "X-Admin-Key: seu-admin-key-aqui"
```

---

## Estrutura de Diretórios

```
rag-documentation-assistant/
├── data/                    # Dados locais (criado automaticamente)
│   ├── docrag.db           # Banco SQLite
│   └── chroma_db/          # Vector database ChromaDB
├── .venv/                   # Ambiente virtual Python
├── migrations/              # Migrations Alembic
├── templates/               # Templates HTML (Flask)
├── static/                  # Assets estáticos
├── fastapi_app.py          # Aplicação FastAPI (NOVO)
├── app.py                   # Aplicação Flask (LEGADO)
├── rag_engine.py           # Motor RAG síncrono
├── rag_engine_async.py     # Motor RAG assíncrono
├── config.py               # Configurações
├── .env                     # Variáveis de ambiente (criar)
└── pyproject.toml          # Dependências Python
```

---

## Troubleshooting

### Erro: "ModuleNotFoundError"

```bash
# Reinstalar dependências
pip install -e .
```

### Erro: "OPENAI_API_KEY not found"

Verifique que o arquivo `.env` existe e tem a chave correta:

```bash
cat .env | grep OPENAI_API_KEY
```

### Erro: "Database connection failed"

Para SQLite, verifique que o diretório `data/` existe:

```bash
mkdir -p data
alembic upgrade head
```

### Erro: "ChromaDB error"

Limpe o banco vetorial:

```bash
rm -rf data/chroma_db
# Reinicie o servidor para recriar
```

### Porta já em uso

```bash
# Windows - matar processo na porta 8000
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Performance lenta

1. Verifique se está usando FastAPI (mais rápido)
2. Aumente workers: `--workers 4`
3. Use cache: verifique `CACHE_TTL` no `.env`

---

## Próximos Passos

1. ✅ Servidor rodando em localhost
2. 📚 Adicionar suas fontes de documentação em `config.py`
3. 🔍 Testar queries no chat
4. 🧪 Rodar testes: `pytest -v`
5. 🚀 Fazer deploy (Railway, Heroku, etc.)

---

## Recursos Adicionais

- **README.md** - Documentação completa do projeto
- **agents/** - Documentação de desenvolvimento (FASE 1-9)
- **tests/** - Suite completa de testes
- **.env.example** - Todas as variáveis de ambiente disponíveis

---

## Suporte

Se encontrar problemas:

1. Verifique os logs do servidor
2. Consulte a seção Troubleshooting acima
3. Rode os testes: `pytest -v`
4. Verifique as issues no GitHub

**Pronto! Você deve ter o servidor rodando em http://localhost:8000** 🎉

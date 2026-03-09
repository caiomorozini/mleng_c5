# PÓS TECH Datathon - Passos Mágicos (MLOps)

## 1) Visão Geral do Projeto

### Objetivo
Construir um modelo preditivo para estimar o risco de defasagem escolar de estudantes atendidos pela Associação Passos Mágicos.

Definição de alvo usada no projeto:
- `risco_defasagem = 1` quando `Defas < 0`
- `risco_defasagem = 0` quando `Defas >= 0`

### Solução Proposta
Foi implementada uma pipeline completa de Machine Learning com:
- carregamento e tratamento de dados;
- engenharia de atributos;
- pré-processamento com imputação + normalização + one-hot;
- treino e seleção de modelo;
- serialização de artefatos com `joblib/json`;
- API FastAPI com endpoint `/predict`;
- monitoramento contínuo com logs e endpoints de drift (`/drift`, `/drift-dashboard`);
- containerização com Docker.

### Stack Tecnológica
- Linguagem: Python 3.12+
- ML: `scikit-learn`, `pandas`, `numpy`
- API: `FastAPI`, `uvicorn`
- Serialização: `joblib`, `json`
- Testes: `pytest`, `pytest-cov`
- Empacotamento: Docker + Docker Compose
- Deploy: local (Docker)
- Monitoramento: logging + dashboard de drift em HTML

## 2) Estrutura do Projeto

```text
mleng_c5/
├── app/
│   ├── main.py                  # Inicialização da API
│   ├── routes.py                # Endpoints (health, predict, drift)
│   ├── schemas.py               # Schemas Pydantic
│   ├── config.py                # Carregamento de artefatos e estado do modelo
│   └── model/
│       └── drift_monitor.py     # Cálculo e dashboard de drift
├── src/
│   ├── utils.py                 # I/O, serialização, helpers
│   ├── feature_engineering.py   # Features derivadas
│   ├── preprocessing.py         # Seleção de colunas e preprocessor
│   ├── evaluate.py              # Métricas e threshold tuning
│   └── train.py                 # Treino, seleção, e persistência de artefatos
├── data/
│   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx
├── models/                      # Gerado após treino
│   ├── predictor.joblib
│   ├── model_config.json
│   ├── metrics.json
│   └── reference_profile.json
├── tests/                       # Testes unitários
├── Dockerfile
├── docker-compose.yml
├── deploy.sh
└── pyproject.toml
```

## 3) Instruções de Deploy

### Pré-requisitos
- Python 3.12+
- `pip`
- Docker e Docker Compose

### Instalação de dependências
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Treinamento do modelo
```bash
PYTHONPATH=. python -m src.train
```

Isso gera os artefatos em `models/`.

### Executar API local (sem Docker)
```bash
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Executar com Docker
```bash
docker compose build
docker compose up -d
```

ou:

```bash
chmod +x deploy.sh
./deploy.sh
```

## 4) Exemplos de Chamadas à API

### Health check
```bash
curl -X GET "http://localhost:8000/health"
```

### Predição
```bash
curl -X POST "http://localhost:8000/predict" \
	-H "Content-Type: application/json" \
	-d '{
		"records": [
			{
				"values": {
					"Fase": 7,
					"Turma": "A",
					"Idade 22": 17,
					"Gênero": "Menina",
					"Ano ingresso": 2019,
					"Instituição de ensino": "Escola Pública",
					"INDE 22": 6.8,
					"IEG": 5.7,
					"IDA": 6.5,
					"IPV": 6.1,
					"IAN": 5.2,
					"Matem": 6.0,
					"Portug": 6.5,
					"Inglês": 7.0,
					"Indicado": "Não",
					"Atingiu PV": "Sim"
				}
			}
		]
	}'
```

Saída esperada (exemplo):
```json
{
	"model_name": "logistic_regression",
	"threshold": 0.43,
	"predictions": [
		{
			"risco_probabilidade": 0.71,
			"risco_defasagem": 1
		}
	]
}
```

### Drift
```bash
curl -X GET "http://localhost:8000/drift"
curl -X GET "http://localhost:8000/drift-dashboard"
```

## 5) Etapas do Pipeline de Machine Learning

- Pré-processamento dos dados: remoção de colunas de identificação/leakage, imputação de faltantes, padronização numérica e one-hot para categóricas.
- Engenharia de features: tempo no programa, médias de notas e índices, flags binárias para atributos booleanos.
- Treinamento e validação: split estratificado treino/teste, comparação entre Regressão Logística e Random Forest.
- Seleção de modelo: escolha do melhor candidato por F1 com restrição mínima de recall.
- Pós-processamento: aplicação de threshold ótimo para classificação final de risco.

## Qualidade e Testes

- Testes unitários implementados com `pytest`.
- Cobertura aferida: **92.24%**.

Executar testes:
```bash
python -m pytest -q -o addopts=''
```

Executar cobertura:
```bash
python -m pytest -q --cov=src --cov=app --cov-report=term-missing -o addopts=''
```

## Link da API

- Local: `http://localhost:8000`
- Docs interativa: `http://localhost:8000/docs`

---

Projeto preparado para apresentação técnica e evolução para deploy em cloud (AWS/GCP/Heroku/Cloud Run) mantendo a mesma imagem Docker.
# PhishRAG

> Intelligent phishing-email analysis with XGBoost risk scoring and LightRAG-based MITRE ATT&CK retrieval.

PhishRAG is a course project for the 114-2 Agentic AI class at National Taiwan University of Science and Technology. It combines a machine-learning pipeline with a graph-aware retrieval system to turn an email into an explainable threat-intelligence report.

The project addresses a practical gap in phishing workflows: a binary filter can flag a message, but analysts also need to understand the likely technique, supporting evidence, relevant ATT&CK knowledge, and possible mitigation or detection context.

## What it does

Given an email and an optional analyst question, PhishRAG:

1. extracts lexical, structural, sender, URL, and suspicious-language signals;
2. estimates phishing risk with a binary XGBoost model;
3. classifies the phishing type and severity with downstream XGBoost models;
4. retrieves related MITRE ATT&CK knowledge through LightRAG; and
5. returns a grounded threat-intelligence report with follow-up question support.

## System architecture

The application runs as four Docker Compose services.

```text
Streamlit frontend (:8501)
        |
        | HTTP REST
        v
Flask backend (:5000)
  - ML inference
  - LightRAG request proxy
        |
        | POST /query
        v
LightRAG API (:9621) ----------------> Ollama-compatible LLM and embedding endpoint
        |
        v
PostgreSQL (:5432)
  - pgvector for vector search
  - Apache AGE for graph storage
  - document and key-value storage
```

The Streamlit interface accepts the email, an optional analysis instruction, a retrieval mode, and advanced retrieval parameters. The Flask backend runs the ML stages, builds the retrieval query, and returns both the model output and the LightRAG response.

## Analysis pipeline

### Layer 1: feature engineering

The pipeline derives email-text features from message structure, suspicious-language patterns, sender and URL signals, and TF-IDF terms. It includes explicit screening for features that could leak labels or encode artefacts of the synthetic dataset.

### Layer 2: phishing risk detection

**Model A** is an XGBoost binary classifier that estimates whether an email is phishing or legitimate and produces a risk score.

### Layer 3: phishing type and severity

**Model B** predicts phishing type. **Model C** predicts low, medium, or high severity and uses Model B probability outputs as additional inputs.

### Layer 4: LightRAG threat-intelligence retrieval

LightRAG queries a MITRE ATT&CK knowledge graph for relevant techniques, detection guidance, analytical IDs, relationships, and mitigations. Supported retrieval modes include `hybrid`, `local`, `global`, `mix`, `naive`, and `bypass`.

## Knowledge base

The ATT&CK knowledge base uses a hub-and-spoke graph centred on phishing techniques and connects them to mitigations, detection sources, detection analytics, threat actors, malware, campaigns, tools, reconnaissance, resource development, and execution stages.

The graph schema includes entities such as ATT&CK techniques, tactics, threat actors, tools, and mitigations. Important relationships include `paired_with`, `uses`, `enables`, and `mitigates`. This structure supports multi-hop questions that require more than a single document match.

## Why graph-aware retrieval

The course evaluation compares naive vector retrieval with hybrid graph-plus-vector retrieval.

| Query type | Observed result |
| --- | --- |
| Factual lookup | Both approaches can return accurate answers; hybrid responses are generally more concise. |
| Structured retrieval | Hybrid retrieval preserves ATT&CK identifiers and graph-linked context more consistently across documents. |
| Situational analysis | Hybrid retrieval can return detection IDs, analytics, and tunable parameters grounded in graph nodes. |
| Out-of-scope queries | Both approaches can decline conservatively; hybrid retrieval can add adjacent facts that are present in the graph. |

The graph does not make the model inherently "smarter." Its value comes from storing names, IDs, and relationships as structured data, which makes multi-hop and identifier-sensitive retrieval more reliable and reduces the risk of filling missing structured facts with generated text.

## Explainability

SHAP analysis explains which features contribute to an individual prediction. In the course experiment, strong phishing signals included the aggregate suspicious-language score, account-related TF-IDF terms, and link-related TF-IDF terms. Features associated with common legitimate business-email patterns pushed predictions toward the legitimate class.

## Evaluation notes

The class presentation reports the following results on the supplied **synthetic** email dataset:

| Stage | Task | Reported result |
| --- | --- | --- |
| Model A | Binary phishing detection | Accuracy 1.00, ROC-AUC 1.00 |
| Model B | Phishing-type classification | Accuracy 1.00, macro F1 1.00 |
| Model C | Severity classification | Accuracy 0.87, macro F1 0.730 |

These figures describe a course experiment on synthetic data, not production performance. Real email environments require separate validation on representative, held-out data and should retain analyst review before any security action.

## Dataset and references

- Synthetic email dataset: 10,000 emails, including 4,000 legitimate emails and 6,000 phishing emails across 10 phishing types. Source: [Phishing and Legitimate Emails Dataset](https://www.kaggle.com/datasets/kuladeep19/phishing-and-legitimate-emails-dataset)
- MITRE ATT&CK knowledge base: [attack-stix-data](https://github.com/mitre-attack/attack-stix-data) and the [MITRE ATT&CK framework](https://attack.mitre.org/)
- Guo et al. (2024), [LightRAG](https://arxiv.org/abs/2410.05779)
- Lundberg and Lee (2017), [SHAP](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- Es et al. (2023), [RAGAS](https://arxiv.org/abs/2309.15217)
- Edge et al. (2024), [GraphRAG](https://arxiv.org/abs/2404.16130)

## Run locally

### Prerequisites

- Docker Desktop with Docker Compose
- An accessible Ollama-compatible LLM and embedding endpoint configured in `LightRAG/.env`
- A prepared LightRAG database or knowledge-base initialization data

The complete LightRAG SQL backup is intentionally excluded from this repository because of its size. Prepare or restore the knowledge base before expecting graph-backed answers.

### Start the stack

```bash
git clone https://github.com/dashubigtree/114-2-Agentic-AI-class.git
cd 114-2-Agentic-AI-class
docker compose up --build -d
docker compose ps
```

When all services are healthy, open [http://localhost:8501](http://localhost:8501).

### Stop the stack

```bash
docker compose down
```

Use `docker compose down -v` only when you intend to delete the local PostgreSQL volume and its stored knowledge-base data.

## Repository guide

| Path | Purpose |
| --- | --- |
| `PhishRAG/` | Flask API, Streamlit dashboard, ML models, and inference pipeline |
| `LightRAG/` | LightRAG service configuration and storage directories |
| `lightrag-package/` | Database initialization assets |
| `docker-compose.yml` | Four-service local deployment |
| [`使用說明書.md`](使用說明書.md) | Detailed Chinese user guide |
| [`技術文件.md`](技術文件.md) | Detailed Chinese technical documentation |

## Scope and responsible use

PhishRAG was created for coursework and research demonstration. It analyses supplied email text and retrieves public ATT&CK knowledge. Do not treat its score, classification, or generated report as a standalone security decision, and do not submit confidential email content to an endpoint that has not been approved for that data.

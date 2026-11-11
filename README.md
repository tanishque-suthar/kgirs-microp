# How Much Can We Trust Automatic Evaluation of RAG Answers?

A Streamlit research mini-project investigating the reliability, alignment, and biases of automatic evaluation metrics (RAGAS and LLM-as-a-Judge) against human ground truth labels on 100 RAGTruth Question-Answering responses.

---

## 1. Project Overview

Automatic evaluators (such as RAGAS and LLM judges) are standard practice for assessing RAG pipelines. However, practitioners often wonder:
- How well do continuous metrics like Faithfulness and Answer Relevancy correlate with discrete human quality grades?
- Are LLM judges susceptible to position bias (option ordering) and verbosity bias (rewarding neutral fluff)?
- Does high retrieval quality (MRR, NDCG) guarantee faithful answer generation?
- How closely do automatic evaluators agree with an independent human annotator?

This project investigates these questions on a stratified 100-sample benchmark from the official RAGTruth QA dataset.

---

## 2. Architecture & Pipeline Modules

The project is structured cleanly into modular components:

```
kgirs-microp/
├── app.py                      # Streamlit entrypoint & Page 1: Overview
├── pages/
│   ├── 1_Overview.py           # Page 1 Mirror
│   ├── 2_Annotate.py           # Page 2: Blind relabelling UI (subset of 35)
│   ├── 3_RAGAS_vs_Human.py     # Page 3: RAGAS vs human correlations & plots
│   ├── 4_LLM_Judge.py          # Page 4: LLM Judge accuracy, kappa, position/verbosity bias
│   ├── 5_Agreement.py          # Page 5: Mini-IAA (pairwise & Fleiss' kappa)
│   ├── 6_Retrieval.py          # Page 6: Retrieval MRR/NDCG vs answer quality
│   └── 7_Limitations.py        # Page 7: Comprehensive study limitations
├── src/
│   ├── config.py               # Central constants, paths, seed=42, model config
│   ├── gemini_client.py        # Unified Gemini client with caching and backoff
│   ├── sampling.py             # RAGTruth QA loader, stratified sampling (40/30/30)
│   ├── ragas_eval.py           # RAGAS Faithfulness & Answer Relevancy with disk cache
│   ├── llm_judge.py            # Gemini judge, position-bias check, verbosity fluff test
│   ├── retrieval_metrics.py    # MRR, NDCG, and proxy passage relevance
│   ├── stats.py                # Spearman, Kendall, Bootstrap 95% CIs, Cohen's & Fleiss' kappa
│   └── ui_helpers.py           # Shared data loaders, sidebar controls, export bundler
├── data/                       # CSV datasets and results
├── cache/                      # Disk caches for API calls (JSON)
├── requirements.txt
└── README.md
```

---

## 3. Setup & Installation

### Step 1: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Gemini API Key
Obtain a free Gemini API key from [Google AI Studio](https://aistudio.google.com/). Set it via an environment variable:
```bash
export GEMINI_API_KEY="your-api-key-here"
```
Or create a `.env` file in the project root:
```bash
echo "GEMINI_API_KEY=your-api-key-here" > .env
```
*(Alternatively, you can enter the API key directly into the Streamlit sidebar at runtime).*

---

## 4. Running the Application

Launch the Streamlit app:
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 5. Running Pipeline Modules via CLI

Each module can also be run independently from the command line:

```bash
# Generate stratified 100-sample dataset
python src/sampling.py

# Run RAGAS smoke test (5 items)
python src/ragas_eval.py

# Run LLM Judge smoke test (5 items with position & fluff tests)
python src/llm_judge.py

# Run Retrieval smoke test (5 items)
python src/retrieval_metrics.py
```

---

## 6. Key Methodological Features

- **Stratified Sampling**: 100 answers (~40 correct / 30 partly / 30 incorrect) sampled reproducibly using fixed random seed (`seed=42`).
- **Position Bias Test**: Every answer is judged twice with rubric options listed in swapped order. Disagreements are tracked as position flips.
- **Verbosity Bias Test**: A subset of ~20 answers has uninformative neutral fluff appended and is re-evaluated to quantify score changes.
- **Robust Caching**: Every LLM call, embedding, and metric calculation is cached to disk in `cache/` so reruns never re-call the Gemini API.
- **Rate Limit Handling**: Exponential backoff with jitter handles free-tier Gemini rate limits gracefully.
- **Export Data**: A dedicated sidebar button allows one-click export of all results as a ZIP bundle.
- **Clean UI**: Strict compliance with no-emoji design guidelines across all pages and charts.

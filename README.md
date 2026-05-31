# Retrieval-Augmented Post-Correction for Domain-Specific ASR in Air Traffic Control

Course project for 261R Natural Language Processing (Korea University, Spring 2026).

This repository implements a Retrieval-Augmented Generation (RAG) framework
that corrects domain-specific errors in Air Traffic Control (ATC) ASR transcripts
without retraining the ASR model.

## Overview

Standard ASR models assign near-zero probability to ATC-specific entities
(callsigns, ICAO commands, facility names) due to the out-of-vocabulary (OOV) problem.
This project addresses the gap by retrieving relevant domain knowledge at inference time
and using an LLM to select the best correction.

```
Speech input
  -> ASR model (Whisper / Wav2Vec2 UWB-ATCC)
  -> BERT-based NER (entity detection)
  -> RAG correction
       Callsign  : OpenSky API + ICAO Doc 8585 airline codes
       Command   : ICAO Doc 9432 phraseology (21 entries)
       Facility  : file_key metadata prior + airport database
  -> LLM reranker (Groq / Gemini / Ollama)
  -> Corrected transcript
```

## Repository Structure

```
261RCOSE46101/
├── notebooks/
│   ├── 00_data_preparation.ipynb     # Extract gold annotations from ATCO2 XML
│   ├── 01_RAG_pipeline.ipynb         # Main RAG post-correction pipeline
│   └── 02_ASR_comparison.ipynb       # Wav2Vec2 vs Whisper baseline comparison
├── scripts/
│   └── atcrag.py                     # Standalone Python version of the pipeline
├── data/
│   └── README.md                     # Data download instructions
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Clone the repository
```bash
git clone https://github.com/skwon616/261RCOSE46101
cd 261RCOSE46101
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set API key for LLM backend (choose one)
```bash
export GROQ_API_KEY=your_key      # https://console.groq.com (free)
export GEMINI_API_KEY=your_key    # https://aistudio.google.com (free)
# Or use Ollama locally: ollama pull llama3.2
```

4. Download the ATCO2 corpus from https://www.atco2.org and place the `.tgz` file in `data/`

## Execution Order

```
00_data_preparation.ipynb   ->  atco2_gold_clean.csv
02_ASR_comparison.ipynb     ->  baseline_result.csv
01_RAG_pipeline.ipynb       ->  atc_rag_results.csv
```

## Results

Evaluation on a 60-utterance sample (random_state=42) from the ATCO2 corpus.

| Model | WER | BoW F1 | Callsign F1 | Command F1 | Value F1 |
|---|---|---|---|---|---|
| Whisper (baseline) | 0.4959 | 0.5790 | 0.3487 | 0.1801 | 0.3540 |
| RAG (ours) | 0.7338 | 0.4578 | 0.2738 | 0.1554 | **0.3672** |
| Wav2Vec2 (baseline) | 0.4886 | 0.6022 | 0.3791 | 0.1929 | 0.3770 |
| RAG on Wav2Vec2 | 0.6811 | 0.4969 | 0.3814 | 0.1444 | **0.4131** |

RAG improves Value F1 consistently across both ASR backends.
Overall WER increase is attributed to LLM over-correction on non-OOV tokens.

> Note: Due to API token constraints, evaluation was performed on 60 utterances
> (~11% of the full ATCO2 test set). Full-scale evaluation is left for future work.

## View Notebooks

GitHub preview may fail for large notebooks.
Use nbviewer: https://nbviewer.org/github/skwon616/261RCOSE46101/blob/main/notebooks/01_RAG_pipeline.ipynb

## Dataset

[ATCO2 Corpus](https://www.atco2.org/) — Zuluaga-Gomez et al., 2022.
Raw audio and XML annotations are not included in this repository.

## Course

261R Natural Language Processing (Korea University, Spring 2026)

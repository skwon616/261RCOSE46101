"""
ATC RAG-based ASR Post-Correction Pipeline
===========================================
This script implements a Retrieval-Augmented Generation (RAG) framework
for post-correcting Air Traffic Control (ATC) ASR transcripts.

Pipeline:
    ASR output (Whisper / Wav2Vec2)
        → NER-based entity detection (BERT)
        → Entity-specific RAG correction
            ├── Instruction  → ICAO Doc 9432 Vector DB
            ├── Callsign     → OpenSky API + ICAO Airline Codes (Doc 8585)
            └── Facility     → file_key prior + Airport DB
        → LLM reranking (Groq / Gemini / Ollama)
        → Evaluation: WER / NER F1

Usage:
    export GROQ_API_KEY=your_key      # or GEMINI_API_KEY
    python atcrag.py

Requirements:
    pip install faiss-cpu sentence-transformers jiwer transformers
                requests pandas numpy editdistance soundfile groq
"""

import os
import re
import math
import json
import torch
import numpy as np
import pandas as pd
import requests
import editdistance
from functools import lru_cache

# ==============================================================
# STEP 0. Configuration
# ==============================================================
# Set data file paths (place in same directory as this script)
BASELINE_CSV = 'baseline_result.csv'   # Whisper ASR output
GOLD_CSV     = 'atco2_gold_clean.csv'  # Gold annotation

# WAV directory for direct audio inference (optional)
WAV_DIR = '1hour_train/test/'

# LLM backend selection: "groq" | "gemini" | "ollama"
# Groq and Gemini require API keys set as environment variables.
# Ollama runs locally (no API key needed): run `ollama pull llama3.2` first.
BACKEND = "groq"

assert os.path.exists(BASELINE_CSV), f'File not found: {BASELINE_CSV}'
assert os.path.exists(GOLD_CSV),     f'File not found: {GOLD_CSV}'


# ==============================================================
# STEP 1. LLM Backend Initialization
# ==============================================================
# call_claude(prompt) is a unified interface regardless of backend.
# The function name reflects its original Anthropic Claude origin,
# but routes to whichever backend is configured above.

if BACKEND == "groq":
    from groq import Groq as _Groq
    _groq_client = _Groq(api_key=os.environ.get("GROQ_API_KEY"))
    _GROQ_MODEL  = "llama-3.1-8b-instant"

    def call_claude(prompt: str, max_tokens: int = 300) -> str:
        resp = _groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

elif BACKEND == "ollama":
    _OLLAMA_MODEL = "llama3.2"

    def call_claude(prompt: str, max_tokens: int = 300) -> str:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": _OLLAMA_MODEL, "prompt": prompt,
                  "stream": False, "options": {"num_predict": max_tokens}},
            timeout=60,
        )
        return resp.json()["response"].strip()

elif BACKEND == "gemini":
    import google.generativeai as _genai
    _genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    _gemini_model = _genai.GenerativeModel("gemini-2.5-flash")

    def call_claude(prompt: str, max_tokens: int = 300) -> str:
        resp = _gemini_model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.0},
        )
        return resp.text.strip()

else:
    raise ValueError(f"Unknown BACKEND: {BACKEND}")

print(f"LLM backend: {BACKEND}")


# ==============================================================
# STEP 2-A. FAISS Vector Database
# ==============================================================
# Encodes text documents into dense vectors using a sentence encoder
# and builds a FAISS index for cosine similarity retrieval.
# Used separately for instruction, callsign, and facility knowledge bases.

from sentence_transformers import SentenceTransformer
import faiss

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


class FAISSVectorDB:
    """
    Lightweight vector store built on FAISS IndexFlatIP.
    Documents are L2-normalized before indexing so inner product
    equals cosine similarity.
    """

    def __init__(self, docs: list[str]):
        self.docs = docs
        vecs = embedder.encode(docs, show_progress_bar=False).astype('float32')
        faiss.normalize_L2(vecs)
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)

    def search(self, query: str, k: int = 5) -> list[str]:
        vec = embedder.encode([query]).astype('float32')
        faiss.normalize_L2(vec)
        scores, indices = self.index.search(vec, k)
        return [self.docs[i] for i in indices[0] if i < len(self.docs)]


# ==============================================================
# STEP 2-B. BERT-based NER Model
# ==============================================================
# Fine-tuned on ATCO2 corpus to detect ATC-specific entities:
#   - CALLSIGN : aircraft identifiers (e.g., "Swiss Two Zero Four Kilo")
#   - COMMAND  : ATC instructions (e.g., "climb", "cleared to land")
#   - VALUE    : numeric values (headings, altitudes, frequencies)
#
# aggregation_strategy='first' reduces subword fragmentation.
# Confidence threshold 0.4: spans below this are left uncorrected.

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline as hf_pipeline
)

NER_MODEL_ID = 'Jzuluaga/bert-base-ner-atc-en-atco2-1h'

_ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_ID)
_ner_model     = AutoModelForTokenClassification.from_pretrained(NER_MODEL_ID)

ner_model = hf_pipeline(
    'token-classification',
    model='./atc_ner_best',      # fine-tuned checkpoint (falls back to NER_MODEL_ID)
    aggregation_strategy='first',
    device=-1                    # CPU inference
)
print('NER model loaded')

_test = 'swiss two zero four kilo climb to flight level three four zero'
for e in ner_model(_test):
    print(f'  [{e["entity_group"]:10s}] {e["word"]:30s} score={e["score"]:.3f}')


# ==============================================================
# STEP 3-A. Instruction Knowledge Base (ICAO Doc 9432)
# ==============================================================
# 21 standard ATC phraseology entries compiled from ICAO Doc 9432.
# Each entry is a (command, full phrase, context) triplet stored
# as a plain string and indexed in FAISS for semantic retrieval.

ICAO_PHRASES = [
    ('climb',              'climb to flight level [FL]',            'altitude instruction ascending aircraft'),
    ('descend',            'descend to flight level [FL]',          'altitude instruction descending aircraft'),
    ('cleared to land',    'cleared to land runway [RWY]',          'landing clearance given to aircraft'),
    ('line up and wait',   'line up runway [RWY] and wait',         'aircraft holds on runway before takeoff'),
    ('cleared for takeoff','cleared for takeoff runway [RWY]',      'takeoff clearance issued'),
    ('contact',            'contact [FACILITY] [FREQ]',             'frequency handoff to next sector'),
    ('squawk',             'squawk [CODE]',                         'transponder code assignment'),
    ('radar contact',      'radar contact',                         'controller confirms radar identification'),
    ('turn left heading',  'turn left heading [HDG]',               'directional instruction left'),
    ('turn right heading', 'turn right heading [HDG]',              'directional instruction right'),
    ('maintain',           'maintain flight level [FL]',            'hold current altitude'),
    ('report',             'report [WAYPOINT/POSITION]',            'position reporting instruction'),
    ('frequency change',   'frequency change approved',             'pilot approved to change frequency'),
    ('hold short',         'hold short of runway [RWY]',            'ground instruction stop before runway'),
    ('taxi via',           'taxi via [TAXIWAY] to [STAND/RWY]',     'ground movement instruction'),
    ('continue',           'continue approach',                     'approach continuation clearance'),
    ('go around',          'go around',                             'missed approach instruction'),
    ('speed',              'reduce speed to [KT] knots',            'speed restriction'),
    ('wind',               'wind [DIR] degrees [SPD] knots',        'wind information'),
    ('qnh',                'QNH [VALUE]',                           'altimeter setting instruction'),
    ('cross',              'cross runway [RWY]',                    'ground instruction cross runway'),
]

instruction_docs = [f"{cmd}: {phrase} | {ctx}" for cmd, phrase, ctx in ICAO_PHRASES]
instruction_db   = FAISSVectorDB(instruction_docs)
INSTRUCTION_VOCAB = [cmd for cmd, _, _ in ICAO_PHRASES]


def edit_distance_match(word: str, threshold: int = 2):
    """Return closest instruction for a single word. None if dist > threshold."""
    candidates = [(w, editdistance.eval(word.lower(), w.lower())) for w in INSTRUCTION_VOCAB]
    best, dist = min(candidates, key=lambda x: x[1])
    return best if dist <= threshold else None


def correct_instruction(asr_text: str) -> str:
    """
    RAG-based instruction correction.

    Retrieves top-4 ICAO phrases semantically similar to the ASR
    hypothesis and passes them to the LLM with strict rules:
      - Fix phonetically confused commands (e.g. 'come' → 'climb')
      - Convert digits to spoken form (e.g. 'heading 270' → 'TWO SEVEN ZERO')
      - Preserve wind, runway, QNH, and frequency values unchanged
      - Do not modify callsigns or facility names
    """
    relevant  = instruction_db.search(asr_text, k=4)
    retrieved = "\n".join(f"  - {r}" for r in relevant)

    lines = [
        "You are an Air Traffic Control (ATC) transcript corrector.",
        f'ASR output (may contain errors): "{asr_text}"',
        "",
        "Relevant ICAO standard phrases retrieved:",
        retrieved,
        "",
        "## Task",
        "1. Correct ATC command words that are phonetically confused",
        "   (e.g. 'come' -> 'climb', 'ready' -> 'radar', 'out of' -> 'cross')",
        "2. Convert numeric digits in heading/altitude/FL to spoken form:",
        "   - heading 270       -> heading TWO SEVEN ZERO",
        "   - climb 5500 feet   -> climb FIVE THOUSAND FIVE HUNDRED feet",
        "   - FL 180            -> flight level ONE EIGHT ZERO",
        "3. CRITICAL - Do NOT convert these:",
        "   - wind direction: 'wind 040 degrees' -> leave unchanged",
        "   - runway numbers: 'runway 25' -> leave unchanged",
        "   - QNH values: 'QNH 1023' -> leave unchanged",
        "   - frequencies: '123.255' -> leave unchanged",
        "4. Do NOT change callsigns or facility names",
        "5. Return ONLY the corrected transcript, no explanation",
        "",
        "Corrected transcript:",
    ]
    return call_claude("\n".join(lines))


# ==============================================================
# STEP 3-B. Callsign Knowledge Base
# ==============================================================
# Callsign candidates are built from two sources:
#   1. OpenSky Network API: live flight callsigns near the airport
#   2. ICAO Doc 8585: airline telephony designators (Wikipedia scrape
#      with hardcoded fallback for 20+ major carriers)
#
# Each entry is stored as "SPOKEN FORM | ICAO: CODE" so the LLM
# receives both the expected pronunciation and the ICAO identifier.
#
# Word overlap score gates correction: if overlap < 0.40, the
# original ASR span is returned unchanged to avoid hallucination.

NUM_WORD_TO_DIGIT = {
    'zero':'0','one':'1','two':'2','three':'3','four':'4',
    'five':'5','six':'6','seven':'7','eight':'8','nine':'9',
}
PHONETIC_ALPHA = {
    'alpha':'A','bravo':'B','charlie':'C','delta':'D','echo':'E',
    'foxtrot':'F','golf':'G','hotel':'H','india':'I','juliet':'J',
    'kilo':'K','lima':'L','mike':'M','november':'N','oscar':'O',
    'papa':'P','quebec':'Q','romeo':'R','sierra':'S','tango':'T',
    'uniform':'U','victor':'V','whiskey':'W','x-ray':'X','yankee':'Y','zulu':'Z',
}

_AIRLINE_FALLBACK = {
    'ANA': ('ALL NIPPON',   'All Nippon Airways'),
    'SWR': ('SWISS',        'Swiss International Air Lines'),
    'EZS': ('EASY',         'easyJet Switzerland'),
    'DLH': ('LUFTHANSA',    'Lufthansa'),
    'BAW': ('SPEEDBIRD',    'British Airways'),
    'AFR': ('AIR FRANCE',   'Air France'),
    'KLM': ('KLM',          'KLM Royal Dutch Airlines'),
    'JTE': ('JETSTAR',      'Jetstar Airways'),
    'QFA': ('QANTAS',       'Qantas Airways'),
    'SIA': ('SINGAPORE',    'Singapore Airlines'),
    'UAE': ('EMIRATES',     'Emirates'),
    'THY': ('TURKISH',      'Turkish Airlines'),
    'KAL': ('KOREAN AIR',   'Korean Air'),
    'AAR': ('ASIANA',       'Asiana Airlines'),
    'RYR': ('RYANAIR',      'Ryanair'),
    'DLH': ('LUFTHANSA',    'Lufthansa'),
    'RXA': ('REX',          'Regional Express'),
    'VOZ': ('VELOCITY',     'Virgin Australia'),
}

# Try Wikipedia scrape; fall back to hardcoded table
def _scrape_wiki_airlines() -> dict:
    mapping = {}
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        url = f'https://en.wikipedia.org/wiki/List_of_airline_codes_({letter})'
        try:
            tables = pd.read_html(url, header=0)
            for tbl in tables:
                cols     = [str(c).lower() for c in tbl.columns]
                icao_col = next((i for i,c in enumerate(cols) if 'icao' in c), None)
                call_col = next((i for i,c in enumerate(cols) if 'call' in c or 'telephony' in c), None)
                name_col = next((i for i,c in enumerate(cols) if 'airline' in c or 'name' in c), None)
                if icao_col is None or call_col is None:
                    continue
                for _, row in tbl.iterrows():
                    icao3 = str(row.iloc[icao_col]).strip().upper()
                    tel   = str(row.iloc[call_col]).strip().upper()
                    name  = str(row.iloc[name_col]).strip() if name_col else ''
                    if len(icao3) == 3 and icao3.isalpha() and tel not in ('NAN','','-'):
                        mapping[icao3] = (tel, name)
        except Exception:
            continue
    return mapping

try:
    _wiki = _scrape_wiki_airlines()
    AIRLINE_PHONETIC = {**_wiki, **_AIRLINE_FALLBACK} if len(_wiki) > 50 else _AIRLINE_FALLBACK
except Exception:
    AIRLINE_PHONETIC = _AIRLINE_FALLBACK

print(f"Airline DB: {len(AIRLINE_PHONETIC)} entries loaded")


def opensky_fetch_callsigns(icao_code: str) -> list:
    """Fetch live callsigns from OpenSky Network near the given airport."""
    AIRPORT_COORDS = {
        'LSGS':(46.2196,7.3267), 'LKPR':(50.1008,14.2600),
        'LSZB':(46.9141,7.4972), 'LSZH':(47.4582,8.5555),
        'YSSY':(-33.9461,151.177),'LZIB':(48.1702,17.2127),
        'LKTB':(49.1513,16.6944),
    }
    if icao_code not in AIRPORT_COORDS:
        return []
    lat, lon = AIRPORT_COORDS[icao_code]
    try:
        resp = requests.get(
            'https://opensky-network.org/api/states/all',
            params={'lamin':lat-1.5,'lamax':lat+1.5,'lomin':lon-1.5,'lomax':lon+1.5},
            timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return list({s[1].strip() for s in (data.get('states') or []) if s[1]})
    except Exception:
        pass
    return []


def icao_callsign_to_spoken(icao_cs: str) -> str:
    """Convert ICAO callsign (e.g. SWR204K) to spoken form (SWISS TWO ZERO FOUR KILO)."""
    prefix = icao_cs[:3].upper()
    spoken_airline, _ = AIRLINE_PHONETIC.get(prefix, (prefix, prefix))
    spoken_parts = []
    for ch in icao_cs[3:]:
        if ch.isdigit():
            word = [k for k,v in NUM_WORD_TO_DIGIT.items() if v == ch]
            spoken_parts.append(word[0].upper() if word else ch)
        elif ch.isalpha():
            phonetic = [k.upper() for k,v in PHONETIC_ALPHA.items() if v == ch.upper()]
            spoken_parts.append(phonetic[0] if phonetic else ch)
    return f"{spoken_airline} {' '.join(spoken_parts)}"


def build_callsign_db(icao_code: str) -> FAISSVectorDB:
    """Build callsign FAISS index for a given airport (ICAO code)."""
    live = opensky_fetch_callsigns(icao_code)
    docs = [f"{icao_callsign_to_spoken(cs)} | ICAO: {cs}" for cs in live[:50]]
    for icao3, (telephony, _) in AIRLINE_PHONETIC.items():
        for num in range(1, 1000, 100):
            docs.append(f"{icao_callsign_to_spoken(f'{icao3}{num:03d}')} | Airline: {telephony}")
    return FAISSVectorDB(docs or ['callsign database empty'])


def _word_overlap_score(span: str, db_entry: str) -> float:
    """Compute word-level overlap ratio between ASR span and DB entry spoken form."""
    span_words = set(span.upper().split())
    db_spoken  = db_entry.split('|')[0].strip().upper()
    db_words   = set(db_spoken.split())
    if not db_words:
        return 0.0
    return len(span_words & db_words) / len(db_words)


def correct_callsign(asr_text: str, callsign_db: FAISSVectorDB) -> str:
    """
    Callsign correction with Levenshtein-gated LLM reranking.

    Word overlap score gates whether correction is applied:
      - overlap >= 0.40: pass top-k candidates to LLM for selection
      - overlap <  0.40: return original ASR span unchanged
    This prevents hallucination when no matching callsign is found.
    """
    relevant = callsign_db.search(asr_text, k=5)
    overlap_score = max((_word_overlap_score(asr_text, r) for r in relevant), default=0.0)

    prompt = f"""You are an ATC callsign corrector.

ASR output (may have callsign errors): "{asr_text}"

Candidate correct callsigns (spoken form | ICAO):
{chr(10).join(f'  - {r}' for r in relevant)}

Word overlap score between ASR and best candidate: {overlap_score:.2f}

## Rules (STRICT)
1. NEVER add words NOT present in the ASR text.
2. If overlap_score < 0.40, return the ASR callsign span unchanged.
3. Callsign formats:
   - Registration (e.g., HB-OSC): phonetic letters → "HOTEL OSCAR SIERRA CHARLIE"
   - Airline flight (e.g., Swiss 204K): airline telephony + digits individually
     → "SWISS TWO ZERO FOUR KILO"
4. Digit rule: convert EACH digit individually.
   "3043" → "THREE ZERO FOUR THREE"
5. Do NOT correct communication words: ROGER, AFFIRM, WILCO, NEGATIVE.
6. Return ONLY the corrected transcript, no explanation.

Corrected transcript:"""

    corrected = call_claude(prompt)
    if overlap_score < 0.40:
        return asr_text
    return corrected


# ==============================================================
# STEP 3-C. Facility Knowledge Base
# ==============================================================
# Facility names are determined by recording metadata (file_key).
# file_key format: 'LSGS_SION_Ground_Control_121_7MHz_20210504_062720'
# The ICAO code and service type uniquely determine the expected
# facility callsign (e.g., 'Sion Ground'), used as a hard prior.
#
# A pre-check using edit distance filters out utterances that
# do not mention any facility keyword, avoiding unnecessary LLM calls.
# Confidence threshold 0.65: uncertain corrections are discarded.

FACILITY_DB = {
    'LSGS': {'name': 'Sion',    'country': 'Switzerland',
             'services': {'Tower':'Sion Tower','Ground':'Sion Ground','ApronN':'Sion Apron','ApronS':'Sion Apron'}},
    'LKPR': {'name': 'Ruzyne',  'country': 'Czech Republic',
             'services': {'Tower':'Prague Tower','Radar':'Prague Radar','Ground':'Prague Ground'}},
    'LSZB': {'name': 'Bern',    'country': 'Switzerland',
             'services': {'Tower':'Bern Tower','Ground':'Bern Ground'}},
    'LSZH': {'name': 'Zurich',  'country': 'Switzerland',
             'services': {'Tower':'Zurich Tower','Approach':'Zurich Approach','Radar':'Zurich Radar','Ground':'Zurich Ground'}},
    'YSSY': {'name': 'Sydney',  'country': 'Australia',
             'services': {'Tower':'Sydney Tower','Ground':'Sydney Ground','Approach':'Sydney Approach'}},
    'LZIB': {'name': 'Stefanik','country': 'Slovakia',
             'services': {'Tower':'Bratislava Tower','Ground':'Bratislava Ground'}},
    'LKTB': {'name': 'Brno',    'country': 'Czech Republic',
             'services': {'Tower':'Brno Tower','Approach':'Brno Approach'}},
}


def parse_file_key(file_key: str) -> dict:
    """Parse ATCO2 file_key into metadata dict."""
    parts = file_key.split('_')
    return {
        'icao':    parts[0],
        'airport': parts[1],
        'service': parts[2],
        'freq':    f'{parts[3]}.{parts[4].replace("MHz","")}',
    }


def get_expected_facility(file_key: str) -> str:
    """Return ground-truth facility callsign derived from file_key metadata."""
    meta = parse_file_key(file_key)
    info = FACILITY_DB.get(meta['icao'])
    if not info:
        return meta['airport'].capitalize() + ' ' + meta['service']
    return info['services'].get(meta['service'], f"{info['name']} {meta['service']}")


def build_facility_vector_db() -> FAISSVectorDB:
    docs = []
    for icao, info in FACILITY_DB.items():
        for svc, callsign in info['services'].items():
            docs.append(f"{callsign} | {icao} | {info['name']} | {info['country']}")
    return FAISSVectorDB(docs)

facility_db = build_facility_vector_db()


def correct_facility(asr_text: str, file_key: str) -> str:
    """
    Facility name correction using file_key metadata as hard prior.

    Pre-check: if no facility keyword (tower, ground, approach, etc.)
    is found within edit distance 0.40, return the original unchanged.
    This avoids unnecessary LLM calls for pilot-only utterances.
    """
    expected = get_expected_facility(file_key)
    meta     = parse_file_key(file_key)
    relevant = facility_db.search(asr_text, k=3)

    facility_kws = ['tower','ground','approach','radar','control',
                    'departure','arrival','centre','center']
    asr_words = asr_text.lower().split()
    has_facility_mention = any(
        editdistance.eval(w, kw) / len(kw) <= 0.40
        for w in asr_words for kw in facility_kws
    )
    if not has_facility_mention:
        return asr_text

    cand_block = '\n'.join('  - ' + r for r in relevant)
    prompt = (
        'You are an ATC facility name corrector.\n\n'
        'Recording metadata:\n'
        f'  Airport ICAO  : {meta["icao"]}\n'
        f'  Airport name  : {meta["airport"]}\n'
        f'  ATC service   : {meta["service"]}\n'
        f'  Frequency     : {meta["freq"]} MHz\n'
        f'  CORRECT facility callsign: "{expected}"\n\n'
        f'ASR output (may have facility name errors): "{asr_text}"\n\n'
        'Similar facility names from database:\n'
        + cand_block + '\n\n'
        '## Rules\n'
        f'1. Replace only the wrongly-spoken facility name with "{expected}".\n'
        '2. Do NOT change callsigns, instructions, or numbers.\n'
        '3. If uncertain (confidence < 0.65), return the original unchanged.\n'
        '4. Return ONLY the corrected transcript, no explanation.\n\n'
        'Corrected transcript:'
    )
    return call_claude(prompt)


# ==============================================================
# STEP 4. ATCRAGPipeline: Entity-Specific Routing
# ==============================================================
# Routes each ASR hypothesis to the appropriate correction function
# based on the NER-detected error_type field in the dataset.
#
# error_type values:
#   'instruction'  → correct_instruction()
#   'callsign'     → correct_callsign()
#   'facility'     → correct_facility()
#   'phonetic code'→ instruction + callsign (two-pass)
#   'recognition'  → no correction (ASR recognition error, not OOV)
#   None / NaN     → _general_correction() (combines all rules)

class ATCRAGPipeline:

    def __init__(self):
        self._callsign_db_cache: dict[str, FAISSVectorDB] = {}
        print('ATCRAGPipeline initialized')

    def _get_callsign_db(self, icao_code: str) -> FAISSVectorDB:
        if icao_code not in self._callsign_db_cache:
            self._callsign_db_cache[icao_code] = build_callsign_db(icao_code)
        return self._callsign_db_cache[icao_code]

    def correct(self, asr_text: str, file_key: str, error_type: str | None = None) -> str:
        if not isinstance(error_type, str) or (isinstance(error_type, float) and math.isnan(error_type)):
            error_type = None

        if error_type == 'instruction':
            return correct_instruction(asr_text)
        elif error_type == 'callsign':
            db = self._get_callsign_db(parse_file_key(file_key)['icao'])
            return correct_callsign(asr_text, db)
        elif error_type in ('facility name', 'facility'):
            return correct_facility(asr_text, file_key)
        elif error_type == 'phonetic code':
            step1 = correct_instruction(asr_text)
            db    = self._get_callsign_db(parse_file_key(file_key)['icao'])
            return correct_callsign(step1, db)
        elif error_type == 'recognition':
            return asr_text
        else:
            return self._general_correction(asr_text, file_key)

    def _general_correction(self, asr_text: str, file_key: str) -> str:
        """
        Fallback correction combining all rules into a single LLM call.
        Used when error_type is unknown or not specified.
        Includes speaker type detection to avoid converting pilot speech
        into ATC command format.
        """
        expected_facility = get_expected_facility(file_key)
        meta = parse_file_key(file_key)

        prompt = f"""You are an ATC (Air Traffic Control) transcript corrector.

Recording context:
  - Airport: {meta['airport']} ({meta['icao']}), {FACILITY_DB.get(meta['icao'], {}).get('country', 'unknown')}
  - ATC unit: {expected_facility}
  - Frequency: {meta['freq']} MHz

ASR transcript to correct: "{asr_text}"

## Correction rules
1. Fix phonetically confused command words (climb/come, cleared/ready, cross/out of)
2. Convert numeric heading/altitude/FL to spoken form
   EXCEPTION: do NOT convert wind direction, runway numbers, QNH, frequencies
3. Fix callsign using airline telephony name + individual digits
   NEVER add prefix words not spoken in the ASR
4. Replace wrong facility names with: "{expected_facility}"
   Only if facility name is explicitly mentioned in ASR

## Speaker Type Detection
- If ASR contains (we are, we have, not ready, request, negative, wilco, roger,
  affirm, standing by) → PILOT speech. Do NOT convert to ATC commands.

## STRICT constraints
- Do NOT add any word not present in the original ASR output
- Do NOT repeat any phrase already in the transcript
- If unsure whether a word is wrong, keep it as-is

Return ONLY the corrected transcript, no explanation:"""

        return call_claude(prompt)


pipeline = ATCRAGPipeline()


# ==============================================================
# STEP 5. WAV Input Mode (Optional)
# ==============================================================
# Direct audio transcription using Wav2Vec2 UWB-ATCC, an ATC-domain
# fine-tuned ASR model. Used when baseline_result.csv is unavailable.
# Requires soundfile for audio loading (no ffmpeg/torchaudio needed).
#
# CTC log-probability confidence score is extracted per file and
# used as a reference-free quality signal in Step 6.

import glob
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

WAV_MODEL_ID  = 'Jzuluaga/wav2vec2-large-960h-lv60-self-en-atc-uwb-atcc'
print(f'Loading ATC ASR model: {WAV_MODEL_ID}')
_asr_processor = Wav2Vec2Processor.from_pretrained(WAV_MODEL_ID)
_asr_model     = Wav2Vec2ForCTC.from_pretrained(WAV_MODEL_ID)
_asr_model.eval()
print('ATC ASR model loaded')


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    duration = len(audio) / orig_sr
    new_len  = int(duration * target_sr)
    old_idx  = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(old_idx, np.arange(len(audio)), audio)


def transcribe_wav(wav_path: str) -> str:
    """Transcribe a WAV file using Wav2Vec2 UWB-ATCC."""
    audio, sr = sf.read(wav_path, dtype='float32', always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = _resample(audio, sr, 16000)
    inputs = _asr_processor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
    with torch.no_grad():
        logits = _asr_model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return _asr_processor.batch_decode(ids)[0].lower()


def get_asr_confidence(wav_path: str) -> float:
    """
    Extract CTC log-probability confidence from Wav2Vec2.
    Returns value in [0, 1]. Higher = model more certain.
    Used as a reference-free ASR quality signal (no gold annotation needed).
    """
    try:
        audio, sr = sf.read(wav_path, dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            audio = _resample(audio, sr, 16000)
        inputs = _asr_processor(audio, sampling_rate=16000, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = _asr_model(**inputs).logits
        log_probs  = torch.nn.functional.log_softmax(logits, dim=-1)
        confidence = log_probs[0].max(dim=-1).values.mean().exp().item()
        return round(float(confidence), 4)
    except Exception as e:
        print(f'[WARN] confidence error: {e}')
        return 0.5

wav_files = sorted(
    glob.glob(os.path.join(WAV_DIR, '*.wav')) +
    glob.glob(os.path.join(WAV_DIR, '*.WAV'))
)

wav_asr_results  = {}
wav_confidences  = {}

if wav_files:
    print(f'Found {len(wav_files)} WAV file(s)')
    for wav_path in wav_files:
        fname    = os.path.basename(wav_path)
        file_key = fname.replace('.wav', '').replace('.WAV', '')
        try:
            asr_text = transcribe_wav(wav_path)
            conf     = get_asr_confidence(wav_path)
            wav_asr_results[file_key] = asr_text
            wav_confidences[file_key] = conf
            print(f'  {fname}: {asr_text[:60]}  (conf={conf:.3f})')
        except Exception as e:
            print(f'  [ERR] {fname}: {e}')
else:
    print(f'[INFO] No WAV files found in {WAV_DIR}. Using baseline_result.csv.')


# ==============================================================
# STEP 6. Run RAG Correction on Baseline CSV
# ==============================================================
# Iterates over the baseline dataset, applies entity-specific RAG
# correction, and stores results alongside original ASR and gold.
# A random sample of 60 utterances is used due to API token limits.
# random_state=42 ensures reproducibility.

baseline_df = pd.read_csv(BASELINE_CSV)
gold_df     = pd.read_csv(GOLD_CSV)
baseline_df = baseline_df.dropna(subset=['gold', 'asr']).reset_index(drop=True)

print(f'Baseline rows : {len(baseline_df)}')
print(f'Avg WER       : {baseline_df["WER"].mean():.4f}')

# Subsample for evaluation (API token constraint)
sample_df = baseline_df.sample(n=min(60, len(baseline_df)), random_state=42).copy()
print(f'Evaluating on {len(sample_df)} sampled utterances')

rag_outputs = []
for _, row in sample_df.iterrows():
    corrected = pipeline.correct(
        asr_text   = row['asr'],
        file_key   = row['file_key'],
        error_type = row.get('error_type'),
    )
    rag_outputs.append(corrected)
    print(f'  ASR: {row["asr"][:60]}')
    print(f'  RAG: {corrected[:60]}')

sample_df['rag_output'] = rag_outputs
sample_df.to_csv('atc_rag_results.csv', index=False)
print('Results saved to atc_rag_results.csv')


# ==============================================================
# STEP 7. Evaluation: WER and NER F1
# ==============================================================
# WER is computed using jiwer between gold and RAG-corrected output.
# NER F1 (Precision / Recall / F1) is computed per entity type
# (Callsign, Command, Value) by running the NER model on both
# gold and corrected transcripts and comparing entity spans.

from jiwer import wer as compute_wer

def compute_wer_scores(df: pd.DataFrame) -> dict:
    asr_wer = compute_wer(list(df['gold']), list(df['asr']))
    rag_wer = compute_wer(list(df['gold']), list(df['rag_output']))
    return {'ASR WER': round(asr_wer, 4), 'RAG WER': round(rag_wer, 4),
            'Delta WER': round(rag_wer - asr_wer, 4)}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Run NER model and return entity spans grouped by type."""
    entities = {'CALLSIGN': [], 'COMMAND': [], 'VALUE': []}
    for ent in ner_model(text):
        grp = ent['entity_group'].upper()
        if grp in entities and ent['score'] >= 0.4:
            entities[grp].append(ent['word'].lower())
    return entities


def compute_ner_f1(gold_text: str, pred_text: str, entity_type: str) -> dict:
    gold_ents = set(extract_entities(gold_text).get(entity_type, []))
    pred_ents = set(extract_entities(pred_text).get(entity_type, []))
    tp = len(gold_ents & pred_ents)
    fp = len(pred_ents - gold_ents)
    fn = len(gold_ents - pred_ents)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {'P': round(precision, 4), 'R': round(recall, 4), 'F1': round(f1, 4)}


wer_scores = compute_wer_scores(sample_df)
print("\n=== WER Results ===")
for k, v in wer_scores.items():
    print(f"  {k}: {v}")

print("\n=== NER F1 Results ===")
for entity_type in ['CALLSIGN', 'COMMAND', 'VALUE']:
    asr_f1s = [compute_ner_f1(r['gold'], r['asr'],        entity_type) for _, r in sample_df.iterrows()]
    rag_f1s = [compute_ner_f1(r['gold'], r['rag_output'], entity_type) for _, r in sample_df.iterrows()]
    asr_avg = {k: round(sum(d[k] for d in asr_f1s) / len(asr_f1s), 4) for k in ['P','R','F1']}
    rag_avg = {k: round(sum(d[k] for d in rag_f1s) / len(rag_f1s), 4) for k in ['P','R','F1']}
    print(f"  {entity_type}")
    print(f"    ASR: P={asr_avg['P']} R={asr_avg['R']} F1={asr_avg['F1']}")
    print(f"    RAG: P={rag_avg['P']} R={rag_avg['R']} F1={rag_avg['F1']}")
    print(f"    ΔF1: {round(rag_avg['F1'] - asr_avg['F1'], 4)}")

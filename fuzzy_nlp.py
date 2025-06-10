from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rapidfuzz.fuzz import token_set_ratio
import spacy
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker
from unidecode import unidecode

# Load NLP resources
nlp = spacy.load("en_core_web_sm")
spell = SpellChecker(language="en")

# Sample documents
documents = [
    "The colour of the knife was red.",
    "He used a dagger in the attack.",
    "The tumor was malignant.",
    "The follow-up appointment was missed.",
    "She witnessed the murder.",
]

# NLP helpers
def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace("_", " "))
    return synonyms

def normalize_text(text):
    text = unidecode(text.lower().strip())
    text = text.replace("-", " ")
    doc = nlp(text)

    tokens = []
    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue
        word = token.lemma_
        corrected = spell.correction(word)
        tokens.append(corrected)
        tokens.extend(get_synonyms(corrected))
    return set(tokens)

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def find_matches(query: str, j_thresh=0.3, fuzz_thresh=70):
    query_tokens = normalize_text(query)
    results = []

    for doc in documents:
        doc_tokens = normalize_text(doc)
        jac = jaccard_similarity(query_tokens, doc_tokens)
        fuzz_score = token_set_ratio(query, doc)

        if jac >= j_thresh or fuzz_score >= fuzz_thresh:
            results.append({
                "text": doc,
                "jaccard": round(jac, 2),
                "fuzz": fuzz_score
            })
    return sorted(results, key=lambda x: (x["jaccard"], x["fuzz"]), reverse=True)

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for POST
class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(request: SearchRequest):
    return find_matches(request.query)

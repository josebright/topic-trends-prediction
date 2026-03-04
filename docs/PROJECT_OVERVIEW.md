# Topic Trends Prediction — Project Overview

A Flask-based API that predicts emerging research trends by analyzing publication data from the OpenAIRE database using natural language processing and machine learning.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Technology Stack](#technology-stack)
3. [Architecture](#architecture)
4. [Core Components](#core-components)
5. [Algorithm Flow](#algorithm-flow)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Project Structure](#project-structure)
9. [Development Setup](#development-setup)
10. [Contributing](#contributing)

---

## Introduction

This project identifies emerging research topics within specific disciplines by:

- **Fetching** publication data from the OpenAIRE database
- **Extracting** and ranking keywords from publication titles using NLP
- **Weighting** keywords by recency, influence, and funding correlation
- **Generating** publication titles and abstracts using AI

It helps researchers and funding bodies discover trending topics, potential collaborators, and funding tendencies.

---

## Technology Stack

| Category | Technology |
|----------|------------|
| Web Framework | Flask, Flask-CORS |
| NLP | spaCy (`en_core_web_sm`), langid |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Generative AI | OpenAI GPT-3.5-turbo-instruct |
| Data Processing | Pandas, NumPy |
| External API | [OpenAIRE Research Publications API](https://api.openaire.eu/) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Flask Application                              │
├─────────────────────────────────────────────────────────────────────────┤
│  POST /generate-titles          │  POST /generate-abstract               │
└──────────────┬──────────────────┴──────────────────┬────────────────────┘
               │                                      │
               ▼                                      ▼
┌──────────────────────────────┐      ┌──────────────────────────────────┐
│   Data Fetching Layer        │      │   Abstract Generation Layer       │
│   • OpenAIRE API             │      │   • GPT-3.5 prompt                │
│   • Retry + backoff          │      │   • Title + FOS context           │
└──────────────┬───────────────┘      └──────────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Text Processing Layer     │
│   • Preprocess titles       │
│   • Extract keywords (spaCy)│
│   • Language filter (langid)│
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Keyword Analysis Layer    │
│   • Temporal decay weights  │
│   • Semantic similarity     │
│   • Aggregate & rank        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│   Title Generation Layer    │
│   • POS-tag keywords        │
│   • Top 20% selection       │
│   • GPT-3.5 generation      │
└──────────────────────────────┘
```

---

## Core Components

### 1. Data Fetching Layer

**Functions:** `fetch_publications`, `extract_publications`, `handle_publications`

- Queries OpenAIRE API with field of study (`fos`), publication type (`instancetype`), and optional filters
- Date range: last 10 years from current date
- Retry logic: 3 attempts with exponential backoff (0.3 × 2^attempt seconds)
- Extracts: DOI, title, authors, date of acceptance, access rights, full-text links, measures (influence), contributors, funding details

### 2. Text Processing Layer

**Functions:** `preprocess_text`, `extract_keywords`, `extract_keywords_from_text`

- **Preprocessing:** Lowercase, remove punctuation, strip HTML tags, normalize whitespace
- **Keyword extraction:** spaCy noun chunks + POS tags (NOUN, ADJ, VERB)
- **Language filter:** English only (via `langid`)

### 3. Keyword Analysis Layer

**Functions:** `calculate_year_weight`, `aggregate_keywords`, `global_sorted_keywords`, `pos_tag_keywords`, `get_top_20_percent_keywords`

- **Temporal weighting:** `weight = e^(-0.1 × (current_year - publication_year))`
- **Semantic similarity:** Cosine similarity between keyword embeddings and field-of-study embedding
- **Composite score:** 40% normalized count + 40% normalized influence + 20% normalized funded count

### 4. Title Generation Layer

**Functions:** `generate_dynamic_template`, `generate_publication_titles`, `log_matching_info`

- Groups keywords by POS (NOUN, ADJ, VERB)
- Selects top 20% per category
- Uses GPT-3.5 to generate publication titles from selected keywords
- Matches generated titles to source publications and authors

### 5. Abstract Generation Layer

**Function:** `generate_abstract` (route handler)

- Builds a prompt from title, field of study, and publication type
- Uses GPT-3.5 to generate a concise abstract

---

## Algorithm Flow

```
1. Receive request (fos, publication_type, concept, page)
2. Fetch publications from OpenAIRE API
3. Extract metadata for each publication
4. Filter non-empty titles
5. For each publication:
   a. Filter English titles (langid)
   b. Extract keywords (spaCy)
   c. Apply year-based decay weight
   d. Compute semantic similarity to FOS
   e. Store weighted keywords
6. Aggregate keywords across all publications
7. Compute final scores (count + influence + funding)
8. Rank and sort keywords globally
9. Select top 20% keywords by POS
10. Generate titles via GPT-3.5
11. Match generated titles to source publications/authors
12. Return results
```

---

## API Reference

### POST `/generate-titles`

Generates publication titles based on field of study and optional keywords.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `fos` | string | Yes | Field of study (e.g., "Computer Science") |
| `publication_type` | string | Yes | Publication type (e.g., "article", "conference") |
| `concept` | string | No | Keywords to filter publications |
| `page` | number | No | Page number for pagination |

**Response:**

```json
[
  {
    "generated_title": "Example Title",
    "matching_publication_titles": ["Source Title 1", "Source Title 2"],
    "matching_authors": ["Author A", "Author B"]
  }
]
```

**Error:** `{"error": "No valid publications found."}`

---

### POST `/generate-abstract`

Generates an abstract for a given title.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `generated_title` | string | Yes | The title to generate an abstract for |
| `fos` | string | Yes | Field of study |
| `publication_type` | string | Yes | Publication type |

**Response:**

```json
{
  "abstract": "Generated abstract text."
}
```

**Error:** `{"error": "Failed to generate abstract"}` with HTTP 500

---

## Configuration

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-3.5 |

Set before running:

```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Project Structure

```
topic-trends-prediction/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md              # Quick start guide
├── docs/
│   └── PROJECT_OVERVIEW.md   # Full technical documentation (this file)
└── .gitignore
```

---

## Development Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd topic-trends-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model (required)
python -m spacy download en_core_web_sm

# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"
```

### Running the Application

```bash
python app.py
```

The API will be available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

### Example Request

```bash
curl -X POST http://127.0.0.1:5000/generate-titles \
  -H "Content-Type: application/json" \
  -d '{"fos": "Computer Science", "publication_type": "article", "concept": "machine learning", "page": 1}'
```

---

## Contributing

Contributions are welcome. Please fork the repository and submit a pull request for improvements or bug fixes.

---

## Contact

For questions or suggestions: [josebright29@gmail.com](mailto:josebright29@gmail.com)

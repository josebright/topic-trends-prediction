# Topic Trends Prediction

Predict emerging research topics by analyzing publication data from the OpenAIRE database using NLP and AI.

**[→ Full Project Overview & Documentation](docs/PROJECT_OVERVIEW.md)**

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set your OpenAI API key

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Run the application

```bash
python app.py
```

API available at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate-titles` | POST | Generate publication titles from field of study and keywords |
| `/generate-abstract` | POST | Generate an abstract for a given title |

---

## Example

```bash
curl -X POST http://127.0.0.1:5000/generate-titles \
  -H "Content-Type: application/json" \
  -d '{"fos": "Computer Science", "publication_type": "article", "concept": "machine learning", "page": 1}'
```

---

## Documentation

For architecture details, algorithm flow, API reference, and configuration, see:

**[docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)**

---

## Contact

[josebright29@gmail.com](mailto:josebright29@gmail.com)

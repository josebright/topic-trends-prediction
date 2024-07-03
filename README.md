# Predicting Emerging Research Topics Using Graph-Based Data and Semantic Analysis

## Introduction
This project aims to predict emerging research trends within specific disciplines by analyzing publication data from the OpenAIRE database. By leveraging graph-based data and semantic analysis, this project identifies potential collaborators, assesses author engagement in scientific publishing, and predicts the tendency of funding in the predicted topics.

## Features
- Fetches publication data from the OpenAIRE database.
- Extracts and preprocesses keywords from publication titles.
- Assigns weights to keywords based on the publication year.
- Ranks keywords considering factors such as funding, repeated count, and citation count.
- Generates publication titles using the top-ranked keywords.
- Provides a dynamic template for generating abstracts based on generated titles and research fields.

## Installation
To run this project, you need to have Python installed. You can install the required dependencies using `pip`.

```bash
pip install -r requirements.txt
```

## Usage
### Running the Application
Start the Flask application by executing the following command:

```bash
python app.py
```

The application will be accessible at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).


### Endpoints

#### Generate Titles
- **Endpoint**: `/generate-titles`
- **Method**: `POST`
- **Description**: Generates a list of publication titles based on specified fields of study and publication type.
- **Request Body**:
```json
{
  "fos": "Field of Study",
  "publication_type": "Publication Type",
  "concept": "Keywords",
  "page": "Page Number"
}
```
- **Response**:
```json
[
  {
    "generated_title": "Generated Title 1",
    "matching_publication_titles": ["Publication Title 1", "Publication Title 2"],
    "matching_authors": ["Author 1", "Author 2"]
  },
  ...
]
```


#### Generate Abstract
- **Endpoint**: `/generate-abstract`
- **Method**: `POST`
- **Description**: Generates an abstract for a given generated title.
- **Request Body**:
```json
{
  "generated_title": "Generated Title",
  "fos": "Field of Study",
  "publication_type": "Publication Type"
}
```
- **Response**:
```json
{
  "abstract": "Generated abstract text."
}
```

## Project Structure
- `app.py`: Main Flask application file.
- `requirements.txt`: List of required Python packages.
- `README.md`: Project documentation.

## Implementation Details
- **Data Fetching**: The application fetches publication data from the OpenAIRE API, filtering by the field of study, publication type, and date range (last 10 years).
- **Data Extraction**: Extracts relevant information such as DOI, title, authors, date of acceptance, access rights, full-text links, measures, contributors, funding details, etc.
- **Keyword Extraction**: Preprocesses the publication titles to extract keywords using Spacy NLP.
- **Keyword Weight Calculation**: Assigns weights to keywords based on the publication year using an exponential decay formula.
- **Keyword Ranking**: Aggregates and ranks keywords considering their frequency, influence, and funding details.
- **Title Generation**: Uses OpenAI's GPT-3.5 model to generate publication titles based on the top-ranked keywords.
- **Abstract Generation**: Generates an abstract for a given title using OpenAI's GPT-3.5 model.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

<!-- ## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details. -->

## Contact
For any questions or suggestions, please contact [josebright29@gmail.com](mailto:josebright29@gmail.com).

---

This project leverages state-of-the-art natural language processing and machine learning techniques to predict emerging research topics, providing valuable insights for researchers and funding bodies.


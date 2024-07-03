import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import spacy
import string
import langid
import time
import requests
import logging
import openai
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')

openai.api_key = os.getenv('OPENAI_API_KEY')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_publications(fos, instancetype, extra_params=None, max_retries=3, backoff_factor=0.3):
    base_url = "https://api.openaire.eu/search/publications"
    current_date = datetime.now()
    ten_years_ago_date = current_date.replace(year=current_date.year - 10)
    headers = {"Content-Type": "application/json"}
    params = {
        'fos': fos,
        'instancetype': instancetype,
        'fromDateAccepted': ten_years_ago_date.strftime("%Y-%m-%d"),
        'toDateAccepted': current_date.strftime("%Y-%m-%d"),
        'sortBy': 'dateofcollection,descending',
        'format': 'json',
    }

    if extra_params:
        params.update(extra_params)

    for attempt in range(max_retries):
        try:
            response = requests.get(base_url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error("Max retries reached. Failing request.")
                return None

def extract_publications(data):
    extracted_data = []
    results = data.get('response', {}).get('results', {}).get('result', [])

    for result in results:
        publication = {
            'doi': None,
            'title': '',
            'authors': [],
            'date_of_acceptance': None,
            'access_rights': None,
            'full_text_links': [],
            'measures': {},
            'contributors': [],
            'is_publicly_funded': None,
            'is_green': None,
            'open_access_color': None,
            'funding_details': []
        }

        # Extract identifiers
        pids = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('pid', [])
        if isinstance(pids, list):
          publication['doi'] = next((pid.get('$') for pid in pids
            if isinstance(pid, dict) and pid.get('@classname') == 'Digital Object Identifier'), None)
        elif isinstance(pids, dict):
          if pids.get('@classname') == 'Digital Object Identifier':
            publication['doi'] = pids.get('$', None)

        # Extract title
        titles = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('title', [])
        if isinstance(titles, list):
          publication['title'] = next((title.get('$') for title in titles
            if isinstance(title, dict) and title.get('@classname') == 'main title'), '')
          if not publication['title']:
            publication['title'] = next((title.get('$') for title in titles
              if isinstance(title, dict) and title.get('@classname') == 'alternative title'), '')
          if not publication['title']:
            publication['title'] = next((title.get('$') for title in titles
              if isinstance(title, dict) and title.get('@classname') == 'subtitle'), '')
        elif isinstance(titles, dict):
          if titles.get('@classname') == 'main title':
            publication['title'] = titles.get('$', '')
          elif titles.get('@classname') == 'alternative title':
            publication['title'] = titles.get('$', '')
          elif titles.get('@classname') == 'subtitle':
            publication['title'] = titles.get('$', '')

        # Extract authors
        creators = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('creator', [])
        if isinstance(creators, list):
          publication['authors'] = [{'name': creator.get('$'), 'rank': creator.get('@rank')} for creator in creators if isinstance(creator, dict)]
        elif isinstance(creators, dict):
            publication['authors'] = [{'name': creators.get('$'), 'rank': creators.get('@rank')}]

        # Extract date of acceptance
        date_of_acceptance = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('dateofacceptance', {})
        if isinstance(date_of_acceptance, dict):
            publication['date_of_acceptance'] = date_of_acceptance.get('$')

        # Extract access rights
        access_rights = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('bestaccessright', {})
        if isinstance(access_rights, dict):
            publication['access_rights'] = access_rights.get('@classname')

        # Extract full text links
        fulltexts = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('children', {}).get('instance', [])
        if isinstance(fulltexts, list):
          publication['full_text_links'] = [fulltext.get('webresource', {}).get('url', {}).get('$') for fulltext in fulltexts if isinstance(fulltext, dict)]
        elif isinstance(fulltexts, dict):
          publication['full_text_links'] = [fulltexts.get('webresource', {}).get('url', {}).get('$')]

        # Extract measures like influence and popularity
        measures = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('measure', [])
        for measure in measures:
            if isinstance(measure, dict):
                publication['measures'][measure.get('@id')] = measure.get('@score')

        # Extract contributors
        contributors = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('contributor', [])
        if isinstance(contributors, list):
          publication['contributors'] = [contributor.get('$') for contributor in contributors if isinstance(contributor, dict)]
        elif isinstance(contributors, dict):
          publication['contributors'] = [contributors.get('$')]

        # Extract publicly funded information
        is_publicly_funded = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('publiclyfunded', {})
        if isinstance(is_publicly_funded, dict):
            publication['is_publicly_funded'] = bool(is_publicly_funded.get('$', False))

        # Extract isgreen information
        is_green = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('isgreen', {})
        if isinstance(is_green, dict):
            publication['is_green'] = bool(is_green.get('$', False))

        # Extract openaccesscolor information
        open_access_colors = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('openaccesscolor', {})
        if isinstance(open_access_colors, list):
          publication['open_access_color'] = [open_access_color.get('$') for open_access_color in open_access_colors if isinstance(open_access_color, dict)]
        elif isinstance(open_access_colors, dict):
            publication['open_access_color'] = open_access_colors.get('$')

        # Extract funding details
        contexts = result.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('context', [])
        if isinstance(contexts, list):
          for context in contexts:
            if isinstance(context, dict):
              if context.get('@type') == 'funding':
                publication['funding_details'].append({'id': context.get('@id'), 'funder_name': context.get('@label')})
        elif isinstance(contexts, dict):
          if contexts.get('@type') == 'funding':
            publication['funding_details'] = [{'id': contexts.get('@id'), 'funder_name': contexts.get('@label')}]


        extracted_data.append(publication)

    return extracted_data

def handle_publications(fos, instancetype, extra_params=None):
    publications_data = fetch_publications(fos, instancetype, extra_params)

    if publications_data and publications_data.get('response', {}).get('results'):
        return extract_publications(publications_data)
    else:
        logging.info("No data returned from the API.")
        return []

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_year_weight(year):
    current_year = datetime.now().year
    decay_factor = 0.1
    return np.round(np.exp(-decay_factor * (current_year - year)), 4)

def extract_keywords_from_text(text):
    preprocessed_text = preprocess_text(text)
    doc = nlp(preprocessed_text)
    keywords = set()

    for chunk in doc.noun_chunks:
        filtered_tokens = [token.text for token in chunk if not token.is_stop and not token.is_digit and not token.text.isnumeric()]
        if filtered_tokens:
            keywords.add(' '.join(filtered_tokens))

    keywords.update(token.text for token in doc if not token.is_stop and not token.is_digit and token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_punct)

    return list(keywords)

def extract_keywords(row, fos):
    date = pd.to_datetime(row['date_of_acceptance'])
    year = date.year
    weight = calculate_year_weight(year)
    lang, _ = langid.classify(row['title'])

    if lang != 'en':
        return {}

    preprocessed_text = preprocess_text(row['title'])
    doc = nlp(preprocessed_text)
    keywords = set()

    for chunk in doc.noun_chunks:
        filtered_tokens = [token.text for token in chunk if not token.is_stop and not token.is_digit and not token.text.isnumeric()]
        if filtered_tokens:
            keyword = ' '.join(filtered_tokens)
            keywords.add(keyword)
            if len(filtered_tokens) > 2:
                keywords.update(' '.join(filtered_tokens[start:start + size])
                for size in range(2, len(filtered_tokens)) for start in range(0, len(filtered_tokens) - size + 1))

    keywords.update(token.text for token in doc if not token.is_stop and not token.is_digit
                    and token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_punct)

    fos_embedding = model.encode(fos, convert_to_tensor=True)
    relevant_keywords = {}

    for keyword in keywords:
        keyword_embedding = model.encode(keyword, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(fos_embedding, keyword_embedding).item()

        adjusted_weight = weight * similarity
        relevant_keywords[keyword] = adjusted_weight
    return relevant_keywords

def aggregate_keywords(df):
    aggregated_keywords = defaultdict(lambda: {'count': 0, 'influence_alt': 0, 'funded_count': 0})

    for _, row in df.iterrows():
        influence_alt = float(row['measures'].get('influence_alt', 0))
        for keyword, weight in row['keywords'].items():
            aggregated_keywords[keyword]['count'] += 1
            aggregated_keywords[keyword]['influence_alt'] += influence_alt * weight
            if row['is_funded']:
                aggregated_keywords[keyword]['funded_count'] += 1

    total_count = df.shape[0]
    total_influence = df['measures'].apply(lambda x: float(x.get('influence_alt', 0))).sum()
    total_funded = df['is_funded'].sum()

    for keyword in aggregated_keywords:
        kw_data = aggregated_keywords[keyword]
        kw_data['normalized_count'] = kw_data['count'] / total_count
        kw_data['normalized_influence_alt'] = kw_data['influence_alt'] / total_influence if total_influence != 0 else 0
        kw_data['normalized_funded_count'] = kw_data['funded_count'] / total_funded if total_funded != 0 else 0
        kw_data['final_score'] = (
            0.4 * kw_data['normalized_count'] +
            0.4 * kw_data['normalized_influence_alt'] +
            0.2 * kw_data['normalized_funded_count']
        )

    return {k: rank + 1 for rank, (k, v) in enumerate(sorted(aggregated_keywords.items(), key=lambda x: x[1]['final_score'], reverse=True))}

def collect_all_keywords(df):
    all_keywords = {}
    for index, row in df.iterrows():
        for keyword, rank in row['keywords_rank'].items():
            if keyword not in all_keywords:
                all_keywords[keyword] = rank
    return all_keywords

def global_sorted_keywords(all_keywords):
    sorted_keywords = sorted(all_keywords.items(), key=lambda item: item[1])
    sorted_keywords_dict = {keyword: rank for keyword, rank in sorted_keywords}
    return sorted_keywords_dict

def get_top_20_percent_keywords(pos_tags):
    top_keywords = {}
    for pos, keywords in pos_tags.items():
        top_n = max(1, len(keywords) // 5)
        top_keywords[pos] = keywords[:top_n]
    return top_keywords

def pos_tag_keywords(keywords_dict):
    pos_tags = {'NOUN': [], 'ADJ': [], 'VERB': []}

    for keyword, weight in keywords_dict.items():
        doc = nlp(keyword)
        for token in doc:
            if token.pos_ in pos_tags:
                pos_tags[token.pos_].append(token.text)

    return pos_tags

def generate_dynamic_template(pos_tags, publication_type):
    prompt = f"Generate a list of publication titles with {publication_type} publication type using the following parts of speech tags:\n"
    for pos, words in pos_tags.items():
        prompt += f"{pos}: {', '.join(words)}\n"

    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200,
        n=5,
        stop=["\n\n"],
        temperature=0.7,
    )

    titles = response.choices[0].text.strip().split('\n')
    return titles

def generate_publication_titles(keywords_dict, publication_type):
    pos_tags = pos_tag_keywords(keywords_dict)
    top_pos_tags = get_top_20_percent_keywords(pos_tags)

    titles = generate_dynamic_template(top_pos_tags, publication_type)

    return titles

def log_matching_info(df, generated_titles):
    match_info = defaultdict(lambda: {"publication_titles": [], "authors": []})

    for title in generated_titles:
        title_keywords = extract_keywords_from_text(title)

        for _, row in df.iterrows():
            matching_keywords = set(title_keywords) & set(row['keywords_rank'].keys())

            if matching_keywords:
                match_info[title]["publication_titles"].append(row['title'])
                match_info[title]["authors"].extend(author['name'] for author in row['authors'])

    return match_info

@app.route('/generate-titles', methods=['POST'])
def generate_titles():
    data = request.json
    print(data)
    fos = data.get('fos')
    publication_type = data.get('publication_type')
    keywords=data.get('concept')
    page = data.get('page')

    publications_list = handle_publications(
        fos,
        publication_type,
        extra_params={
            'page': page,
            'size': 20,
            'keywords': keywords
        }
    )

    if publications_list:
        df = pd.DataFrame(publications_list)
        df = df[df['title'].notnull() & (df['title'] != '')]

        if not df.empty:
            df['is_funded'] = df['funding_details'].apply(lambda x: len(x) > 0)
            df['keywords'] = df.apply(lambda row: extract_keywords(row, fos), axis=1)
            ranked_keywords = aggregate_keywords(df)
            df['keywords_rank'] = df['keywords'].apply(lambda x: {k: ranked_keywords[k] for k in x})

            global_sorted_keywords_dict = global_sorted_keywords(collect_all_keywords(df))
            generated_titles = generate_publication_titles(global_sorted_keywords_dict, publication_type)
            match_info = log_matching_info(df, generated_titles)

            result = []
            for gen_title, info in match_info.items():
                result.append({
                    "generated_title": gen_title,
                    "matching_publication_titles": info["publication_titles"],
                    "matching_authors": list(set(info["authors"]))
                })

            return jsonify(result)

    return jsonify({"error": "No valid publications found."})

@app.route('/generate-abstract', methods=['POST'])
def generate_abstract():
    data = request.json
    generated_title = data.get('generated_title')
    fos = data.get('fos')
    publication_type = data.get('publication_type')

    prompt = (
        f"Generate an abstract for a {publication_type} titled '{generated_title}' in the field of {fos}. "
        "The abstract should be concise and informative, covering the main contributions and findings of the paper."
    )

    try:
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=250,
            temperature=0.7,
            n=1,
            stop=None
        )
        abstract = response.choices[0].text.strip()
        print(response)
        return jsonify({"abstract": abstract})
    except Exception as e:
        logging.error(f"Error generating abstract: {e}")
        return jsonify({"error": "Failed to generate abstract"}), 500

if __name__ == "__main__":
    app.run(debug=True)


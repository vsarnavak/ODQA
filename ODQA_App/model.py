import nltk
from indicnlp import common

# Set the path to the Indic NLP Resources directory
INDIC_NLP_RESOURCES = "/path_to_resources"

common.set_resources_path(INDIC_NLP_RESOURCES)


import indicnlp
from indicnlp.tokenize import indic_tokenize
import requests
import re
import torch
from indicnlp.tokenize import indic_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

# List of common Tamil stop words
tamil_stop_words = [
    "மற்றும்", "இது", "அது", "இவை", "அவை", "என்று", "ஒரு", "இல்லை",
    "உள்ள", "என்ற", "என்ன", "இருந்து", "யார்", "எப்போது", "எங்கே",
    "ஏன்", "எது", "எப்படி", "எத்தனை", "கொண்டு", "போல", "என"
]

# Function to remove prefixes
import re

# Function to remove suffixes
def remove_suffixes(word):
    suffixes = ["ன்", "ம்", "கள்", "து", "ல்", "டன்", "னின்", "அ", "ஆ", "உம்", "இன்", "கின்", "அவ்"]
    for suffix in suffixes:
        if word.endswith(suffix):
            return word
    return word

def get_wikipedia_infobox(title, language='ta'):
    url = f"https://{language}.wikipedia.org/w/api.php"

    params = {
        'action': 'query',
        'titles': title,
        'prop': 'revisions',
        'rvprop': 'content',
        'format': 'json',
        'rvsection': 0
    }

    response = requests.get(url, params=params)
    data = response.json()

    # Parse the infobox from the returned data
    pages = data['query']['pages']
    infobox = None

    for page_id, page in pages.items():
        if 'revisions' in page:
            content = page['revisions'][0]['*']
            # Infobox usually starts with '{{Infobox'
            infobox_start = content.find('{{Infobox')
            if infobox_start != -1:
                infobox_end = content.find('}}', infobox_start)
                infobox = content[infobox_start:infobox_end+2]
    return infobox

# Function to fix endings after suffix removal
def fix_endings(word):
    # Specific case for "டின்" to "டு"
    if word.endswith("ட்டின்"):
        word = word.replace('ட்டின்', "டு")
    elif word.endswith("வின்"):
        word = word.replace("வின்","")
    # Handling Vallinam consonants
    if word.endswith("ற்"):
        word = word[:-1] + "ற"
    if word.endswith("வ்"):
        word = word[:-1] + "வ"
    return word

# Complete Tamil stemmer function
def tamil_stemmer(word):
    word = remove_suffixes(word)
    word = fix_endings(word)
    return word

# Function to process tokens with stemming
def process_tokens(words_list):
    tamil_regex = r'[^\u0B80-\u0BFF ]+'  # Tamil Unicode range
    filtered_words = []

    for word in words_list:
        cleaned_word = re.sub(tamil_regex, '', word).strip()
        stemmed_word = tamil_stemmer(cleaned_word)
        if stemmed_word and stemmed_word not in tamil_stop_words:
            filtered_words.append(stemmed_word)

    return filtered_words

# Tokenize and stem the question
def tokenize_question(question):

    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("ta")
    normalized_text = normalizer.normalize(question)
    tokens = list(indic_tokenize.trivial_tokenize(normalized_text, "ta"))
    tokens = [i for i in tokens if i not in tamil_stop_words]
    print(tokens)
    processed_tokens_list = process_tokens(tokens)
    print(processed_tokens_list)
    return processed_tokens_list

# Wikipedia search function
def wikipedia_search(query, language='ta'):
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'utf8': 1,
    }
    response = requests.get(url, params=params)
    search_results = response.json()

    return search_results['query']['search'] if 'query' in search_results else []

# Wikipedia article content extraction function
def get_wikipedia_article(title, language='ta'):
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'prop': 'extracts',
        'exintro': True,
        'explaintext': True,
        'titles': title,
        'format': 'json',
        'utf8': 1,
    }
    response = requests.get(url, params=params)
    pages = response.json()['query']['pages']

    for page_id, page in pages.items():
        if 'extract' in page:
            return page['extract']

    return ""

# Load XLM-RoBERTa model and tokenizer

# Function to answer the question using Wikipedia search and BERT ranking
def answer_question(question, language='ta'):
    tokens = tokenize_question(question)
    combined_content = ""

    # Perform search using the stemmed tokens to retain context
    search_query = ' '.join(tokens)

    search_results = wikipedia_search(search_query, language)
    print(search_results)
    # Loop through search results and extract content
    # infobox=''
    # for i in tokens:

    #     infobox += str(get_wikipedia_infobox(i, language='ta'))
    #     print(infobox)
    #     combined_content += infobox +'\n'

    for result in search_results:
        title = result['title']
        content = get_wikipedia_article(title, language)
        combined_content += content + "\n"
    # Split the combined content into sentences
    sentences = combined_content.split(". ")
    context = ''
    for i in sentences:
       context+=i+'\n'

    # If any token from the question matches in combined content
    if len(context) == 0:
      return "Sorry:( No match"
    return context


from transformers import BertForQuestionAnswering, BertTokenizer

# Load the model and tokenizer from the unzipped directory
model = BertForQuestionAnswering.from_pretrained('./trained_model')
tokenizer = BertTokenizer.from_pretrained('./trained_model')
def finalfun(test_question):
    # test_question = input()
    print("hi")
    test_context = answer_question(test_question)
    print(test_context)
    inputs = tokenizer(test_question, test_context, truncation=True, padding=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits)
    answer_tokens = inputs['input_ids'][0][start_index:end_index + 50]
    answer = tokenizer.decode(answer_tokens)
    # print(answer)
    if answer.startswith("<s>"):
        answer = answer[3:]
    if len(answer)!=0:
        print("Predicted Answer:", answer)
    else:
        print("Sorry:(... Unable to fetch appropriate results!")
    return [test_context, answer]

print("Write Question")
qn = input()
finalfun(qn)
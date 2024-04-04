"""
Helper Functions

Functions used for various privatization methods, including:
* Reading / Writing Data
* Data Processing
* Miscellaneous
* Documents > Sentences > Documents
"""

import json
import os
import random

import nltk
import numpy as np
import torch
from absl import logging

# Reading/Writing Data

def get_text(json_line):
    if 'fullText' in json_line: return json_line['fullText']
    else:
        return json_line['syms'][0]

def get_data(data_path, run_id, full_json=False):
    """ Read in JSONL of texts into flat list of documents """
    with open(data_path , 'r') as f:
        if full_json:
            data = [json.loads(line) for line in f]
        else:
            data = [get_text(json.loads(line)) for line in f]
        if "debug" in run_id:
            logging.info("DEBUG MODE: includes first 5000 documents only.")
            data = data[:5000]
    return data

def save_privatized_documents(input_path, privatized_documents, output_fname, run_id):
    """ Save JSONL with back_translated documents """
    data = get_data(input_path, run_id, full_json=True)
    
    with open(output_fname, 'w') as outfile:
        for i in range(len(data)):
            data[i]['fullText'] = privatized_documents[i][0]
            json.dump(data[i], outfile)
            outfile.write('\n')

# Data Processing 

def create_batches(docs, batch_size):
    """ Batch documents or sentences into set batch size"""
    batches = [docs[idx:idx+batch_size] for idx in range(0, len(docs), batch_size)]
    return batches

def truncate_sentence(sentence, max_length):
    """ Truncate sentences longer than max length to avoid mBART error"""
    if len(sentence) > max_length:
        sentence = sentence[:max_length]
    return sentence




def tokenize(text, tokenizer, max_length, eval=False):
    """ Tokenize text """
    text = text if isinstance(text, list) else [text]
    params = dict(truncation=True, max_length=max_length, return_tensors="pt")
    if eval is True:
        params['padding'] = 'max_length'
    tokenized_data = tokenizer(
        text, **params
    )
    return tokenized_data["input_ids"], tokenized_data["attention_mask"]

# Miscellaneous 

def set_seeds(random_seed):
    """ Set random seeds for reproducibility """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

def set_device(gpus):
    """ Set device type """
    device = "cuda" if gpus >= 1 and torch.cuda.is_available() else "cpu"
    logging.info(f'Using {device}')
    return device

# Documents > Sentences > Documents 

def orig_docs2sents(documents, truncation_length, flatten=False):
    """ Split documents to sentences (from EasyNMT).
    Args:
        documents (list[str]): documents to break apart into sentences
        truncation_length (int): length to truncate sentence. If -1, no truncation performed.
        flatten (bool, optional): whether to return flattened list of sentences"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    split_sentences = [[sent.strip() for para in doc.splitlines()
                                        for sent in nltk.sent_tokenize(para.strip()) if len(sent) > 0]
                                        for doc in documents] 
    sents_per_doc_cumulative = np.cumsum([len(doc)for doc in split_sentences])
    
    if truncation_length > 0:
        split_sentences_truncated = [[truncate_sentence(sent, truncation_length) for sent in doc] for doc in split_sentences]
        num_truncated = sum([sum([len(sent) != len(sent_trunc) for sent, sent_trunc in zip(doc, doc_trunc)]) for doc, doc_trunc in zip(split_sentences, split_sentences_truncated)])
        proportion_truncated = num_truncated/len([x for y in split_sentences_truncated for x in y]) * 100
        logging.info(f'{round(proportion_truncated,0)}% of sentences truncated at {truncation_length} characters')
    
    if flatten:
        split_sentences = [item for list in split_sentences for item in list] 
    return sents_per_doc_cumulative, split_sentences

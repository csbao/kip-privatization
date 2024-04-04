import json
import os
import random
import time
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from absl import app, logging, flags
from einops import rearrange
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5Tokenizer, pipeline
from generator import Generator
from helper_functions import tokenize, create_batches, orig_docs2sents
from sentence_transformers import SentenceTransformer

FLAGS = flags.FLAGS


flags.DEFINE_string("input_data_path", None, "File to transfer", required=True)
flags.DEFINE_string("output_path", None, "Directory where the modified query texts will be stored", required=True)
flags.DEFINE_string("model_path", 'models/', "Directory where the models are stored")
flags.DEFINE_integer("token_max_length", 200, "Number of tokens to use per text sample")
flags.DEFINE_integer("random_seed", 777, "Random seed")
flags.DEFINE_string("model_name_to_use", '', "")
flags.DEFINE_string("model_start_file", "", "")
flags.DEFINE_bool("rerank", False, "If true, will rerank the output generated texts") # to be added soon.
flags.DEFINE_integer("batch_size", 1, "Batch size")

# Parameters for generation
flags.DEFINE_integer("num_generations", 4, "Number of texts to generate")
flags.DEFINE_string("decoding_param", '4', "beam width")
flags.DEFINE_float("beam_diversity_penalty", 10, "diversity penalty")
flags.DEFINE_integer("num_beam_groups", 1, "num beam groups")

# Inherited from reranking code
flags.DEFINE_integer("cola_sbert_batch_size", 64, "Number of documents to batch for COLA and SBERT models")
flags.DEFINE_string("luar_checkpoint_path", 'models/LUAR.pth', "path to LUAR checkpoint")
flags.DEFINE_integer("sent_interval", 4, "how much to batch outputs") #calculate LUAR, SBERT, & COLA scores")




def set_seeds(random_seed):
    """ Set random seeds for reproducibility """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

def get_device() -> str:
    if 'USE_CPU' in os.environ:
        logging.info('using cpu')
        return 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'
        logging.info(f'using device: {device}: {torch.cuda.get_device_name()}')
        return device
    else:
        raise RuntimeError('No GPU found. Set USE_CPU in env to use cpu.')


def load_model(device, model_prefix):
    model = Generator(os.path.join(FLAGS.model_path, model_prefix), seq2seq=("gpt" not in model_prefix), max_input_length = FLAGS.token_max_length, max_output_length=FLAGS.token_max_length+20, device=device)
    if FLAGS.model_start_file is not None and FLAGS.model_start_file != "":
        model.reload(FLAGS.model_start_file)
    model.eval()
    return model



def load_tokenizer(model_prefix="keep_it_simple"):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(FLAGS.model_path, model_prefix))
    return tokenizer



import re
def normalize(text):
    normalized = text.lower()
    normalized = normalized.replace("\n", " ")
    normalized = re.sub(r"(<\w{1,}>)", lambda x: x[0].upper(), normalized)
    normalized = re.sub(r"([ ]{2,})", " ", normalized)
    normalized = re.sub(r"(\W)(?=\1)", "", normalized)
    return normalized

def compute_cosdist_v2(inputs, outputs):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

    distances = []
    for j, (inp, out) in enumerate(zip(inputs, outputs)):
        dist = (1-cos(inp, out)).tolist()
        distances.append(dist)
    return distances




def luar_codist_orig(original_documents, privatized_documents, num_generations, model_path, luar_checkpoint_path, batch_size, device, token_max_length):
    """ Calculate cosine distance between original LUAR embedding and privatized luar_embedding
        Args:
        original_documents (list[str]): List of original documents.
        privatized_documents (list[list[str]]): List of lists of privatized documents.
            Outer list length = number of original documents.
            Inner list length = num_generations.
        num_versions (int): number of privatized versions generated.
        model_path (str): path to models.
        batch_size (int): batch size for model.
        device (str): machine type.
        token_max_length (int): maximum tokens in a document to include in LUAR embedding.
    """

    luar_model = torch.jit.load(luar_checkpoint_path)
    luar_model.to(torch.device(device))
    luar_tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "paraphrase-distilroberta-base-v1"))

    original_documents_batched = create_batches(original_documents, batch_size)
    privatized_documents_batched = create_batches(privatized_documents, batch_size)
    luar_scores = []

    for orig, priv in tqdm(zip(original_documents_batched, privatized_documents_batched), total =len(original_documents_batched)):
        priv_flat = priv
        input_ids, attention_mask = tokenize(orig, luar_tokenizer, token_max_length)
        generated_ids, output_mask = tokenize(priv_flat, luar_tokenizer, token_max_length)

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        generated_ids = generated_ids.to(device)
        output_mask = output_mask.to(device)

        B, N = input_ids.shape
        P = num_generations

        # Change shapes to [B, N, E, max_token_length] for LUAR
        generated_ids = generated_ids.unsqueeze(1).unsqueeze(1)
        output_mask = output_mask.unsqueeze(1).unsqueeze(1)
        input_ids = input_ids.unsqueeze(1).unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        # Get comment embeddings
        inputs, _ = luar_model([input_ids, attention_mask])
        outputs, _ = luar_model([generated_ids, output_mask])
        luar = compute_cosdist_v2(inputs, outputs)
        luar_scores.extend(luar)
    return luar_scores



def get_cola_model(model_path, batch_size):
    """ Loads COLA model. """
    cola_model = os.path.join(model_path, 'roberta-base-CoLA')
    cola_tokenizer = AutoTokenizer.from_pretrained(cola_model)
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    cola = pipeline("text-classification", model=cola_model, tokenizer=cola_tokenizer, device=0, batch_size=batch_size, **tokenizer_kwargs)
    return cola


def calc_cola(cola, docs):
    """ COLA calculation helper function. """
    _ , doc_sents = orig_docs2sents(docs, truncation_length=-1)
    scores = [[int(score['label'] == 'LABEL_1') for score in cola(doc)] for doc in tqdm(doc_sents)]
    return scores

def cola_propdiff(original_documents, privatized_documents, num_generations, model_path, batch_size):
    """Computes the sentence-level COLA differences between two lists of documents.

    Args:
        original_documents (list[str]): List of original documents.
        privatized_documents (list[list[str]]): List of lists of privatized documents.
            Outer list length = number of original documents.
            Inner list length = num_generations.
        num_versions (int): number of privatized versions generated
        model_path (str): path to models
        batch_size (int): batch size for model
    """
    cola = get_cola_model(model_path, batch_size)
    original_scores = calc_cola(cola,original_documents)
    privatized_scores = calc_cola(cola, [txt for gen_txts in privatized_documents for txt in gen_txts])
    privatized_scores = [privatized_scores[i:i+num_generations] for i in range(0, len(privatized_scores), num_generations)]
    diffs = [[[1 - abs(p - o) for p, o in zip(pred, original)] for pred in preds]
            for preds, original in zip(privatized_scores, original_scores)]
    # Weirdness where input text is empty string...
    diffs = [[d if d else [0 for _ in p] for d, p in zip(diff, preds)]
            for diff, preds in zip(diffs, privatized_scores)]
    scores = [[sum(x)/max(1,len(x)) for x in diff] for diff in diffs]
    return scores




def sbert_cosim_v2(original_documents, privatized_documents, num_generations, model_path, batch_size, single_run=False):
    """Computes the cosine similarity between SBERT embeddings of original and privatized docuemnts.

    Args:
        original_documents (list[str]): List of original documents.
        privatized_documents (list[list[str]]): List of lists of privatized documents.
            Outer list length = number of original documents.
            Inner list length = num_generations.
        num_versions (int): number of privatized versions generated.
        model_path (str): path to models.
        batch_size (int): batch size for model.
    """
    model = SentenceTransformer(os.path.join(model_path, 'all-mpnet-base-v2'), device="cuda")
    emb_orig = torch.tensor(model.encode(original_documents, batch_size=batch_size))
    emb_orig = emb_orig.to(device='cuda')
    emb_priv = torch.tensor(model.encode(privatized_documents, batch_size=batch_size))
    emb_priv = emb_priv.to(device='cuda')
    scores = compute_cosdist_v2(emb_orig, emb_priv)
    scores = (1 - np.array(scores)).tolist()
    return scores

# Reranking

def rerank(document_scores, weights=[1, 1, 2,], harmonic_mean=False):
    """Aggregates the metrics for each generated document and returns the indices
    sorted by increasing score.

    Args:
        document_scores (dict): dictionary of metric scores with keys 'sbert',
            'cola', and 'luar'. Values are list[list[float]].
            Outer list length = num_documents.
            Nested list length = num_generations.
        weights (list[int]): weights to apply to each metric when aggregating. Should
             be in the order: sbert, cola, luar.
    """

    document_scores['sbert'] = np.array(document_scores['sbert']) * weights[0]
    document_scores['cola'] = np.array(document_scores['cola']) * weights[1]
    document_scores['luar'] = np.array(document_scores['luar']) * weights[2]
    if harmonic_mean:
        aggregated_scores = [np.mean([1/s, 1/c, 1/l], axis=0) for s, c, l in zip(*document_scores.values())]
    else:
        aggregated_scores = [np.mean([s, c, l], axis=0) for s, c, l  in zip(*document_scores.values())]
    return np.argsort(aggregated_scores)

def privatize(query_text, output_path):
    """Privatizes input texts.

    Args:
        query_text (pandas.DataFrame): DataFrame of texts to privatize.
        output_path (str): Path to write outputs to
    """
    
    logging.info("Loading model")
    device = get_device()
    model = load_model(device, model_prefix=FLAGS.model_name_to_use)
    tokenizer = load_tokenizer(model_prefix=FLAGS.model_name_to_use)
    
    start_i = tokenizer.bos_token_id
    input_texts = []
    generated_sents_overall = []
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    f = open(os.path.join(output_path), 'w', encoding='utf-8') # to reset it
    for queries in tqdm(query_text):
        generated_sents = []
        input_text = [prefix + str(t).replace('\n', ' ') for t in queries['fullText']]
        P = FLAGS.num_generations if FLAGS.rerank else 1

        params = {"max_length": 256, "do_sample": False, "num_beams": int(FLAGS.decoding_param), "diversity_penalty": float(FLAGS.beam_diversity_penalty), "beam_size": 4, "num_return_sequences": P}
        model_outputs = model.generate(input_text, num_runs=P,  generate_by_sent=True, sample=True, sent_interval=FLAGS.sent_interval, **params)
        model_output = model_outputs[0]
        if FLAGS.rerank:
            logging.info("Reranking samples")
            num_return_seq = len(model_outputs)
            reranking_metrics = {'luar': [], 'sbert': [], 'cola': []}
            model_outputs = [x['output_text'] for x in model_outputs]
            reranking_metrics = {'luar': luar_codist_orig(input_text * len(model_outputs), model_outputs, num_return_seq,  FLAGS.model_path, FLAGS.luar_checkpoint_path, FLAGS.batch_size, device, FLAGS.token_max_length),
                                'sbert': sbert_cosim_v2(input_text * len(model_outputs), model_outputs, num_return_seq,  FLAGS.model_path, FLAGS.cola_sbert_batch_size),
                                'cola': cola_propdiff(input_text * len(model_outputs), model_outputs, num_return_seq, FLAGS.model_path, FLAGS.cola_sbert_batch_size)[0]
                                }
            best_sent_idx = rerank(reranking_metrics)[-1]
            best_sample = model_outputs[int(best_sent_idx)]
            model_output = best_sample
            generated_sents.append(model_output)
            queries_dict = queries.to_dict(orient="records")
            for sent, data in zip(generated_sents, queries_dict):
                with open(os.path.join(output_path), 'a', encoding='utf-8') as f:
                    doc = data.copy()
                    doc["fullText"] = sent.replace("Paraphrase:", "").strip()
                    f.write(json.dumps(doc) + "\n")
            

    return None 

def run_privatization(query_JSON_file, output_file):
    """
    Args:
        query_JSON_file (str): Path to input file of queries to privatize.
    """
    queries = pd.read_json(
        query_JSON_file,
        lines=True,
        chunksize=1,
    )
    privatize(queries, output_file)
        

def main(argv):
    start_time = time.time()
    set_seeds(FLAGS.random_seed)

    with torch.no_grad():
        run_privatization(FLAGS.input_data_path, FLAGS.output_path)
    
    logging.info(f"{time.time() - start_time} seconds to create privatized file {FLAGS.output_path}")


if __name__ == "__main__":
    app.run(main)

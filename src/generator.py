from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BartTokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import utils_sampling
import torch, tqdm, time, os
import numpy as np
from .helper_functions import orig_docs2sents
#torch.set_printoptions(threshold=10000)
torch.set_printoptions(profile="full")

def pad(data, padval=0):
    return torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=padval)

class Generator:
    def __init__(self, model_card, max_input_length=300, seq2seq=False, max_output_length=25, device='cuda'):
        self.model_card = model_card
        if "bart" in model_card:
            self.tokenizer = BartTokenizerFast.from_pretrained(self.model_card)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_card)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_card)

        self.start_id = self.tokenizer.bos_token_id
        if "bart" in model_card:
            self.start_id = 2
        if "t5" in model_card:
            self.start_id = 0

        self.seq2seq = seq2seq

        self.model.to(device)
        self.device = device

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.mode = "eval"

    def reload(self, from_file, strict=True):
        loaded_dict = torch.load(from_file)
        loaded_dict = {k.replace("module.module.", ""): v for k, v in loaded_dict.items()}
        print(self.model.load_state_dict(loaded_dict, strict=strict))



    def preprocess_input(self, texts):
        tokenizer_outs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="longest")
        encs = tokenizer_outs["input_ids"]
        attention_mask = tokenizer_outs["attention_mask"]


        encs = encs[:, :self.max_input_length].to(self.device)
        attention_mask = attention_mask[:, :self.max_input_length].to(self.device)
        return encs, attention_mask



    def encode(self, encoded_texts):
        input_ids, attention_mask = encoded_texts

        if not self.seq2seq:
            model_outs = self.model(input_ids=input_ids, past_key_values=None)
            return model_outs["past_key_values"]
        else:
            encoder = self.model.get_encoder()
            encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            return {"encoder_outputs": encoder_outputs, "attention_mask": attention_mask, "past": None}




    def generate_beam_batch(self, bodies, beam_size=3, max_output_length=100,  num_runs=1, **kwargs):
        def clean_text(text):
            cleaned_text = re.sub(r'^[\s.,:;!-]+', '', text)  # Remove leading spaces and punctuations
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra spaces
            return cleaned_text.strip()

        model_inputs, attention_mask = self.preprocess_input(bodies)

        outputs = self.model.generate(model_inputs, **kwargs)
        if num_runs > 1:
            sublists = [outputs[i::num_runs] for i in range(num_runs)]
        else:
            sublists = outputs

        t = []
        for sublist in sublists:
            words = self.tokenizer.batch_decode(sublist, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            t.append({"output_text": " ".join([out_txt for out_txt in words])})

        o = t
        c = t
        if num_runs > 1:
            su = [t[i::num_runs] for i in range(num_runs)]
            o = list(map(lambda x: list(map(lambda y: clean_text(y['output_text']), x)), su))
            c = [list(item) for item in zip(*o)] # stitch into num_runs number of items

        return c



    def generate(self, bodies, max_batch_size=8, beam_size=1, ckpt_runs=1, num_runs=1, progress=False, sort_score=False, keep_unique=False, **kwargs):
        assert not (beam_size > 1 and ckpt_runs > 1), "Cannot ask for beam search and ckpt generation at the same time"

        N_start = len(bodies)
        if num_runs > 1:
            bodies = [body for body in bodies for i in range(num_runs)]
        N = len(bodies)

        outputs = []
        iterator = range(0, N, max_batch_size)
        if progress:
            iterator = tqdm.tqdm(iterator)
        for i in iterator:
            batch_bodies = bodies[i:min(N, i+max_batch_size)]
            with torch.no_grad():
                    batch_outputs = []
                    sents_per_doc, original_sentences = docs2sents(batch_bodies, 400,  flatten=True)
                    batch_outputs_intermediate = []
                    if ("dipper" in self.model_card):
                        lex_diversity = 20
                        order_diversity = 20
                        prompt = f"lexical = {lex_diversity}, order = {order_diversity}"
                    else:
                        prompt = "Paraphrase this:"
                    for j in range(0, len(original_sentences), k):
                        chunk = list(map(lambda x: prompt + "  <sent>" + x + "</sent> ", original_sentences[j:j+k]))
                        tmp = self.generate_beam_batch(chunk, beam_size=beam_size, max_output_length=300, sample=False,num_runs=num_runs, **kwargs)
                        batch_outputs_intermediate.append(tmp)
                    if num_runs > 1:
                        transformed = [list(item) for item in zip(*[sublist[0] for sublist in batch_outputs_intermediate])]
                        for batch in transformed:
                            batch_outputs.append({'output_text': clean_text(' '.join(batch))})
                    else:
                        batch_outputs = [clean_text(" ".join(map(lambda y: y["output_text"], x))) for x in batch_outputs_intermediate ]
            outputs += batch_outputs

        return outputs

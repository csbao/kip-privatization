# Keep It Private
This is the repository for the project Keep it Private: Unsupervised Privatization of Online Text

## How to Install

### 1. Download the model weights

```bash
pip install huggingface_hub
huggingface-cli download csbao/kip-dipper-large --local-dir models/dipper-large
huggingface-cli download csbao/kip-dipper-large dipper_cd_v130.bin --local-dir models
```

### 2. Set up the environment

```bash
./scripts/setup.sh
```

This will create a conda environment named `kip` and install all dependencies.


## Keep It Private Overview
Keep it Private performs authorship transfer by performing authorship transfer using a
seq2seq model that was adversarially fine-tuned via reinforcement learning using
a set of rewards (Privacy, Sense, and Soundness metrics)

## Input Format
The input file should be a JSONL file, one JSON object per line, with a `fullText` key:
```
{"fullText": "hi! this is the first document to privatize."}
{"fullText": "This is another text input. It can have multiple sentences..."}
```

## Command template

```
$ conda activate kip
$ python src/generate.py --input_data_path ${INPUT_DATA_PATH} \
      --output_path ${OUTPUT_PATH} \
      --model_path models \
      --model_name_to_use dipper-large \
      --model_start_file ${BIN_FILE}  \ 
      --token_max_length 256 \
```
### Example

```bash
python src/generate.py --input_data_path {JSONFILE} \
      --output_path {OUTPUT_FILE} \
      --model_path models \
      --model_name_to_use dipper-large \
      --model_start_file models/dipper_cd_v130.bin \
      --token_max_length 256
```

## Parameters
1. `input_data_path`: path to the query documents to be privatized
2. `output_path`: file to save the privatized documents
3. `model_start_file`: path to trained KiP model
4. `model_name_to_use`: path to pre-trained base model
5. `token_max_length`: max cutoff length of output
6. `random_seed`: initialize all random seed to this value
7. `lex_diversity`: lexical diversity level (20, 40, or 60)
8. `order_diversity`: order diversity level (20, 40, or 60)

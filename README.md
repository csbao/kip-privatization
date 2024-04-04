# Keep It Private
This is the repository for the project Keep it Private: Unsupervised Privatization of Online Text

## How to Install

- The `requirements` will need to be installed. 
```
$ scripts/setup.sh
```

This will create a conda environment named `kip`. You will need to download the checkpoints
into the `models` directory.  The script will prompt you to run the correct commands.


## Keep It Private Overview
Keep it Private performs authorship transfer by performing authorship transfer using a
seq2seq model that was adversarially fine-tuned via reinforcement learning using
a set of rewards (Privacy, Sense, and Soundness metrics)


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

```
python src/generate.py --input_data_path {JSONFILE} \
      --output_path {OUTPUT_FILE} \
      --model_path models \
      --model_name_to_use dipper-large \
      --model_start_file models/dipper_v130.bin \
      --token_max_length 256 \
```

## Parameters
1. `input_data_path`: path to the query documents to be privatized
2. `output_path`: file to save the privatized documents
3. `model_start_file`: path to trained KiP model
4. `model_name_to_use`: path to pre-trained base model
5. `token_max_length`: max cutoff length of output
6. `random_seed`: initialize all random seed to this value

# Quick LLM finetuning using modal labs and axolotl

Modal labs is a serverless platform which allows remote python code execution on GPUs and CPUs. Axolotl is a tool for finetuning various models.
Let's run this [example](https://github.com/modal-labs/llm-finetuning) which finetunes a model to output valid SQL query given a valid schema and a natural language query.

Install modal locally. It will prompt you for a key from your registered modal account.

    pip install modal
    python -m modal setup

Install locally hugging face (HF) libraries we will use to inspect our finetuning data prepared by axolotl preprocessing step. You do not need a GPU. You might need to login to HF. You will need access to gated models on HF (base_model in axolotl config file as explained later).
    
    pip install -U transformers datasets

Easiest way to login to HF from your terminal (so you can run code/notebooks).

    huggingface-cli login

Optionally set following to use weights and biases to track the finetune run.

    export ALLOW_WANDB = true

 Add [Hugging face](https://huggingface.co/settings/tokens) and [Weights and biases](https://wandb.ai/authorize) access tokens to modal secrets.
 This modal finetune repo expects the secrets to be name *huggingface* and *wandb*.
 
 Clone the LLM finetune repository from modal and navigate to that.

    git clone https://github.com/modal-labs/llm-finetuning.git
    cd llm-finetuning

Let's use axolotl config file `config/mistral.yml` with finetuning data file `data/sqlqa.jsonl`.
In config file, `format:` shows what axolotl generated prompt will look like. Axolotl ingests data in jsonl format as shown in the data file to generate prompts as required by the model.

    modal run --detach src.train --config=config/mistral.yml --data=data/sqlqa.jsonl --preproc-only

Modal will show you a run-id, which allows you to get the preprocessed data. For e.g.

    Training complete. Run tag: axo-2024-06-15-17-12-42-86d0    

Use this run tag to retrieve the directory from modal onto local directory `_debug_data`

    export RUN_TAG='axo-2024-06-15-17-12-42-86d0'
    modal volume ls example-runs-vol {RUN_TAG}
    rm -rf _debug_data
    modal volume get example-runs-vol {RUN_TAG}/last_run_prepared  _debug_data

Run the following python code and verify the prepared finetuned data.
You can optionally use this [notebook](https://github.com/modal-labs/llm-finetuning/blob/main/nbs/inspect_data.ipynb) in the repo to perform the same.

```python
import yaml, os
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

# find tokenizer
with open('../config/mistral.yml', 'r') as f:
    cfg = yaml.safe_load(f)
model_id = cfg['base_model']
tok = AutoTokenizer.from_pretrained(model_id)
# load HF dataset
ds_dir = Path(f'_debug_data/{RUN_TAG}/last_run_prepared')
ds_path = [p for p in ds_dir.iterdir() if p.is_dir()][0]
ds = load_from_disk(str(ds_path))
# check data
print(tok.decode(ds['input_ids'][0]))

```
Verify the data is formatted correctly as expected from axolotl config file
```
<s> [INST] Using the schema context below, generate a SQL query that answers the question. CREATE TABLE head (age INTEGER) How many heads of the departments are older than 56 ? [/INST]  [SQL] SELECT COUNT(*) FROM head WHERE age > 56 [/SQL]</s>
```
Now resume the finetune run
```
modal run --detach src.train --config=config/mistral.yml\ --data=data/sqlqa.jsonl\ --run-to-resume {RUN_TAG}
```

Check inference on your finetuned model.

    modal run -q src.inference --run-name {RUN_TAG}

To query the finetuned model use the same format as the prepared data
```
[INST] Using the schema context below, generate a SQL query that answers the question. CREATE TABLE head (age INTEGER) How many heads of the departments are older than 56 ? [/INST]
```
and it should give response as

    [SQL] SELECT COUNT(*) FROM head WHERE age > 56 [/SQL]

That's it, you have finetuned a LLM model and performed inference on it!

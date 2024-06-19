# Quick LLM finetuning using modal labs and axolotl (WIP)

Modal labs is a serverless platform which allows remote python code execution on GPUs and CPUs. 
Let's run an example finetune job from here [llm-finetuning](https://github.com/modal-labs/llm-finetuning).

## Setup
Install modal locally. It will prompt you for a key from your registered modal account.

    pip install modal
    python -m modal setup

Install locally hugging face libraries we will use to inspect our finetuning data which is output from axolotl preprocessing step. You do not need a GPU.
    
    pip install -U transformers datasets
 
 Add [Hugging face](https://huggingface.co/settings/tokens) and [Weights and biases](https://wandb.ai/authorize) access tokens to modal. 
 This modal finetune repo expects the secrets to be name **huggingface** and **wandb**. The keys inside the secrets are **HF_TOKEN** and **WANDB_API_KEY** respectively.
 Set environment variable `ALLOW_WANDB = true` locally if you wish to use wandb logging in your project. Check if you have relevant gated permission for LLMs you wish
 to use in hugging face.
 
 Clone the LLM finetune repository from modal and navigate to that.

    git clone https://github.com/modal-labs/llm-finetuning.git
    cd llm-finetuning









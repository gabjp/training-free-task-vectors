# Running Task Vectors

Task vectors are trained with Axolotl. First, you have to convert the model generations to the expected format using the script `cws/prepare_data.py` to train the models. The model- and trait-specific configs are
 `task_vector_sft_configs/<model>-<trait>-ft-baseline.yml`. Update the dataset and output paths in the
selected config, then run, for example:

```bash
axolotl train task_vector_sft_configs/llama-evil.yml
```

Repeat with the corresponding Llama or Qwen config for each trait. Use the
merged model checkpoint produced by Axolotl for evaluation and composition.
Evaluation commands are provided in the main README.

## Negation

Negate a trained LoRA task vector with `reverse_lora.py`:

```bash
python reverse_lora.py \
    --lora_path <LORA_ADAPTER_PATH> \
    --pretrained_model meta-llama/Llama-3.1-8B-Instruct \
    --save_path models/llama_evil_negated
```

`LORA_ADAPTER_PATH` must point to the adapter saved by Axolotl. Change the base
model and adapter for Qwen or another trait.

## Composition

Train one model per trait from the same base model, then compose their merged
checkpoints with `merge-models.py`:

```bash
python merge-models.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --models \
        models/llama_evil \
        models/llama_hallucinating \
        models/llama_sycophantic \
    --output_path models/llama_composed \
    --merge_mode sum
```

For pairwise composition, pass two checkpoints. `sum` adds their parameter
differences from the base model; use `avg` to average them. Evaluate the output
checkpoint as described in the main README.
# Running Contrastive Weight Steering

Train one positive and one negative LoRA adapter for each trait with Axolotl,
using the corresponding configs `task_vector_sft_configs/<model>-<trait>.yml` and  `task_vector_sft_configs/<model>-non-<trait>.yml` . Both adapters
must be trained from the same base model.

## Applying an Edit

```bash
python cws/weight_steer_lora.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --positive_adapter <POSITIVE_ADAPTER_PATH> \
    --negative_adapter <NEGATIVE_ADAPTER_PATH> \
    --coefficient -4 \
    --output_dir models/llama_evil_cws
```

CWS applies `W_base + k(W_pos - W_neg)`. A positive coefficient strengthens the
trait, while a negative coefficient suppresses it. Change the model, adapter paths, coefficient, and output path as
needed. Evaluate the saved checkpoint using the main README.

## Composition

First save one CWS checkpoint per trait, using negative coefficients for
negation. Compose checkpoints created from the same base model with
`merge-models.py`:

```bash
python merge-models.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --models \
        models/llama_evil_cws \
        models/llama_hallucinating_cws \
        models/llama_sycophantic_cws \
    --output_path models/llama_cws_composed \
    --merge_mode sum
```

For pairwise composition, pass two checkpoints. `sum` combines their parameter
differences from the common base model. Evaluate the composed checkpoint as
described in the main README.
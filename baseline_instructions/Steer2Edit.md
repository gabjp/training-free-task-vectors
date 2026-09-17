# Running Steer2Edit

Steer2Edit uses the same input means and steering directions used by TFTV.
First, compute these statistics as described in **Computing a TFTV, Step 2**
of the main README.

## Apply a Steer2Edit edit

The following example edits Llama-3.1-8B-Instruct for the `evil` trait:

```bash
python steer2edit.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --vectors_path directions/in_out_dirs/Llama-3.1-8B-Instruct-evil/evil_linear_out_response_mean_diff.pt \
    --mu_path directions/in_out_dirs/Llama-3.1-8B-Instruct-evil/evil_linear_in_mean_both.pt \
    --save_path outputs/steer2edit/llama-evil \
    --alpha <ALPHA> \
    --mlp_rho <MLP_RHO> \
    --attn_rho <ATTN_RHO>
```

Here, `--vectors_path` points to the steering direction and `--mu_path` points
to the corresponding input mean. `--alpha`, `--mlp_rho`, and `--attn_rho` are
Steer2Edit hyperparameters.

The edited model is saved at `--save_path`. It can then be evaluated by using
that directory as `<MODEL_PATH>` in the evaluation commands from the main
README. Replace `evil` with `hallucinating`, `sycophantic`, or another supported trait
as needed.

## Running the Negation Experiments

To reverse a Steer2Edit update, add the `--reverse` flag. For example:

```bash
python steer2edit.py \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --vectors_path directions/in_out_dirs/Llama-3.1-8B-Instruct-evil/evil_linear_out_response_mean_diff.pt \
    --mu_path directions/in_out_dirs/Llama-3.1-8B-Instruct-evil/evil_linear_in_mean_both.pt \
    --save_path models/llama_evil_negated \
    --alpha 0.3 \
    --mlp_rho 0.7 \
    --attn_rho 0.1 \
    --reverse
```

## Running the Composition Experiments

First save one Steer2Edit checkpoint per trait. Use --reverse for suppression
edits. All checkpoints must use the same base model.

Compose them with merge-models.py:
```bash
python merge-models.py \
    --base_model meta-llama/Llama-3.1-8B-Instruct \
    --models \
        models/llama_evil_negated \
        models/llama_hallucinating_negated \
        models/llama_sycophantic_negated \
    --output_path models/llama_composed_negated \
    --merge_mode sum
```
For pairwise composition, pass only two checkpoints. sum adds their parameter
differences from the base model; Evaluate the merged
checkpoint as described in the main README.
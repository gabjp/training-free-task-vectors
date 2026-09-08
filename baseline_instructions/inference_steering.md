# Inference Steering Baseline

This file provides example commands for running the **inference-time activation steering** baseline used in the TFTV paper. The implementation follows the Persona Vectors setup.

## 1. Generate contrastive completions

Inference steering uses the same positive and negative generations used to construct TFTVs.

If these generations have not already been computed, generate them with `eval.eval_persona` as described in the main README. The released generations can also be reused directly.

For example, the corresponding files are expected to have the form:

```text
results/persona_vector_gen/<trait>_pos_instruct.csv
results/persona_vector_gen/<trait>_neg_instruct.csv
```

## 2. Compute Persona Vectors

Before running inference steering, compute the activation-space Persona Vector with `generate_persona_vec.py`.

The same positive and negative generations used for TFTV can be reused here:

```bash
python generate_persona_vec.py \
    --model_name <MODEL_NAME> \
    --pos_path results/persona_vector_gen/<TRAIT>_pos_instruct.csv \
    --neg_path results/persona_vector_gen/<TRAIT>_neg_instruct.csv \
    --trait <TRAIT> \
    --save_dir directions/persona_vectors/<MODEL>-<TRAIT>/
```

The script filters generations using the trait and coherence scores, computes positive-minus-negative hidden-state differences across layers, and saves:

```text
directions/persona_vectors/<MODEL>-<TRAIT>/
├── <TRAIT>_prompt_avg_diff.pt
├── <TRAIT>_response_avg_diff.pt
└── <TRAIT>_prompt_last_diff.pt
```

For the inference-steering experiments, use the response-average Persona Vector:

```text
<TRAIT>_response_avg_diff.pt
```

## 3. Persona evaluation

Inference steering is applied by adding the Persona Vector to model activations during generation.

Example:

```bash
python -m eval.eval_persona \
    --model <MODEL_NAME> \
    --trait <TRAIT> \
    --output_path results/inference_steering/<OUTPUT>.csv \
    --persona_instruction_type <PERSONA_INSTRUCTION_TYPE> \
    --assistant_name <ASSISTANT_NAME> \
    --judge_model gpt-4.1-mini-2025-04-14 \
    --version eval \
    --steering_type response \
    --coef <COEFFICIENT> \
    --vector_path directions/persona_vectors/<MODEL>-<TRAIT>/<TRAIT>_response_avg_diff.pt \
    --layer <LAYER>
```

To evaluate steering in the opposite direction, reverse the sign of the coefficient:

```bash
--coef -<COEFFICIENT>
```

## 4. Utility evaluation

`eval_utility.py` supports inference steering directly.

### MMLU

```bash
python eval_utility.py \
    --model <MODEL_NAME> \
    --task mmlu \
    --device cuda:0 \
    --batch_size <BATCH_SIZE> \
    --steering_vector_path <PERSONA_VECTOR_PATH> \
    --steering_coef <COEFFICIENT> \
    --steering_layer <LAYER> \
    --steering_positions response \
    --output_file results/inference_steering/mmlu.json
```

### GSM8K

```bash
python eval_utility.py \
    --model <MODEL_NAME> \
    --task gsm8k \
    --device cuda:0 \
    --batch_size <BATCH_SIZE> \
    --steering_vector_path <PERSONA_VECTOR_PATH> \
    --steering_coef <COEFFICIENT> \
    --steering_layer <LAYER> \
    --steering_positions response \
    --output_file results/inference_steering/gsm8k.json
```

## 5. Out-of-distribution evaluation

The same interface can be used for the OOD tasks.

### Moral Stories

```bash
python eval_utility.py \
    --model <MODEL_NAME> \
    --task moral_stories \
    --device cuda:0 \
    --batch_size <BATCH_SIZE> \
    --steering_vector_path <PERSONA_VECTOR_PATH> \
    --steering_coef <COEFFICIENT> \
    --steering_layer <LAYER> \
    --steering_positions response \
    --output_file results/inference_steering/moral_stories.json
```

### TruthfulQA

```bash
python eval_utility.py \
    --model <MODEL_NAME> \
    --task truthfulqa \
    --device cuda:0 \
    --batch_size <BATCH_SIZE> \
    --steering_vector_path <PERSONA_VECTOR_PATH> \
    --steering_coef <COEFFICIENT> \
    --steering_layer <LAYER> \
    --steering_positions response \
    --output_file results/inference_steering/truthfulqa.json
```

### Sycophancy

```bash
python eval_utility.py \
    --model <MODEL_NAME> \
    --task sycophancy \
    --device cuda:0 \
    --batch_size <BATCH_SIZE> \
    --steering_vector_path <PERSONA_VECTOR_PATH> \
    --steering_coef <COEFFICIENT> \
    --steering_layer <LAYER> \
    --steering_positions response \
    --output_file results/inference_steering/sycophancy.json
```

## 6. Attention-only steering

For the matched-control analysis, inference steering can also be applied specifically at the attention output projection.

When using a steering-vector file keyed by module name, pass:

```bash
--attn_only
```

to `eval_utility.py` and to `eval.eval_persona`.

Example:

```bash
python eval_utility.py \
    --model <MODEL_NAME> \
    --task mmlu \
    --device cuda:0 \
    --batch_size <BATCH_SIZE> \
    --steering_vector_path <ATTENTION_VECTOR_PATH> \
    --steering_coef <COEFFICIENT> \
    --steering_layer <LAYER> \
    --steering_positions response \
    --attn_only \
    --output_file results/inference_steering/attention_only_mmlu.json
```

For standard inference steering, omit `--attn_only`. If you are using this option, pass the steering vector computed from TFTV instructions to `--steering_vector_path`.

## 7. Inference Steering Composition

Inference steering directions can be composed by taking a weighted sum of the corresponding Persona Vectors.

Use `merge-vectors.py`:

```bash
python merge-vectors.py \
    --inputs \
        directions/persona_vectors/evil_response_avg_diff.pt \
        directions/persona_vectors/hallucinating_response_avg_diff.pt \
        directions/persona_vectors/sycophantic_response_avg_diff.pt \
    --coefs 1.0 1.0 1.0 \
    --output directions/persona_vectors/composed_evil_hallucinating_sycophantic.pt
```
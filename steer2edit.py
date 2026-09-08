import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--vectors_path", type=str, required=True)
    parser.add_argument("--mu_path", required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--mlp_rho", type=float, required=True)
    parser.add_argument("--attn_rho", type=float, required=True)

    args = parser.parse_args()
    mu = torch.load(args.mu_path, map_location="cpu")
    vectors = torch.load(args.vectors_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    num_layers = len(model.model.layers)

    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim", None) or (hidden_size // num_heads)

    # build layer list
    LAYERS_MLP = [
        f"model.layers.{i}.mlp.down_proj"
        for i in range(num_layers)
    ]

    LAYERS_ATTN = [
        f"model.layers.{i}.self_attn.o_proj"
        for i in range(num_layers)
    ]

    sd = model.state_dict()

    for key in tqdm(LAYERS_MLP):
        v = vectors[key]
        v = v / torch.linalg.vector_norm(v)

        W = sd[key + ".weight"]
        v = v.to(device=W.device, dtype=W.dtype)

        g = torch.stack([F.cosine_similarity(v, W[:, column] * mu[key][column].to(device=W.device, dtype=W.dtype), dim=0) \
                         for column in range(W.size()[1])]).squeeze()
                         
        k_scalar = torch.stack([torch.sign(torch.dot(W[:, column], v)) for column in range(W.size()[1])]).squeeze()

        lamb = torch.sign(g) * (1 / (args.mlp_rho * (1 - args.alpha))) * torch.clamp(g.abs() - args.mlp_rho * args.alpha, min=0)

        weight_direction = v.unsqueeze(1) @ (lamb * k_scalar).unsqueeze(0) 
        sd[key + ".weight"] =  W + weight_direction


    for key in tqdm(LAYERS_ATTN):
        v = vectors[key]
        v = v / torch.linalg.vector_norm(v)

        W = sd[key + ".weight"]                      # [H, H]
        v = v.to(device=W.device, dtype=W.dtype)

        x_mean = mu[key].to(device=W.device, dtype=W.dtype)   # [H]
        x_mean_heads = x_mean.view(num_heads, head_dim)       # [A, D]

        weight_direction = torch.zeros_like(W)

        for j in range(num_heads):
            c0, c1 = j * head_dim, (j + 1) * head_dim
            W_block = W[:, c0:c1]                                   # [H, D]

            # scalar score g
            g = F.cosine_similarity(v, W_block @ x_mean_heads[j], dim=0)

            lam = torch.sign(g) * (1 / (args.attn_rho * (1 - args.alpha))) * \
                torch.clamp(g.abs() - args.attn_rho * args.alpha, min=0)

            if lam == 0:
                continue

            # k_hat in the head subspace
            k_vec = W_block.T @ v                                    # [D]
            k_norm = torch.linalg.vector_norm(k_vec)

            if k_norm == 0:
                continue

            k_hat = k_vec / k_norm                                   # [D]

            # rank-1 block update: [H, 1] @ [1, D] -> [H, D]
            block_direction = lam * (v.unsqueeze(1) @ k_hat.unsqueeze(0))
            weight_direction[:, c0:c1] = block_direction

        sd[key + ".weight"] = W + weight_direction

    model.load_state_dict(sd)


    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)
    print("Saved to", args.save_path)



if __name__ == "__main__":
    main()
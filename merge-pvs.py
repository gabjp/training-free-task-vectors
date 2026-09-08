import argparse
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="1 to 3 tensor paths")
    parser.add_argument("--coefs", nargs="+", type=float, required=True, help="one coef per tensor")
    parser.add_argument("--output", required=True, help="output tensor path")
    args = parser.parse_args()

    if not (1 <= len(args.inputs) <= 3):
        raise ValueError("Provide between 1 and 3 input tensors.")

    if len(args.inputs) != len(args.coefs):
        raise ValueError("--inputs and --coefs must have the same length.")

    result = 0
    for path, coef in zip(args.inputs, args.coefs):
        tensor = torch.load(path, map_location="cpu")
        result = result + coef * tensor

    torch.save(result, args.output)

if __name__ == "__main__":
    main()
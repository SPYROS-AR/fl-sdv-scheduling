import argparse
import numpy as np

def generate_parameters(num_sweeps: int, output_file: str = "params.txt"):
    # Set a seed for reproducible sweep generation
    np.random.seed(100)

    # Log-uniform distributions for penalty weights (alpha, gamma)
    alphas = np.exp(np.random.uniform(np.log(0.001), np.log(0.01), num_sweeps))
    gammas = np.exp(np.random.uniform(np.log(0.005), np.log(0.1), num_sweeps))

    # Uniform distributions for linear scalars (beta, delta)
    betas = np.random.uniform(0.5, 2.0, num_sweeps)
    deltas = np.random.uniform(10.0, 30.0, num_sweeps)

    with open(output_file, 'w') as f:
        for i in range(num_sweeps):
            f.write(f"{alphas[i]:.6f} {betas[i]:.4f} {gammas[i]:.6f} {deltas[i]:.4f}\n")
            
    print(f"Successfully generated {num_sweeps} parameter combinations in {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate hyperparameter sweep combinations.")
    parser.add_argument("--num_sweeps", type=int, default=100, help="Number of parameter combinations to generate")
    args = parser.parse_args()
    
    generate_parameters(args.num_sweeps)
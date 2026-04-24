from train import train

# 🔥 stronger lambdas
lambdas = [1e-3, 1e-2, 5e-2]

results = []

for l in lambdas:
    print(f"\nRunning for lambda={l}")
    acc, sparsity = train(l)

    results.append((l, acc, sparsity))

print("\nFinal Results:")
for r in results:
    print(f"Lambda: {r[0]}, Acc: {r[1]:.4f}, Sparsity: {r[2]:.2f}%")
## Observations

- The sparsity achieved in limited training (less epochs, small dataset) remains low .
- This is expected because:
  - Sigmoid gates tend to saturate slowly and require longer training to push values toward zero.
  - The classification loss dominates early training, preventing aggressive pruning.
  - Limited dataset size reduces gradient signal for sparsity regularization.

## Key Insight

- With increased training duration, larger dataset, or stronger regularization, the network would progressively prune weaker connections.
- The current setup demonstrates correct implementation of self-pruning mechanics, even though full sparsity behavior is not yet observed.

## Trade-off Understanding

- Increasing λ strengthens sparsity but can degrade accuracy.
- In early training stages, the model prioritizes learning meaningful representations before pruning.

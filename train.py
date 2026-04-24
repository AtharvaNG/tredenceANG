import torch
import torch.nn as nn
from model import Net
from utils import get_data, compute_sparsity

def train(lambda_val=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, test_loader = get_data()

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(2):  # 🔥 slightly more than 1
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            cls_loss = criterion(outputs, y)

            # 🔥 improved sparsity loss
            sparsity_loss = 0
            for module in model.modules():
                if hasattr(module, "gate_scores"):
                    gates = torch.sigmoid(module.gate_scores)
                    sparsity_loss += torch.mean(gates ** 2)

            loss = cls_loss + lambda_val * 100 * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    acc = evaluate(model, test_loader, device)
    sparsity = compute_sparsity(model)

    return acc, sparsity


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total
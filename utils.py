import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_data():
    transform = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # small dataset for speed
    train_subset = Subset(train_dataset, range(500))
    test_subset = Subset(test_dataset, range(200))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64)

    return train_loader, test_loader


def compute_sparsity(model):
    total = 0
    zero = 0

    for module in model.modules():
        if hasattr(module, "gate_scores"):
            gates = torch.sigmoid(module.gate_scores)
            total += gates.numel()
            zero += (gates < 0.2).sum().item()

    return 100 * zero / total
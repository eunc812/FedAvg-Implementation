import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleNet


@torch.no_grad()
def evaluate(model, testloader, device="cpu"):
    model.eval()
    correct, total = 0, 0
    for x, y in testloader:
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), -1)  # MNIST flatten
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/fedavg_mnist.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tfm = transforms.Compose([transforms.ToTensor()])
    testset = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    testloader = DataLoader(testset, batch_size=256, shuffle=False)

    model = SimpleNet(dim=784, num_classes=10).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    acc = evaluate(model, testloader, device=device)
    print(f"\nTest Accuracy: {acc*100:.2f}%")


if __name__ == "__main__":
    main()

import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import SimpleNet
from client import Client
from server import Server


def make_iid_client_loaders(trainset, num_clients=10, batch_size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(trainset), generator=g)
    shard = len(trainset) // num_clients

    loaders = []
    for k in range(num_clients):
        part = idx[k * shard:(k + 1) * shard].tolist()
        subset = Subset(trainset, part)
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False))
    return loaders


def train_fedavg(clients, global_model, rounds=20, local_epochs=1, fraction=1.0, seed=0):
    torch.manual_seed(seed)
    server = Server(global_model)

    num_clients = len(clients)
    m = max(1, int(num_clients * fraction))

    for t in range(rounds):
        selected = torch.randperm(num_clients)[:m].tolist()

        client_states, client_sizes = [], []
        for cid in selected:
            client = clients[cid]
            client.set_weights(server.global_model)          # broadcast
            state, size = client.local_update(local_epochs)  # LocalUpdate
            client_states.append(state)
            client_sizes.append(size)

        server.aggregate_fedavg(client_states, client_sizes) # aggregate
        print(f"--- Round {t+1}/{rounds} done ---")

    return server.global_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="checkpoints/fedavg_mnist.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    torch.manual_seed(args.seed)

    tfm = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)

    client_loaders = make_iid_client_loaders(
        trainset, num_clients=args.clients, batch_size=args.batch_size, seed=args.seed
    )

    global_model = SimpleNet(dim=784, num_classes=10).to(device)
    clients = [Client(global_model, dl, lr=args.lr, device=device) for dl in client_loaders]

    global_model = train_fedavg(
        clients=clients,
        global_model=global_model,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        fraction=args.fraction,
        seed=args.seed,
    ).to(device)

    # save
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": global_model.state_dict(),
            "args": vars(args),
        },
        args.save_path,
    )
    print(f"\nSaved checkpoint to: {args.save_path}")


if __name__ == "__main__":
    main()

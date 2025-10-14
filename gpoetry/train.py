import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .data import SpanishPoetryDataset


def train_step(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()

    train_loss = 0
    for batch in train_dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        logits = model(x)

        # The logits have shape (batch_size (b), sequence_len (t), vocab_size (v))
        # To use the loss function we need to reshape them to (batch_size * sequence_len, vocab_size)
        b, t, v = logits.shape
        logits = logits.view(b * t, v)
        y = y.view(b * t)

        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    return train_loss


def val_step(
    model: nn.Module,
    val_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> float:
    model.eval()
    val_loss = 0

    with torch.inference_mode():
        for batch in val_dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)

            batch_size_current, seq_len, vocab_size = logits.shape
            logits = logits.view(batch_size_current * seq_len, vocab_size)
            y = y.view(batch_size_current * seq_len)

            loss = loss_fn(logits, y)

            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    return val_loss


def train(
    model: nn.Module,
    dataset: SpanishPoetryDataset,
    train_size: float = 0.8,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> None:
    model.to(device)

    n = len(dataset)
    train_len = int(train_size * n)
    val_len = n - train_len
    x_train, x_val = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        x_train,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        x_val,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # If we area using padding tokens we need to remove them from the logits
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = train_step(
            model=model,
            train_dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss = val_step(
            model=model,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"Epoch: {epoch} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
        )

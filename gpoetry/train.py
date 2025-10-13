import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from .data import SpanishPoetryDataset


def train_step(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    block_size: int | None = None,
) -> float:
    train_loss = 0
    for batch in train_dataloader:
        if not block_size:
            batch = batch[:block_size].to(device)
        else:
            batch = batch.to(device)

        # We use all tokens except the last one (x) for training the model
        # Then we use all tokens (y) to use it in the loss function
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = model(x)

        # The logits have shape (batch_size (b), sequence_len (t), vocab_size (v))
        # To use the loss function we need to reshape them to (batch_size * sequence_len, vocab_size)
        b, t, v = logits.shape
        logits = logits.reshape(b * t, v)
        y = y.reshape(b * t)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    return train_loss


def test_step(
    model: nn.Module,
    test_dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    block_size: int | None = None,
) -> float:
    model.eval()
    test_loss = 0

    with torch.inference_mode():
        for batch in test_dataloader:
            if not block_size:
                inputs = batch.to(device)
            else:
                inputs = batch[:block_size].to(device)

            x = inputs[:, :-1]
            y = inputs[:, 1:]

            logits = model(x)

            batch_size_current, seq_len, vocab_size = logits.shape
            logits = logits.reshape(batch_size_current * seq_len, vocab_size)
            y = y.reshape(batch_size_current * seq_len)

            loss = loss_fn(logits, y)

            test_loss += loss.item()

    test_loss /= len(test_dataloader)
    return test_loss


def train(
    model: nn.Module,
    dataset: SpanishPoetryDataset,
    train_size: float = 0.8,
    batch_size: int = 32,
    epochs: int = 10,
    lr: float = 1e-3,
    pad_token_id: int = 0,
    device: str = "cpu",
    block_size: int | None = None,
) -> None:
    model.to(device)

    n = len(dataset)
    train_len = int(train_size * n)
    val_len = n - train_len
    x_train, x_test = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(
        x_train,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=lambda x: pad_sequence(
            x, batch_first=True, padding_value=pad_token_id
        ),
        pin_memory=True,
        shuffle=True,
    )
    test_loader = DataLoader(
        x_test,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=lambda x: pad_sequence(
            x, batch_first=True, padding_value=pad_token_id
        ),
        pin_memory=True,
    )

    # If we area using padding tokens we need to remove them from the logits
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = train_step(
            model=model,
            train_dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            block_size=block_size,
        )
        test_loss = test_step(
            model=model,
            test_dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
            block_size=block_size,
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | "
                f"train_loss: {train_loss:.4f} | "
                f"test_loss: {test_loss:.4f} | "
            )

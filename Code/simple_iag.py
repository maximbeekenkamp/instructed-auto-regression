import warnings
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# Filter out specific PyTorch transformer warning about nested tensors
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*"
)

torch.manual_seed(123)
np.random.seed(123)

# Parameters
input_dim = 3  # (t, x_coords, X(t, x))
hidden_dim = 64
num_layers = 2
num_heads = 4
dropout = 0.1
learning_rate = 0.001
batch_size = 16
num_epochs = 100
max_seq_len = 128
patience = 50


def generate_ode_data(
    t_range: Tuple[float, float] = (0, 1), dt: float = 0.01, init_val: float = 1.0
) -> Tuple[List[Tuple[List[List[float]], float]], np.ndarray, np.ndarray]:
    """
    Generates data from analytical solution for the ODE dx/dt = x; for x(0) = 1, t in [1,0]

    Args:
        t_range: Tuple of (start, end) for time domain.
        dt: Time step size.
        init_val: Initial value for x(0).

    Returns:
        A tuple containing:
            - sequences: List of (input_sequence, target) pairs for training.
            - times: Array of time values.
            - X_values: Array of analytical solution values X(t).
    """
    times = np.arange(t_range[0], t_range[1] + dt, dt)
    x_coords = np.zeros_like(times)  # For a 1D ODE, x_coords is just a placeholder

    X_values = init_val * np.exp(times)

    # Create sequences of observations as data batches
    sequences = []
    for i in range(len(times) - 1):
        for seq_len in range(
            1, min(max_seq_len + 1, i + 1)
        ):  # Complexity scales quadratically with sequence length
            if i >= seq_len:
                input_seq = []
                for j in range(seq_len):
                    idx = i - seq_len + j
                    input_seq.append([times[idx], x_coords[idx], X_values[idx]])

                target = X_values[i]

                sequences.append((input_seq, target))

    return sequences, times, X_values


class ODEDataset(Dataset):
    """
    Custom PyTorch Dataset for ODE sequence data.
    """

    def __init__(self, sequences: List[Tuple[List[List[float]], float]]) -> None:
        """
        Initialises ODE Dataset class.

        Args:
            sequences: List of (input_sequence, target_value) pairs where each input sequence is a
            list of [time, x_coord, value] entries and target is a float.
        """
        self.sequences = sequences

    def __len__(self) -> int:
        """
        Get the number of sequences in the dataset.

        Returns:
            The total number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[List[float]], float]:
        """
        Get a specific sequence target pair.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            A tuple containing (input_sequence, target).
        """
        input_seq, target = self.sequences[idx]
        # Return raw sequences, will be processed in collate_fn
        return input_seq, target


# Transformer Encoder Model for IAG
class IAGTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialise the IAGTransformer model.

        Args:
            input_dim: Dimension of the input features.
            hidden_dim: Dimension of the hidden layers.
            num_layers: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super(IAGTransformer, self).__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.positional_encoder = nn.Parameter(torch.zeros(max_seq_len, hidden_dim))
        nn.init.normal_(self.positional_encoder, mean=0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.output_projection = nn.Linear(
            hidden_dim, 1
        )  # Ouput dim = 1 (predicts a single value)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass thorugh the IAGTransformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            mask: Optional padding mask where true values don't attend, and false values attend.
                Has shape: [batch_size, seq_len].

        Returns:
            Output tensor of shape [batch_size, 1] with predictions.
        """
        # Project input to hidden dimension
        x = self.input_projection(x)

        seq_len = x.size(1)
        x = x + self.positional_encoder[:seq_len, :]

        # Pass through transformer encoder
        # In PyTorch, attention mask works as: True = don't attend, False = attend
        # Add warning filter to suppress the nested tensor warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*nested tensors is in prototype stage.*",
            )
            encoded = self.transformer_encoder(x, src_key_padding_mask=mask)

        # For each sequence, use the last non-padded token for prediction
        batch_size = x.size(0)

        if mask is not None:
            # Find the last non-masked position for each sequence
            seq_lengths = (~mask).sum(dim=1).long() - 1  # -1 to get 0-based index

            # Gather the features at these positions
            batch_indices = torch.arange(batch_size, device=x.device)
            features = encoded[batch_indices, seq_lengths]
        else:
            # If no masking, use the last token
            features = encoded[:, -1]

        # Project to output dimension
        output = self.output_projection(features)

        return output


# Custom collate function for padding sequences
def collate_fn(
    batch: List[Tuple[List[List[float]], float]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for padding sequences of different lengths.

    Args:
        batch: List of (sequence, target) pairs.

    Returns:
        A tuple containing:
            - padded_seqs: Tensor of padded input sequences [batch_size, max_len, input_dim].
            - targets: Tensor of target values [batch_size, 1].
            - padding_mask: Boolean mask for padding positions [batch_size, max_len].

    """
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, targets = zip(*batch)

    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    lengths = [seq.size(0) for seq in sequences]
    max_len = max(lengths)

    padded_seqs = torch.zeros(
        len(sequences), max_len, sequences[0].size(1), dtype=torch.float32
    )

    # Fill with actual data
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq

    # Create padding mask (True for padding positions, False for actual data)
    padding_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, length in enumerate(lengths):
        padding_mask[i, length:] = True

    return padded_seqs, targets, padding_mask


# Validation function
def validate_model(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module
) -> float:
    """
    Validation phase executed in train_model().

    Args:
        model: The model being validated.
        val_loader: DataLoader for validation data.
        criterion: Loss function.

    Returns:
        The validation loss.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, mask in val_loader:
            outputs = model(inputs, mask)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    num_epochs: int,
    patience: int,
) -> Tuple[List[float], List[float], float]:
    """
    Train the model with early stopping.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimisation algorithm to use.
        scheduler: Learning rate scheduler.
        num_epochs: Maximum number of training epochs.
        patience: Number of epochs with no improvement after which training will be stopped.

    Returns:
        A tuple containing:
            - training_losses: List of average training loss per epoch.
            - validation_losses: List of average validation loss per epoch.
            - best_val_loss: Float of the best validation loss.
    """
    model.train()
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0

    # Rich progress bar
    console = Console()
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        # Main epoch progress
        epoch_task = progress.add_task("[green]Training progress:", total=num_epochs)

        for epoch in range(num_epochs):
            # Batch progress for current epoch
            batch_task = progress.add_task(
                f"[cyan]Epoch {epoch+1}/{num_epochs}:", total=len(train_loader)
            )

            # Training phase
            model.train()
            epoch_loss = 0
            for inputs, targets, mask in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs, mask)
                loss = criterion(outputs, targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                progress.update(batch_task, advance=1)

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            val_task = progress.add_task(
                f"[yellow]Validating Epoch {epoch+1}:", total=1
            )
            avg_val_loss = validate_model(model, val_loader, criterion)
            val_losses.append(avg_val_loss)
            progress.update(val_task, advance=1)
            progress.remove_task(val_task)

            # Update learning rate based on validation loss
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Update epoch progress bar
            progress.update(
                epoch_task,
                description=f"[green]Training progress: (Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6e})",
            )

            # Remove the batch progress bar after epoch completion
            progress.remove_task(batch_task)

            # Check for early stopping
            if early_stop_counter >= patience:
                console.print(
                    f"[bold yellow]Early stopping triggered after {epoch+1} epochs!"
                )
                # Restore best model
                model.load_state_dict(best_model_state)
                break

    console.print("[bold green]Training completed!")
    console.print("[bold]Final Training Loss:", f"{train_losses[-1]:.6f}")
    console.print("[bold]Final Validation Loss:", f"{val_losses[-1]:.6f}")
    console.print("[bold]Best Validation Loss:", f"{best_val_loss:.6f}")

    return train_losses, val_losses, best_val_loss


def predict_trajectory(
    model: nn.Module, t_values: np.ndarray, x0: float = 1.0
) -> List[float]:
    """Predict the ODE solution trajectory using the trained model.

    This function uses the trained model to generate a sequence of predicted values starting from
    an initial condition x0. The model is auto-regressive with each prediction feeding bask as the
    input for the next prediction.

    Args:
        model: The trained IAGTransformer model.
        t_values: Array of time points.
        x0: Initial value x(0).

    Returns:
        A list of predicted values corresponding to each time point in t_values.
    """
    model.eval()
    predictions = [x0]

    current_sequence = [[0.0, 0.0, x0]]  # [t, x_coords, X(t)]

    # Use rich for prediction progress
    console = Console()
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        prediction_task = progress.add_task(
            "[green]Generating predictions...", total=len(t_values) - 1
        )

        with torch.no_grad():
            for i, t in enumerate(t_values[1:]):
                input_tensor = torch.tensor([current_sequence], dtype=torch.float32)
                # No padding mask needed because we're not batching
                mask = None

                next_value = model(input_tensor, mask).item()
                predictions.append(next_value)

                current_sequence.append([t, 0.0, next_value])

                if len(current_sequence) > max_seq_len:
                    current_sequence = current_sequence[-max_seq_len:]

                progress.update(prediction_task, advance=1)

    return predictions


# Main execution
def main() -> None:
    """Main execution.

    Orchestrates the entire workflow.
        1. Generate training data
        2. Create and train the IAGTransformer model
        3. Generate predictions and visualise results
    """
    console = Console()
    console.print("[bold magenta]ODE Solution with Implicit Attention Guidance[/]")

    # Generate data
    with console.status("[bold green]Generating training data..."):
        sequences, times, true_values = generate_ode_data()

    console.print(f"[bold green][/]Generated {len(sequences)} training sequences")

    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    console.print(f"[bold]Training set:[/] {len(train_sequences)} sequences")
    console.print(f"[bold]Validation set:[/] {len(val_sequences)} sequences")

    # Create data loaders
    train_dataset = ODEDataset(train_sequences)
    val_dataset = ODEDataset(val_sequences)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Print a sample batch to verify
    for inputs, targets, mask in train_loader:
        console.print("[bold]Batch information:[/]")
        console.print(f"Inputs shape: {inputs.shape}")
        console.print(f"Targets shape: {targets.shape}")
        console.print(f"Mask shape: {mask.shape}")

        # Sample first sequence
        sample_table = Table(title="First 3 entries of first sequence")
        sample_table.add_column("Time")
        sample_table.add_column("X Coord")
        sample_table.add_column("Value")

        for i in range(min(3, inputs.shape[1])):
            sample_table.add_row(
                f"{inputs[0][i][0]:.4f}",
                f"{inputs[0][i][1]:.4f}",
                f"{inputs[0][i][2]:.4f}",
            )

        console.print(sample_table)
        console.print(f"Sample target: {targets[0].item():.4f}")
        break

    model = IAGTransformer(input_dim, hidden_dim, num_layers, num_heads, dropout)
    model_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model parameters:[/] {model_params:,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    console.print("\n[bold yellow]Starting Training...[/]")

    train_losses, val_losses, best_val_loss = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        patience,
    )

    console.print("\n[bold yellow]Generating Predictions...[/]")

    predictions = predict_trajectory(model, times)

    console.print("\n[bold green]Creating visualizations...[/]")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(times, true_values, "b-", label="True Solution (e^t)")
    plt.plot(times, predictions, "r--", label="IAG Predictions")
    plt.xlabel("Time (t)")
    plt.ylabel("X(t)")
    plt.title("ODE Solution: dx/dt = x, x(0) = 1")
    plt.legend()
    plt.grid(True)
    plt.savefig("../Data/Images/simple_ode_comparison.png")
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("../Data/Images/simple_ode_training_validation_loss.png")
    plt.show()

    console.print(
        "\n[bold green]Done! Saved plots as iag_ode_comparison.png and iag_training_validation_loss.png[/]"
    )


if __name__ == "__main__":
    main()

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
from scipy.integrate import solve_ivp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# Filter out specific PyTorch transformer warning about nested tensors
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*nested tensors is in prototype stage.*"
)

torch.manual_seed(123)
np.random.seed(123)

# Parameters
input_dim = 4  # (t, x_coords, x(t), v(t)); where v(t) = dx/dt
hidden_dim = 64
num_layers = 2
num_heads = 4
dropout = 0.1
learning_rate = 0.001
batch_size = 16
num_epochs = 100
max_seq_len = 128
patience = 50


def van_der_pol_equation(t, y, mu=1.0):
    """
    Van der Pol oscillator defined as a system of first-order ODEs:
    dx/dt = v
    dv/dt = mu*(1-x^2)*v - x

    Args:
        t: Time variable (not used in autonomous systems but required by solve_ivp)
        y: State vector [x, v]
        mu: Parameter controlling nonlinearity and damping

    Returns:
        dy/dt: State derivatives [dx/dt, dv/dt]
    """
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x
    return [dxdt, dvdt]


def generate_van_der_pol_data(
    t_range: Tuple[float, float] = (0, 10),
    dt: float = 0.05,
    mu: float = 1.0,
    initial_conditions: List[float] = [0.5, 0],
) -> Tuple[List[Tuple[List[List[float]], List[float]]], np.ndarray, np.ndarray]:
    """
    Generates data from numerical solution for the Van der Pol oscillator

    Args:
        t_range: Tuple of (start, end) for time domain.
        dt: Time step size.
        mu: Parameter controlling nonlinearity and damping of the oscillator.
        initial_conditions: Initial values for [x(0), v(0)].

    Returns:
        A tuple containing:
            - sequences: List of (input_sequence, target) pairs for training.
            - times: Array of time values.
            - solution: Array of numerical solution values [x(t), v(t)].
    """
    times = np.arange(t_range[0], t_range[1] + dt, dt)

    # Solve the ODE using scipy's solve_ivp (with RK45 method)
    sol = solve_ivp(
        lambda t, y: van_der_pol_equation(t, y, mu),
        [t_range[0], t_range[1]],
        initial_conditions,
        method="RK45",
        t_eval=times,
        rtol=1e-6,
        atol=1e-9,
    )

    x_values = sol.y[0]  # Position
    v_values = sol.y[1]  # Velocity
    solution = np.vstack((x_values, v_values)).T  # [time_steps, 2]

    x_coords = np.zeros_like(times)

    sequences = []
    for i in range(len(times) - 1):
        for seq_len in range(1, min(max_seq_len + 1, i + 1)):
            if i >= seq_len:
                input_seq = []
                for j in range(seq_len):
                    idx = i - seq_len + j
                    # input: [time, x_coord, x(t), v(t)]
                    input_seq.append(
                        [times[idx], x_coords[idx], x_values[idx], v_values[idx]]
                    )

                # next state vector: [x(t), v(t)]
                target = [x_values[i], v_values[i]]

                sequences.append((input_seq, target))

    return sequences, times, solution


class VanDerPolDataset(Dataset):
    """
    Custom PyTorch Dataset for Van der Pol sequence data.
    """

    def __init__(self, sequences: List[Tuple[List[List[float]], List[float]]]) -> None:
        """
        Initialises the dataset.

        Args:
            sequences: List of (input_sequence, target_value) pairs where each input sequence
                      is a list of [time, x_coord, x(t), v(t)] entries and target is [x(t+1), v(t+1)].
        """
        self.sequences = sequences

    def __len__(self) -> int:
        """
        Get the number of sequences in the dataset.

        Returns:
            The total number of sequences.
        """
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[List[List[float]], List[float]]:
        """
        Get a specific sequence-target pair.

        Args:
            idx: Index of the sequence to retrieve.

        Returns:
            A tuple containing (input_sequence, target).
        """
        input_seq, target = self.sequences[idx]
        # Return raw sequences, will be processed in collate_fn
        return input_seq, target


# Transformer Encoder Model for IAG (modified for vector output)
class IAGTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialise the IAGTransformer model.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output vector.
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

        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the IAGTransformer.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
            mask: Optional padding mask where true values don't attend, and false values attend.
                Has shape: [batch_size, seq_len].

        Returns:
            Output tensor of shape [batch_size, output_dim] with predictions.
        """
        # Project input to hidden dimension
        x = self.input_projection(x)

        seq_len = x.size(1)
        x = x + self.positional_encoder[:seq_len, :]

        # Pass through transformer encoder
        # In PyTorch's attention mask: True = don't attend, False = attend
        # Add warning filter to suppres the nested tensor warning
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

            batch_indices = torch.arange(batch_size, device=x.device)
            features = encoded[batch_indices, seq_lengths]
        else:
            features = encoded[:, -1]

        output = self.output_projection(features)

        return output


# Custom collate function for padding sequences (modified for vector targets)
def collate_fn(
    batch: List[Tuple[List[List[float]], List[float]]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for padding sequences of different lengths.

    Args:
        batch: List of (sequence, target) pairs.

    Returns:
        A tuple containing:
            - padded_seqs: Tensor of padded input sequences [batch_size, max_len, input_dim].
            - targets: Tensor of target vectors [batch_size, output_dim].
            - padding_mask: Boolean mask for padding positions [batch_size, max_len].
    """
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, targets = zip(*batch)

    sequences = [torch.tensor(seq, dtype=torch.float32) for seq in sequences]
    targets = torch.tensor(targets, dtype=torch.float32)  # Shape: [batch_size, 2]

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

    # Progress bar
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

            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            progress.update(
                epoch_task,
                description=f"[green]Training progress: (Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6e})",
            )

            progress.remove_task(batch_task)

            if early_stop_counter >= patience:
                console.print(
                    f"[bold yellow]Early stopping triggered after {epoch+1} epochs"
                )
                model.load_state_dict(best_model_state)
                break

    console.print("[bold green]Training completed")
    console.print("[bold]Final Training Loss:", f"{train_losses[-1]:.6f}")
    console.print("[bold]Final Validation Loss:", f"{val_losses[-1]:.6f}")
    console.print("[bold]Best Validation Loss:", f"{best_val_loss:.6f}")

    return train_losses, val_losses, best_val_loss


def predict_trajectory(
    model: nn.Module, t_values: np.ndarray, initial_state: List[float] = [0.5, 0]
) -> np.ndarray:
    """
    Predict the Van der Pol oscillator trajectory using the trained model.

    This function uses the trained model to generate a sequence of predicted values starting from
    an initial state. The model is auto-regressive with each prediction feeding back as the
    input for the next prediction.

    Args:
        model: The trained IAGTransformer model.
        t_values: Array of time points.
        initial_state: Initial state [x(0), v(0)].

    Returns:
        An array of predicted state vectors corresponding to each time point in t_values.
    """
    model.eval()
    predictions = [initial_state]

    current_sequence = [
        [0.0, 0.0, initial_state[0], initial_state[1]]
    ]  # [t, x_coords, x(t), v(t)]

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

                next_state = model(input_tensor, mask).squeeze().tolist()
                if not isinstance(next_state, list):  # Handle scalar case
                    next_state = [next_state, 0.0]

                predictions.append(next_state)

                current_sequence.append([t, 0.0, next_state[0], next_state[1]])

                if len(current_sequence) > max_seq_len:
                    current_sequence = current_sequence[-max_seq_len:]

                progress.update(prediction_task, advance=1)

    return np.array(predictions)


def plot_phase_space(true_solution: np.ndarray, predicted_solution: np.ndarray) -> None:
    """
    Plot the phase space diagram comparing the true and predicted solutions.

    Args:
        true_solution: Array of true state vectors [x(t), v(t)].
        predicted_solution: Array of predicted state vectors [x(t), v(t)].
    """
    plt.figure(figsize=(10, 8))
    plt.plot(true_solution[:, 0], true_solution[:, 1], "b-", label="True Solution")
    plt.plot(
        predicted_solution[:, 0],
        predicted_solution[:, 1],
        "r--",
        label="IAG Predictions",
    )
    plt.xlabel("Position (x)")
    plt.ylabel("Velocity (v)")
    plt.title("Van der Pol Oscillator - Phase Space")
    plt.legend()
    plt.grid(True)
    plt.savefig("../Data/Images/van_der_pol_phase_space.png")
    plt.show()


def main() -> None:
    """
    Main execution.

    Orchestrates the entire workflow:
        1. Generate training data for the Van der Pol oscillator
        2. Create and train the IAGTransformer model
        3. Generate predictions and visualise results
    """
    console = Console()
    console.print(
        "[bold magenta]Van der Pol Oscillator with Implicit Attention Guidance[/]"
    )

    # Model parameters
    mu = 1.0  # Standard van der Pol parameter
    initial_state = [0.5, 0]  # Initial x(0) and v(0)
    output_dim = 2  # x and v components

    with console.status("[bold green]Generating training data..."):
        sequences, times, true_solution = generate_van_der_pol_data(
            t_range=(0, 10), dt=0.05, mu=mu, initial_conditions=initial_state
        )

    console.print(f"[bold green][/]Generated {len(sequences)} training sequences")

    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    console.print(f"[bold]Training set:[/] {len(train_sequences)} sequences")
    console.print(f"[bold]Validation set:[/] {len(val_sequences)} sequences")

    train_dataset = VanDerPolDataset(train_sequences)
    val_dataset = VanDerPolDataset(val_sequences)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Print sample batch
    for inputs, targets, mask in train_loader:
        console.print("[bold]Batch information:[/]")
        console.print(f"Inputs shape: {inputs.shape}")
        console.print(f"Targets shape: {targets.shape}")
        console.print(f"Mask shape: {mask.shape}")

        # Sample first sequence
        sample_table = Table(title="First 3 entries of first sequence")
        sample_table.add_column("Time")
        sample_table.add_column("X Coord")
        sample_table.add_column("Position (x)")
        sample_table.add_column("Velocity (v)")

        for i in range(min(3, inputs.shape[1])):
            sample_table.add_row(
                f"{inputs[0][i][0]:.4f}",
                f"{inputs[0][i][1]:.4f}",
                f"{inputs[0][i][2]:.4f}",
                f"{inputs[0][i][3]:.4f}",
            )

        console.print(sample_table)
        console.print(
            f"Sample target: Position={targets[0][0]:.4f}, Velocity={targets[0][1]:.4f}"
        )
        break

    model = IAGTransformer(
        input_dim, output_dim, hidden_dim, num_layers, num_heads, dropout
    )
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

    predicted_solution = predict_trajectory(model, times, initial_state)

    console.print("\n[bold green]Creating visualizations...[/]")

    # Plot time series for position and velocity
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Position plot
    ax1.plot(times, true_solution[:, 0], "b-", label="True Position")
    ax1.plot(times, predicted_solution[:, 0], "r--", label="Predicted Position")
    ax1.set_xlabel("Time (t)")
    ax1.set_ylabel("Position (x)")
    ax1.set_title("Van der Pol Oscillator - Position")
    ax1.legend()
    ax1.grid(True)

    # Velocity plot
    ax2.plot(times, true_solution[:, 1], "b-", label="True Velocity")
    ax2.plot(times, predicted_solution[:, 1], "r--", label="Predicted Velocity")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("Velocity (v)")
    ax2.set_title("Van der Pol Oscillator - Velocity")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("../Data/Images/van_der_pol_time_series.png")
    plt.show()

    # Plot phase space
    plot_phase_space(true_solution, predicted_solution)

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
    plt.savefig("../Data/Images/van_der_pol_training_validation_loss.png")
    plt.show()

    console.print(
        "\n[bold green]Done. Saved plots as van_der_pol_phase_space.png, "
        "van_der_pol_time_series.png, and van_der_pol_training_validation_loss.png[/]"
    )


if __name__ == "__main__":
    main()

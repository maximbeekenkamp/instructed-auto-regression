import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.animation import FuncAnimation
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
grid_size = 16
input_dim = 5  # (t, x_coord, y_coord, u_velocity, v_velocity)
hidden_dim = 64
num_layers = 2
num_heads = 4
dropout = 0.1
learning_rate = 0.001
batch_size = 16
num_epochs = 20
max_seq_len = 6
patience = 10


def compute_vortex(
    x: np.ndarray, y: np.ndarray, x0: float, y0: float, strength: float, radius: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute velocity field for a single regularised (Lamb-Oseen) vortex.

    Computes the 2D velocity field (u,v) at each point on a grid, resulting from a vortex centered
    at (x0,y0) with a given circulation strength and core radius. It uses the Lamb-Oseen model to
    avoid singularities at the vortex centre. The velocity magnitude decreases with distance from
    the vortex centre and is regularised near the core using the factor: (1 - exp(-r^2 / radius^2))
    which limits the velocity at small radii while preserving the expected 1/r decay at large
    distances.

    Args:
        x, y: Grid coordinates
        x0, y0: Center of vortex
        strength: Vortex strength (positive for counterclockwise)
        radius: Vortex core radius

    Returns:
        u, v: Velocity components
    """
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    r2 = np.maximum(r2, 1e-6)

    factor = strength / (2 * np.pi) * (1 - np.exp(-r2 / (radius**2)))

    u = -factor * (y - y0) / r2
    v = factor * (x - x0) / r2

    return u, v


def generate_vortex_street_data(
    nx: int = grid_size,
    ny: int = grid_size,
    domain_size: float = 10.0,
    num_frames: int = 50,
    dt: float = 0.1,
    u_freestream: float = 1.0,
) -> Tuple[
    List[Tuple[List[List[float]], List[float]]], np.ndarray, np.ndarray, np.ndarray
]:
    """
    Generate data for a vortex street behind a cylinder. The resulting velocity fields are used to
    create temporal input-output sequences for training.

    Args:
        nx, ny: Grid dimensions
        domain_size: Physical size of the domain (from 0 to domain_size along x, and symmetric
                     about 0 in y)
        num_frames: Number of time steps to simulate
        dt: Time step size
        u_freestream: Free stream velocity (along x-axis)

    Returns:
        sequences: List of (input_sequence, target) pairs for training
        x, y: Grid coordinates
        velocity_field: Time series of the velocity field [num_frames, 2, ny, nx]
    """
    console = Console()
    console.print("[bold green]Generating vortex street data...[/]")

    x = np.linspace(0, domain_size, nx)
    y = np.linspace(-domain_size / 2, domain_size / 2, ny)
    X, Y = np.meshgrid(x, y)

    # Parameters for vortex shedding
    cylinder_x = domain_size * 0.2
    cylinder_y = 0.0
    cylinder_radius = domain_size * 0.05
    cylinder_mask = (X - cylinder_x) ** 2 + (Y - cylinder_y) ** 2 < cylinder_radius**2

    vortex_strength = 1.0
    vortex_radius = domain_size * 0.03
    shedding_period = 20 * dt  # Time between vortex shedding events

    velocity_field = np.zeros((num_frames, 2, ny, nx))
    u = np.empty((ny, nx), dtype=np.float32)
    v = np.empty((ny, nx), dtype=np.float32)

    vortices = []  # Each entry: [x, y, strength, radius]

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Generating flow field...", total=num_frames)

        for t in range(num_frames):
            # Base flow (free stream)
            u.fill(u_freestream)
            v.fill(0.0)

            # Add cylinder (just a region of zero velocity)
            u[cylinder_mask] = 0
            v[cylinder_mask] = 0

            # Shed new vortices periodically
            if t > 0 and t % int(shedding_period / dt) == 0:
                offset = 1.5 * cylinder_radius
                # Upper vortex (negative strength - clockwise)
                # Lower vortex (positive strength - counterclockwise)
                vortices.extend(
                    [
                        [
                            cylinder_x + offset,
                            cylinder_y + cylinder_radius,
                            -vortex_strength,
                            vortex_radius,
                        ],
                        [
                            cylinder_x + offset,
                            cylinder_y - cylinder_radius,
                            vortex_strength,
                            vortex_radius,
                        ],
                    ]
                )

            # Update vortex positions
            if vortices:
                vortices_np = np.array(vortices)
                vortices_np[:, 0] += u_freestream * dt * 0.8
                vortices = vortices_np[vortices_np[:, 0] < domain_size].tolist()

            for vortex_x, vortex_y, strength, radius in vortices:
                du, dv = compute_vortex(X, Y, vortex_x, vortex_y, strength, radius)
                u += du
                v += dv

            velocity_field[t, 0] = u
            velocity_field[t, 1] = v

            progress.update(task, advance=1)

    console.print(
        f"[bold green]Generated velocity field with {len(vortices)} vortices in the final frame[/]"
    )

    # Build sequence-target pairs
    sequences = []

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task(
            "[cyan]Creating training sequences...", total=(num_frames - 1) * nx * ny
        )

        for t in range(num_frames - 1):
            for i in range(ny):
                for j in range(nx):
                    # Create sequences of different lengths
                    for seq_len in range(1, min(max_seq_len + 1, t + 1)):
                        input_seq = [
                            [
                                (t - seq_len + k) * dt,
                                x[j],
                                y[i],
                                velocity_field[t - seq_len + k, 0, i, j],
                                velocity_field[t - seq_len + k, 1, i, j],
                            ]
                            for k in range(seq_len)
                        ]
                        target = [
                            velocity_field[t, 0, i, j],
                            velocity_field[t, 1, i, j],
                        ]
                        sequences.append((input_seq, target))
                    progress.update(task, advance=1)

    console.print(f"[bold green]Generated {len(sequences)} training sequences[/]")

    return sequences, x, y, velocity_field


class VortexStreetDataset(Dataset):
    """
    Custom PyTorch Dataset for Vortex Street sequence data.
    """

    def __init__(self, sequences: List[Tuple[List[List[float]], List[float]]]) -> None:
        """
        Initialises the dataset.

        Args:
            sequences: List of (input_sequence, target_value) pairs where each input sequence
                      is a list of [time, x_coord, y_coord, u, v] entries and target is [u, v].
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


# Transformer Encoder Model for IAG
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

        seq_len = min(x.size(1), self.positional_encoder.size(0))
        x = x + self.positional_encoder[:seq_len, :].unsqueeze(
            0
        )  # this is to prevent longer sequences to pass through prepocessing

        # Pass through transformer encoder
        # In PyTorch's attention mask: True = don't attend, False = attend
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
            valid_positions = ~mask
            seq_lengths = (
                valid_positions.sum(dim=1).long() - 1
            )  # -1 to get 0-based index

            batch_indices = torch.arange(batch_size, device=x.device)
            features = encoded[batch_indices, seq_lengths]
        else:
            features = encoded[:, -1]

        output = self.output_projection(features)

        return output


# Custom collate function for padding sequences
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
                optimizer.zero_grad(set_to_none=True)

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
                break

    # Load the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    console.print("[bold green]Training completed")
    console.print("[bold]Final Training Loss:", f"{train_losses[-1]:.6f}")
    console.print("[bold]Final Validation Loss:", f"{val_losses[-1]:.6f}")
    console.print("[bold]Best Validation Loss:", f"{best_val_loss:.6f}")

    return train_losses, val_losses, best_val_loss


def predict_flow_field(
    model: nn.Module,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    initial_velocity_field: np.ndarray,
    num_future_steps: int = 10,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Predict future velocity fields using the trained model.

    Args:
        model: The trained IAGTransformer model.
        x_grid, y_grid: Grid coordinates.
        initial_velocity_field: Initial velocity field [T, 2, ny, nx] where T is the number of initial time steps.
        num_future_steps: Number of future time steps to predict.
        dt: Time step size.

    Returns:
        Predicted velocity field for future time steps [num_future_steps, 2, ny, nx].
    """
    model.eval()

    # Get dimensions
    num_initial_steps = initial_velocity_field.shape[0]
    ny, nx = len(y_grid), len(x_grid)

    # Storage for predictions
    predicted_fields = np.zeros((num_future_steps, 2, ny, nx))

    console = Console()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        time_task = progress.add_task(
            "[green]Predicting future states:", total=num_future_steps
        )

        with torch.no_grad():
            for future_step in range(num_future_steps):
                # Create progress bar for grid points
                grid_task = progress.add_task(
                    f"[cyan]Processing time step {future_step+1}/{num_future_steps}:",
                    total=ny * nx,
                )

                # Predict each grid point independently
                for i in range(ny):
                    for j in range(nx):
                        # Create input sequence for this point
                        input_seq = []

                        # Use the available history (limited to max_seq_len)
                        history_len = min(max_seq_len, num_initial_steps + future_step)

                        for t in range(history_len):
                            time_idx = num_initial_steps + future_step - history_len + t

                            # If time index is within the initial data
                            if time_idx < num_initial_steps:
                                u = initial_velocity_field[time_idx, 0, i, j]
                                v = initial_velocity_field[time_idx, 1, i, j]
                            else:
                                # Otherwise use our predicted data
                                pred_idx = time_idx - num_initial_steps
                                u = predicted_fields[pred_idx, 0, i, j]
                                v = predicted_fields[pred_idx, 1, i, j]

                            # Create sequence element [time, x, y, u, v]
                            input_seq.append(
                                [(time_idx) * dt, x_grid[j], y_grid[i], u, v]
                            )

                        # Convert to tensor and predict
                        input_tensor = torch.tensor([input_seq], dtype=torch.float32)
                        prediction = model(input_tensor).squeeze().numpy()

                        # Store prediction
                        predicted_fields[future_step, 0, i, j] = prediction[0]  # u
                        predicted_fields[future_step, 1, i, j] = prediction[1]  # v

                        progress.update(grid_task, advance=1)

                progress.remove_task(grid_task)
                progress.update(time_task, advance=1)

    return predicted_fields


def calculate_vorticity(
    u: np.ndarray, v: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """
    Calculate vorticity from velocity field.

    Args:
        u, v: Velocity components
        dx, dy: Grid spacing

    Returns:
        vorticity: Vorticity field
    """
    # Calculate vorticity: omega = dv/dx - du/dy
    dudy = np.gradient(u, axis=0) / dy
    dvdx = np.gradient(v, axis=1) / dx

    vorticity = dvdx - dudy
    return vorticity


def visualize_flow_field(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Visualise the flow field with streamlines and vorticity contours.

    Args:
        x, y: Grid coordinates
        u, v: Velocity components
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Calculate velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)

    # Calculate vorticity
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    vorticity = calculate_vorticity(u, v, dx, dy)

    # Plot streamlines and velocity magnitude
    strm = ax1.streamplot(
        X, Y, u, v, density=1.5, color=vel_mag, linewidth=1.5, cmap="viridis"
    )
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title(f"Streamlines - {title}")
    ax1.set_aspect("equal")
    fig.colorbar(strm.lines, ax=ax1, label="Velocity magnitude")

    # Plot vorticity contours
    vort_plot = ax2.contourf(X, Y, vorticity, levels=20, cmap="RdBu_r", extend="both")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title(f"Vorticity - {title}")
    ax2.set_aspect("equal")
    fig.colorbar(vort_plot, ax=ax2, label="Vorticity")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_flow_animation(
    x: np.ndarray,
    y: np.ndarray,
    velocity_fields: np.ndarray,
    save_path: str,
    fps: int = 10,
) -> None:
    """
    Create an animation of the flow field evolution.

    Args:
        x, y: Grid coordinates
        velocity_fields: Velocity fields at different time steps [T, 2, ny, nx]
        save_path: Path to save the animation
        fps: Frames per second
    """
    # Create meshgrid
    X, Y = np.meshgrid(x, y)

    # Calculate vorticity for each time step
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    vorticities = np.zeros((velocity_fields.shape[0], X.shape[0], X.shape[1]))

    for t in range(velocity_fields.shape[0]):
        vorticities[t] = calculate_vorticity(
            velocity_fields[t, 0], velocity_fields[t, 1], dx, dy
        )

    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Initial plots
    u_init = velocity_fields[0, 0]
    v_init = velocity_fields[0, 1]
    vel_mag_init = np.sqrt(u_init**2 + v_init**2)

    strm = ax1.streamplot(
        X,
        Y,
        u_init,
        v_init,
        density=1.5,
        color=vel_mag_init,
        linewidth=1.5,
        cmap="viridis",
    )

    # Get velocity magnitude range for consistent colorbar
    vel_max = np.max(
        [
            np.sqrt(velocity_fields[t, 0] ** 2 + velocity_fields[t, 1] ** 2).max()
            for t in range(velocity_fields.shape[0])
        ]
    )

    # Get vorticity range for consistent colorbar
    vort_max = max(abs(vorticities.min()), abs(vorticities.max()))
    vort_levels = np.linspace(-vort_max, vort_max, 20)

    vort_plot = ax2.contourf(
        X, Y, vorticities[0], levels=vort_levels, cmap="RdBu_r", extend="both"
    )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_title("Streamlines")
    ax1.set_aspect("equal")

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Vorticity")
    ax2.set_aspect("equal")

    cbar1 = fig.colorbar(strm.lines, ax=ax1, label="Velocity magnitude")
    cbar2 = fig.colorbar(vort_plot, ax=ax2, label="Vorticity")

    plt.tight_layout()

    # Function to update the animation
    def update(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()

        # Get current velocity components
        u = velocity_fields[frame, 0]
        v = velocity_fields[frame, 1]
        vel_mag = np.sqrt(u**2 + v**2)

        # Update streamplot
        strm = ax1.streamplot(
            X, Y, u, v, density=1.5, color=vel_mag, linewidth=1.5, cmap="viridis"
        )

        # Update vorticity plot
        vort_plot = ax2.contourf(
            X, Y, vorticities[frame], levels=vort_levels, cmap="RdBu_r", extend="both"
        )

        # Set labels and titles
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title(f"Streamlines - Frame {frame}")
        ax1.set_aspect("equal")

        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_title(f"Vorticity - Frame {frame}")
        ax2.set_aspect("equal")

        return strm.lines, vort_plot

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=velocity_fields.shape[0],
        interval=1000 / fps,  # in milliseconds
        blit=False,
    )

    # Save animation
    anim.save(save_path, writer="pillow", fps=fps)

    plt.close(fig)

    print(f"Animation saved to {save_path}")


def main() -> None:
    """
    Main execution function for the vortex street IAG model.

    Orchestrates the entire workflow:
    1. Generate training data (vortex street flow fields)
    2. Create and train the IAGTransformer model
    3. Generate predictions and visualise results
    """
    console = Console()
    console.print(
        "[bold magenta]Vortex Street Flow Simulation with Implicit Attention Guidance[/]"
    )

    # Generate data
    with console.status("[bold green]Generating vortex street training data..."):
        sequences, x_grid, y_grid, velocity_field = generate_vortex_street_data()

    console.print(f"[bold green]Generated {len(sequences)} training sequences")

    # Split into training and validation sets
    train_size = int(0.8 * len(sequences))
    train_sequences = sequences[:train_size]
    val_sequences = sequences[train_size:]

    console.print(f"[bold]Training set:[/] {len(train_sequences)} sequences")
    console.print(f"[bold]Validation set:[/] {len(val_sequences)} sequences")

    # Create datasets
    train_dataset = VortexStreetDataset(train_sequences)
    val_dataset = VortexStreetDataset(val_sequences)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Print sample batch information
    for inputs, targets, mask in train_loader:
        console.print("[bold]Batch information:[/]")
        console.print(f"Inputs shape: {inputs.shape}")
        console.print(f"Targets shape: {targets.shape}")
        console.print(f"Mask shape: {mask.shape}")

        # Sample first sequence
        sample_table = Table(title="First 3 entries of first sequence")
        sample_table.add_column("Time")
        sample_table.add_column("X Coord")
        sample_table.add_column("Y Coord")
        sample_table.add_column("U Velocity")
        sample_table.add_column("V Velocity")

        for i in range(min(3, inputs.shape[1])):
            sample_table.add_row(
                f"{inputs[0][i][0]:.4f}",
                f"{inputs[0][i][1]:.4f}",
                f"{inputs[0][i][2]:.4f}",
                f"{inputs[0][i][3]:.4f}",
                f"{inputs[0][i][4]:.4f}",
            )

        console.print(sample_table)
        console.print(f"Sample target: U={targets[0][0]:.4f}, V={targets[0][1]:.4f}")
        break

    # Create the model
    model = IAGTransformer(input_dim, 2, hidden_dim, num_layers, num_heads, dropout)
    model_params = sum(p.numel() for p in model.parameters())
    console.print(f"[bold]Model parameters:[/] {model_params:,}")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    console.print("\n[bold yellow]Starting Training...[/]")

    # Train the model
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

    # Use a subset of the initial velocity field for prediction
    initial_frames = 10
    future_frames = 20
    initial_field = velocity_field[:initial_frames]

    # Predict future flow fields
    predicted_fields = predict_flow_field(
        model, x_grid, y_grid, initial_field, num_future_steps=future_frames
    )

    console.print("\n[bold green]Creating visualisations...[/]")

    # Plot initial and final frames for comparison
    visualize_flow_field(
        x_grid,
        y_grid,
        initial_field[-1, 0],
        initial_field[-1, 1],
        "Last Initial Frame",
        save_path="../Data/Images/Vortex_Street/vortex_initial_frame.png",
    )

    visualize_flow_field(
        x_grid,
        y_grid,
        predicted_fields[-1, 0],
        predicted_fields[-1, 1],
        "Final Predicted Frame",
        save_path="../Data/Images/Vortex_Street/vortex_final_prediction.png",
    )

    # Create animations
    create_flow_animation(
        x_grid,
        y_grid,
        initial_field,
        save_path="../Data/Images/Vortex_Street/vortex_initial_animation.gif",
    )

    create_flow_animation(
        x_grid,
        y_grid,
        predicted_fields,
        save_path="../Data/Images/Vortex_Street/vortex_prediction_animation.gif",
    )

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
    plt.savefig("../Data/Images/Vortex_Street/vortex_training_validation_loss.png")
    plt.show()

    console.print(
        "\n[bold green]Done. Visualisation images and animations have been saved.[/]"
    )


if __name__ == "__main__":
    main()

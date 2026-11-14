"""Introductory MLP: fit sin(2*pi*x) with PyTorch on CPU."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(7)
np.random.seed(7)

# Training and validation inputs are distinct.
x_train = torch.linspace(0, 1, 101).unsqueeze(1)
x_val = torch.linspace(0.005, 0.995, 100).unsqueeze(1)
y_train = torch.sin(2 * torch.pi * x_train)
y_val = torch.sin(2 * torch.pi * x_val)
loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

# One input, one output, and two small hidden layers.
model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32),
                      nn.Tanh(), nn.Linear(32, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()
train_loss, val_loss = [], []

# Training
for epoch in range(200):
    model.train()
    for xb, yb in loader:
        prediction = model(xb)           # Forward: predict y from x.
        loss = loss_fn(prediction, yb)   # Loss: compare to sin(2*pi*x).
        optimizer.zero_grad()           # Clear gradients from previous batch.
        loss.backward()                 # Backward: compute gradients.
        optimizer.step()                # Update the MLP parameters.
    # Validation
    model.eval()
    with torch.no_grad():
        train_loss.append(loss_fn(model(x_train), y_train).item())
        val_loss.append(loss_fn(model(x_val), y_val).item())
    if (epoch + 1) % 40 == 0:
        print(f"Epoch {epoch+1:3d}: train={train_loss[-1]:.6f}, val={val_loss[-1]:.6f}")

# Inference
model.eval()
with torch.no_grad():
    prediction = model(x_val).numpy().ravel()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
axes[0].semilogy(range(1, 201), train_loss, label="Training")
axes[0].semilogy(range(1, 201), val_loss, label="Validation")
axes[0].set(xlabel="Epoch", ylabel="Mean squared error", title="Training history")
axes[0].legend()
axes[1].plot(x_val.numpy().ravel(), y_val.numpy().ravel(), label="Exact sine")
axes[1].plot(x_val.numpy().ravel(), prediction, "--", label="MLP prediction")
axes[1].set(xlabel="x", ylabel="y", title="Function approximation")
axes[1].legend()
fig.tight_layout()
output = Path(__file__).with_name("mlp_sine_results.png") if "__file__" in globals() else Path("mlp_sine_results.png")
fig.savefig(output, dpi=160)
print(f"Saved {output}")
plt.show()

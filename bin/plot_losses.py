import matplotlib.pyplot as plt
import h5py

# Load losses
with h5py.File('../results/train_losses.h5') as f:
    losses = f['train_losses']

# Save plot
plt.ioff()
plt.plot(losses)
plt.savefig('../results/train_losses.png')

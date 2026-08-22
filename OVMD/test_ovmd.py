import numpy as np
import matplotlib.pyplot as plt

from ovmd import OVMD


# ----------------------
# generate signal
# ----------------------

t = np.linspace(0, 1, 2000)

signal = (
    np.sin(2*np.pi*10*t)
    +
    0.5*np.sin(2*np.pi*40*t)
)



# ----------------------
# OVMD
# ----------------------

result = OVMD(
    signal,
    alpha=2000,
    K=3
)


modes = result["u"]

print("modes shape:", modes.shape)

print("omega:")
print(result["omega"])


print("modes shape:", modes.shape)



# ----------------------
# plot
# ----------------------

plt.figure(figsize=(10,8))


plt.subplot(4,1,1)

plt.plot(t, signal)

plt.title("Original Signal")



for i in range(3):

    plt.subplot(4,1,i+2)

    plt.plot(
        t,
        modes[i,:]
    )

    plt.title(
        f"Mode {i+1}"
    )


plt.tight_layout()

plt.savefig(
    "ovmd_result.png",
    dpi=300
)

plt.close()
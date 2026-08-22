import numpy as np
from RVMD.rvmd import rvmd


S = 50
T = 200


t = np.arange(T)


Q = np.zeros(
    (S,T)
)


for i in range(S):

    Q[i,:] = np.sin(
        2*np.pi*0.05*t
    )



mode,info,restart = rvmd(
    Q,
    K=1,
    Alpha=2000,
    MaximumSteps=500,
    Display="iter"
)


print(mode["phi"].shape)

print(mode["c"].shape)

print(mode["omega"])
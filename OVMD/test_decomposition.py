import numpy as np

from ovmd_decomposition import OVMDDecomposition


t = np.linspace(0,1,2000)


X = (
    np.sin(2*np.pi*10*t)
    +
    0.5*np.sin(2*np.pi*40*t)
)


noise = 0.2*np.random.randn(len(t))


Xn = X + noise



out, outn = OVMDDecomposition(
    X,
    Xn,
    t,
    twoF=2000,
    tau=0,
    K=3,
    tol=1e-5,
    N=1000
)


print(out.keys())

print(outn.keys())

print("clean error:",
      out["ErrL2"])

print("noise error:",
      outn["ErrL2"])
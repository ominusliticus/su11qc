# $SU(1, 1)$ Quantum Computing Simulations

This code solves the system

$$
    \begin{pmatrix}
        \langle \dot K_0 \rangle \\
        \langle \dot K_1 \rangle \\
        \langle \dot K_2 \rangle
    \end{pmatrix}
    =
    \begin{pmatrix}
        0 & 0 & H_1 \\
        0 & 0 & -H_0 \\
        H_1 & H_0 & 0 \\
    \end{pmatrix}
    \begin{pmatrix}
        \langle K_0 \rangle \\
        \langle K_1 \rangle \\
        \langle K_2 \rangle
    \end{pmatrix}
$$

Where we have

$$
    H_0 = E_k + \frac{g_1}{g_0} \cos{\omega t}, \quad \text{and} \quad H_1 = 1 + \frac{g_1}{g_0} \cos{\omega t}
$$

## Dependencies
- Python $\geq$ 3.8
- numpy
- scipy
- matplotlib
- icecream


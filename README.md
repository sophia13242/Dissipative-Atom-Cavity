# Cavity QED simulations

Simulations of single- and multi-atom systems coupled to a cavity using **QuTiP**, exploring coherence, decoherence, and spin-echo phenomena.

---

## Overview

- Atoms modeled as **two-level systems** interacting with a cavity.
- Hamiltonian includes atom-cavity coupling (`g`) and atomic frequency (`w0`).
- Optional dissipation: atomic decay (`γ`), cavity decay (`κ`), and dephasing (`β`).
- Quasi-static noise modeled as **random z-rotations** on the atomic Bloch vector.

---

## Physics

- **Coherent evolution**: The system evolves under the **Jaynes-Cummings Hamiltonian**  

$$
H = \hbar \omega_0 a^\dagger a + \frac{\hbar \omega_0}{2} \sigma_z + \hbar g (a \sigma_+ + a^\dagger \sigma_-)
$$  

for a single atom coupled to a cavity mode. The Bloch vector of the atom rotates around the z-axis in the x-y plane, preserving its length, reflecting coherent evolution.

- **Lindblad dephasing**: The effect of decoherence is included via the **Lindblad master equation**  

$$
\dot{\rho} = -\frac{i}{\hbar}[H, \rho] + \sum_j \left( c_j \rho c_j^\dagger - \frac{1}{2} \{ c_j^\dagger c_j, \rho \} \right)
$$  

with collapse operators \(c_j\) for cavity leakage, atomic decay, and dephasing. Coherence in the x-y plane shrinks over time due to dephasing.

- **Quasi-static noise & spin echo**: Quasi-static dephasing is modeled as a random but constant detuning  

$$
H_\text{noise} = H + \frac{\delta \phi}{2} \sigma_z
$$  

for each realization. A π-pulse at time \(t_\text{echo}\) inverts the Bloch vector, partially cancelling the accumulated phase and restoring coherence.

- **Multi-atom extension**: For multiple atoms, the **Tavis-Cummings Hamiltonian**  

$$
H = \hbar \omega_0 a^\dagger a + \sum_{i=1}^{N_\text{atoms}} \frac{\hbar \omega_0}{2} \sigma_z^{(i)} + \sum_{i=1}^{N_\text{atoms}} \hbar g (a \sigma_+^{(i)} + a^\dagger \sigma_-^{(i)})
$$  

describes collective dynamics. The system exhibits correlated Bloch vector evolution, which can be visualized for each atom on the Bloch sphere.



---

## Simulations

1. Single atom coherent rotation (`|+⟩` initial state).  
2. Decoherence with Lindblad operators (`β>0`).  
3. Quasi-static noise with spin-echo pulse.  
4. Multiple atoms coupled to the cavity, observing collective dynamics.

---

## Visualization

- Expectation values: ⟨n_atom⟩, ⟨n_cavity⟩, |coherence|.  
- Bloch sphere GIFs showing rotation, decoherence, and echo effects.

---

## Requirements

Python ≥ 3.9, QuTiP, NumPy, Matplotlib, ipywidgets.

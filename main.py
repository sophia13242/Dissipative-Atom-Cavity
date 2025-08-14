import numpy as np
import qutip as qt
from src.system import AtomCavitySystem
import matplotlib.pyplot as plt

def main():
    # Create system
    system = AtomCavitySystem(N=10, g=(0.1 * 2 * np.pi), w0=(1.0 * 2 * np.pi))

    # Build Hamiltonian
    H = system.create_hamiltonian()

    # Initial state |1⟩_cavity ⊗ |e⟩_atom
    psi11 = qt.ket2dm(qt.tensor(qt.basis(system.N, 1), qt.basis(2, 1)))

    # Collapse operators (decay only)
    gamma = 0.05
    sm = system.sm  # atomic lowering operator
    c_ops_dec_only = [np.sqrt(gamma) * sm]

    # Expectation values
    e_ops = system.create_expectation_values(
        number_atom=True,
        number_cavity=True,
        coherence_ge=True,
        coherence_eg=True
    )

    # Time evolution
    g = system.g
    times = np.linspace(0, 10 * 2 * np.pi / g, 1000)
    result = qt.mesolve(H, psi11, times, c_ops_dec_only, e_ops)

    # Plot results
    qt.plot_expectation_values(
        [result],
        ylabels=["<n_cav>", "<n_atom>", "coherence", "coherence"]
    )

    plt.show()


if __name__ == "__main__":
    main()

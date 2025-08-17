import numpy as np
import qutip as qt
import imageio
import os

class AtomCavitySystem:
    def __init__(self, N, g, w0, gamma=0.1, kappa=0.05, beta=0.02):
        self.N = N
        self.g = g
        self.w0 = w0
        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta
        
        # Define operators once
        self.a = qt.tensor(qt.destroy(N), qt.identity(2))
        self.sm = qt.tensor(qt.identity(N), qt.destroy(2))
        self.sz = self.sm.dag() * self.sm - self.sm * self.sm.dag()
        self.coh_ge = qt.tensor(qt.qeye(N), qt.basis(2, 0) * qt.basis(2, 1).dag())
        self.coh_eg = self.coh_ge.dag()

        # Default collapse ops and expectation ops
        self.c_ops = []
        self.e_ops = []

    def create_collapse_operators(self, leaking=False, decay=False, dephasing=False):
        self.c_ops = []
        if leaking:
            self.c_ops.append(np.sqrt(self.gamma) * self.a)
        if decay:
            self.c_ops.append(np.sqrt(self.kappa) * self.sm)
        if dephasing:
            self.c_ops.append(np.sqrt(self.beta) * self.sz)
        return self.c_ops

    def create_expectation_values(self, number_atom=False, number_cavity=False, coherence_ge=False, coherence_eg=False):
        self.e_ops = []
        if number_atom:
            self.e_ops.append(self.sm.dag() * self.sm)
        if number_cavity:
            self.e_ops.append(self.a.dag() * self.a)
        if coherence_ge:
            self.e_ops.append(self.coh_ge)
        if coherence_eg:
            self.e_ops.append(self.coh_eg)
        return self.e_ops

    def create_hamiltonian(self):
        H_cavity = self.w0 * self.a.dag() * self.a
        H_atom = self.w0 * self.sm.dag() * self.sm
        H_int = self.g * (self.a.dag() * self.sm + self.a * self.sm.dag())
        return H_cavity + H_atom + H_int

    def create_initial_state(self, state_cavity, state_atom):
        if state_cavity == '0':
            cavity = qt.basis(self.N, 0)
        elif state_cavity == '1':
            cavity = qt.basis(self.N, 1)
        else:
            raise ValueError("Invalid cavity state. Use '0' or '1'.")

        if state_atom == 'g':
            atom = qt.basis(2, 0)
        elif state_atom == 'e':
            atom = qt.basis(2, 1)
        elif state_atom == '+':
            atom = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        elif state_atom == '-':
            atom = (qt.basis(2, 0) - qt.basis(2, 1)).unit()
        else:
            raise ValueError("Invalid atom state. Use 'g', 'e', '+', or '-'.")

        return qt.ket2dm(qt.tensor(cavity, atom))
    
    def get_reduced_atom_state(self, rho):
        return rho.ptrace(1)
    
    #helper functions for visualization

    def plot_bloch_vector(self, rho):
        rho_atom = self.get_reduced_atom_state(rho)

        # Expectation values of Pauli operators
        sx = qt.sigmax()
        sy = qt.sigmay()
        sz = qt.sigmaz()

        bloch_vector = [
            qt.expect(sx, rho_atom),
            qt.expect(sy, rho_atom),
            qt.expect(sz, rho_atom)
        ]

        # Draw Bloch sphere
        b = qt.Bloch()
        b.add_vectors(bloch_vector)
        b.show()


    def create_bloch_gif(self, result, filename, duration=0.05):
        # Compute Bloch vectors
        bloch_vectors = []
        for rho in result.states:
            rho_atom = self.get_reduced_atom_state(rho)
            vector = [
                qt.expect(qt.sigmax(), rho_atom),
                qt.expect(qt.sigmay(), rho_atom),
                qt.expect(qt.sigmaz(), rho_atom)
            ]
            bloch_vectors.append(vector)

        # Create a fresh Bloch sphere
        b = qt.Bloch()
        b.vector_color = ['r']
        b.point_color = ['b']

        # Save each frame
        frames = []
        for i, vec in enumerate(bloch_vectors):
            b.clear()
            b.add_vectors(vec)
            frame_file = f"temp_frame_{i}.png"
            b.save(frame_file)
            frames.append(frame_file)

        # Convert to GIF
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(filename, images, duration=duration)

        # Clean up
        for f in frames:
            os.remove(f)

        print(f"Animation saved as {filename}")



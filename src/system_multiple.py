import numpy as np
import qutip as qt
import imageio
import os
from PIL import Image

class MultipleAtomsCavitySystem:
    """
    A class representing multiple two-level atoms coupled to a single quantized cavity mode.  
    The model generalizes the Jaynes–Cummings model to multiple atoms, including atom–cavity 
    coupling, photon leakage, atomic decay, and dephasing.  

    Visualization tools allow tracking each atom’s reduced state on separate Bloch spheres 
    and combining them into a single animated GIF.  

    Parameters
    ----------
    N : int
        Dimension of the cavity Hilbert space (photon number cutoff).
    g : float
        Atom–cavity coupling strength.
    w0 : float
        Resonant frequency of the atom(s) and cavity (assumed identical).
    num_atoms : int, optional
        Number of two-level atoms coupled to the cavity. Default is 1.
    gamma : float, optional
        Cavity photon leakage rate. Default is 0.1.
    kappa : float, optional
        Atomic spontaneous emission rate. Default is 0.05.
    beta : float, optional
        Atomic pure dephasing rate. Default is 0.02.
    """

    def __init__(self, N, g, w0, num_atoms=1, gamma=0.1, kappa=0.05, beta=0.02):
        self.N = N
        self.g = g
        self.w0 = w0
        self.num_atoms = num_atoms
        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta

    def atom_operator(self, op, atom_index):
        """
        Construct an operator acting on a specific atom (with identity on all others).

        Parameters
        ----------
        op : Qobj
            Single-qubit operator (e.g., σ-, σz).
        atom_index : int
            Index of the atom (0-based).

        Returns
        -------
        Qobj
            Operator acting on the full Hilbert space.
        """
        op_list = [qt.qeye(self.N)]  # cavity identity
        for j in range(self.num_atoms):
            op_list.append(op if j == atom_index else qt.qeye(2))
        return qt.tensor(op_list)

    def create_operators(self):
        """
        Create the basic operators for the system:
        - Cavity annihilation operator.
        - Atomic lowering operators for each atom.
        - Pauli-Z operators for each atom.
        """
        # cavity annihilation
        self.a = qt.tensor([qt.destroy(self.N)] + [qt.qeye(2) for _ in range(self.num_atoms)])

        # atomic lowering and Pauli-Z operators for each atom
        self.sm_list = [self.atom_operator(qt.destroy(2), i) for i in range(self.num_atoms)]
        self.sz_list = [sm.dag()*sm - sm*sm.dag() for sm in self.sm_list]

    def create_hamiltonian(self):
        """
        Construct the system Hamiltonian under the multi-atom Jaynes–Cummings model.

        Returns
        -------
        Qobj
            Hamiltonian of the form:
            H = w0 * (a†a + Σ σ†σ) + g * Σ (a†σ + a σ†)
        """
        H_cavity = self.w0 * self.a.dag() * self.a
        H_atoms = sum(self.w0 * sm.dag() * sm for sm in self.sm_list)
        H_int = sum(self.g * (self.a.dag() * sm + self.a * sm.dag()) for sm in self.sm_list)
        return H_cavity + H_atoms + H_int

    def create_collapse_operators(self, leaking=False, decay=False, dephasing=False):
        """
        Define collapse (dissipative) operators for the master equation.

        Parameters
        ----------
        leaking : bool
            If True, include cavity photon leakage at rate gamma.
        decay : bool
            If True, include atomic spontaneous emission at rate kappa.
        dephasing : bool
            If True, include atomic pure dephasing at rate beta.

        Returns
        -------
        list of Qobj
            Collapse operators to be passed to `mesolve`.
        """
        c_ops = []
        if leaking:
            c_ops.append(np.sqrt(self.gamma) * self.a)
        if decay:
            for sm in self.sm_list:
                c_ops.append(np.sqrt(self.kappa) * sm)
        if dephasing:
            for sz in self.sz_list:
                c_ops.append(np.sqrt(self.beta) * sz)
        return c_ops

    def create_initial_state(self, state_cavity, atom_states):
        """
        Construct the initial density matrix of the system.

        Parameters
        ----------
        state_cavity : str
            Initial cavity Fock state. Options:
            - '0', '1', '2', '3' : corresponding Fock states.
        atom_states : list of str
            List of atomic states, one per atom. Options:
            - 'g' : ground state |g⟩
            - 'e' : excited state |e⟩
            - '+' : superposition (|g⟩ + |e⟩)/√2
            - '-' : superposition (|g⟩ - |e⟩)/√2

        Returns
        -------
        Qobj
            Density matrix of the initial cavity–atom system.
        """
        # cavity
        if state_cavity == '0':
            cavity = qt.basis(self.N, 0)
        elif state_cavity == '1':
            cavity = qt.basis(self.N, 1)
        elif state_cavity == '2':
            cavity = qt.basis(self.N, 2)
        elif state_cavity == '3':
            cavity = qt.basis(self.N, 3)
        else:
            raise ValueError("Invalid cavity state.")

        # atoms
        atom_kets = []
        for s in atom_states:
            if s == 'g':
                atom_kets.append(qt.basis(2, 0))
            elif s == 'e':
                atom_kets.append(qt.basis(2, 1))
            elif s == '+':
                atom_kets.append((qt.basis(2, 0) + qt.basis(2, 1)).unit())
            elif s == '-':
                atom_kets.append((qt.basis(2, 0) - qt.basis(2, 1)).unit())
            else:
                raise ValueError(f"Invalid atom state {s}.")
        
        return qt.ket2dm(qt.tensor([cavity] + atom_kets))
    
    def get_reduced_atom_state(self, rho, atom_index):
        """
        Compute the reduced density matrix for a specific atom.

        Parameters
        ----------
        rho : Qobj
            Full system density matrix.
        atom_index : int
            Index of the atom (0-based).

        Returns
        -------
        Qobj
            Reduced density matrix of the atom.
        """
        return rho.ptrace(atom_index+1)  # cavity = 0, atoms start at 1

    def create_multi_bloch_gif(self, result, filename):
        """
        Generate an animated GIF showing the evolution of each atom’s state
        on separate Bloch spheres, combined into a single frame per time step.

        Parameters
        ----------
        result : Result
            Output of `mesolve`, containing the time-evolved states.
        filename : str
            Output file name for the GIF.

        Notes
        -----
        - Each atom is plotted on its own Bloch sphere in a horizontal row.
        - Temporary PNG files are created for each Bloch sphere and deleted
          after combination.
        - Frame durations are scaled by simulation time steps.
        """
        times = np.array(result.times)
        dts = np.diff(times)
        dts = np.append(dts, dts[-1]) * 4

        frames = []
        for ti, rho in enumerate(result.states):
            bloch_imgs = []
            for atom in range(self.num_atoms):
                rho_atom = self.get_reduced_atom_state(rho, atom)
                vec = [
                    qt.expect(qt.sigmax(), rho_atom),
                    qt.expect(qt.sigmay(), rho_atom),
                    qt.expect(qt.sigmaz(), rho_atom)
                ]

                b = qt.Bloch()
                b.vector_color = ['r']
                b.point_color = ['b']
                b.add_vectors(vec)
                fname = f"_bloch_{ti}_{atom}.png"
                b.save(fname)
                bloch_imgs.append(Image.open(fname))

            # combine horizontally
            widths, heights = zip(*(im.size for im in bloch_imgs))
            total_width = sum(widths)
            max_height = max(heights)
            combined = Image.new("RGB", (total_width, max_height), (255,255,255))

            x_offset = 0
            for im in bloch_imgs:
                combined.paste(im, (x_offset,0))
                im.close()
                x_offset += im.size[0]

            framefile = f"_frame_{ti}.png"
            combined.save(framefile)
            frames.append(framefile)

            # cleanup atom images
            for atom in range(self.num_atoms):
                os.remove(f"_bloch_{ti}_{atom}.png")

        # assemble gif
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(filename, images, duration=dts.tolist())

        for f in frames:
            os.remove(f)

        print(f"Multi-Bloch animation saved as {filename}")

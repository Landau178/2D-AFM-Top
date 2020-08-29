import pythtb as pytb
import numpy as np
import matplotlib.pyplot as plt
import string as st
from mpl_toolkits.mplot3d import Axes3D
import tb_utilities as tbu
import os



class Simulation_TB():
    def __init__(self, dir):
        [t, lamb, h] = np.loadtxt(dir + 'parameters.dat')
        self.dir = dir
        self.prepare_directory()
        self.model = self.create_CrAs2(t, lamb, h)

    def prepare_directory(self):
        self.make_dir('images')
        self.make_dir('Saved_simulations')
        self.make_dir('regular_grids')

    def make_dir(self, folder):
        try:
            os.makedirs(self.dir + folder + '/')
        except OSError:
            pass

    def create_CrAs2(self, t, lamb, h):
        self.n_orb = 4
        self.n_spin = 2
        self.n_bands = self.n_orb * self.n_spin

        h = np.array([0,0, h])
        sq3 = np.sqrt(3)
        a1 = np.array([1, 0, 0])
        a2 = np.array([-1/2, sq3/2, 0])
        a3 = np.array([0, 0, 2/3])
        l1 = tbu.vector_SU2(0.5 * a1)
        l2 = tbu.vector_SU2(0.5 * a1 + 0.5 * a2)
        l3 = tbu.vector_SU2(0.5 * a2)
        lat = [list(a1), list(a2), list(a3)]
        orb = [[1/3, 1/6, 0], # spin up
            [5/6, 1/6, 0],  # spin down
            [1/3, 2/3, 0],  # spin down
            [5/6, 2/3, 0]] # spin up
        tb_model = pytb.tb_model(2, 3, lat=lat, orb=orb, nspin=2)
        # nearest neighbours hoppings
        t = t * np.identity(2)
        sigma_z = np.array([[1, 0], [0, -1]])
        h = tbu.vector_SU2(h)
        alpha = sq3/24 * lamb
        beta = lamb /36
        tb_model.set_hop(t + 1j*alpha * sigma_z + 1j*beta*l1, 0, 1, [0, 0, 0])
        tb_model.set_hop(t + 1j*alpha * sigma_z + 1j*beta*l3, 0, 2, [0, 0, 0])
        tb_model.set_hop(t - 1j*alpha * sigma_z + 1j*beta*l2, 0, 3, [0, 0, 0])
        tb_model.set_hop(t - 1j*alpha * sigma_z - 1j*beta*l1, 0, 1, [-1, 0, 0])
        tb_model.set_hop(t + 1j*alpha * sigma_z - 1j*beta*l2, 0, 3, [-1, -1, 0])
        tb_model.set_hop(t - 1j*alpha * sigma_z - 1j*beta*l3, 0, 2, [0, -1, 0])

        tb_model.set_hop(t + 1j*alpha * sigma_z + 1j*beta*l3, 1, 3, [0, 0, 0])
        tb_model.set_hop(t - 1j*alpha * sigma_z + 1j*beta*l2, 1, 2, [1, 0, 0])
        tb_model.set_hop(t - 1j*alpha * sigma_z - 1j*beta*l3, 1, 3, [0, -1, 0])
        tb_model.set_hop(t + 1j*alpha * sigma_z - 1j*beta*l2, 1, 2, [0, -1, 0])

        tb_model.set_hop(t - 1j*alpha * sigma_z - 1j*beta*l1, 2, 3, [-1, 0, 0])
        tb_model.set_hop(t + 1j*alpha * sigma_z + 1j*beta*l1, 2, 3, [0, 0, 0])

        tb_model.set_onsite([-h, h, h, -h])
        return tb_model

    def plot_bands_path(self):
        path=[[0.,0.],[1./3.,1./3.],[0.5,0],[2/3.,-1/3.], [0.,0.]]
        # labels of the nodes
        label=(r'$\Gamma $',r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')

        # call function k_path to construct the actual path
        (k_vec,k_dist,k_node) = self.model.k_path(path,301)
        evals = self.model.solve_all(k_vec)
        #Plot of bands
        fig, ax = plt.subplots()
        ax.set_xlim(k_node[0],k_node[-1])
        ax.set_xticks(k_node)
        ax.set_xticklabels(label)
        for n in range(len(k_node)):
          ax.axvline(x=k_node[n],linewidth=0.5, color='k')
        ax.set_xlabel("Path in k-space")
        ax.set_ylabel("Band energy")
        for i in range(self.n_bands):
            ax.plot(k_dist,evals[i])
        fig_name = 'bands_k_path.png'
        dir = self.dir + 'images/' + fig_name
        fig.savefig(dir, dpi=500)
        plt.close()

    def eigen_functions_grid(self, n_step=100):
        w_square=pytb.wf_array(self.model,[n_step,n_step])
        all_kpt=np.zeros((n_step,n_step,2))
        bands = np.zeros((n_step, n_step, self.n_bands))
        all_evec = np.zeros((n_step, n_step, self.n_bands, self.n_orb,
                self.n_spin), dtype='complex')
        for i in range(n_step):
            for j in range(n_step):
                kpt= np.array([i, j]) / n_step
                all_kpt[i,j,:]=kpt
                (eval,evec)=self.model.solve_one(kpt,eig_vectors=True)
                bands[i, j, :] = eval
                w_square[i, j] = evec
                all_evec[i, j, :, :] = evec
        self.k_grid = all_kpt
        self.bands = bands
        self.evecs = all_evec
        self.wf = w_square

    def save_current_grid(self):
        n_step = np.shape(self.k_grid)[0]
        str_step = str(n_step) + 'x' + str(n_step)
        grid_folder = 'regular_grids/' + str_step
        self.make_dir(grid_folder)
        np.save(self.dir + grid_folder + '/k_grid', self.k_grid)
        np.save(self.dir + grid_folder + '/evals', self.bands)
        np.save(self.dir + grid_folder + '/evecs', self.evecs)

    def load_grid(self, n_step):
        str_step = str(n_step) + 'x' + str(n_step)
        grid_folder = 'regular_grids/' + str_step
        w_square=pytb.wf_array(self.model,[n_step,n_step])
        self.k_grid = np.load(self.dir + grid_folder + '/k_grid.npy')
        self.bands = np.load(self.dir + grid_folder + '/evals.npy')
        self.evecs = np.load(self.dir + grid_folder + '/evecs.npy')
        for i in range(n_step):
            for j in range(n_step):
                w_square[i, j] = self.evecs[i, j, :, :, :]
        self.wf = w_square

    def plot_bands_colormap()

import pythtb as pytb
import numpy as np
import matplotlib.pyplot as plt
import string as st
from mpl_toolkits.mplot3d import Axes3D


def create_CrAs2(t_nn, lamb, h):
    sq3 = np.sqrt(3)
    a1 = np.array([1, 0, 0])
    a2 = np.array([-1/2, sq3/2, 0])
    a3 = np.array([0, 0, 2/3])
    l1 = vector_SU2(0.5 * a1)
    l2 = vector_SU2(0.5 * a1 + 0.5 * a2)
    l3 = vector_SU2(0.5 * a2)
    lat = [list(a1), list(a2), list(a3)]
    orb = [[1/3, 1/6, 0], # spin up
        [5/6, 1/6, 0],  # spin down
        [1/3, 2/3, 0],  # spin down
        [5/6, 2/3, 0]] # spin up
    tb_model = pytb.tb_model(2, 3, lat=lat, orb=orb, nspin=2)
    # nearest neighbours hoppings
    t_nn = t_nn * np.identity(2)
    sigma_z = np.array([[1, 0], [0, -1]])
    h = vector_SU2(h)
    alpha = sq3/24 * lamb
    beta = lamb /36
    tb_model.set_hop(t_nn + 1j*alpha * sigma_z + 1j*beta*l1, 0, 1, [0, 0, 0])
    tb_model.set_hop(t_nn + 1j*alpha * sigma_z + 1j*beta*l3, 0, 2, [0, 0, 0])
    tb_model.set_hop(t_nn - 1j*alpha * sigma_z + 1j*beta*l2, 0, 3, [0, 0, 0])
    tb_model.set_hop(t_nn - 1j*alpha * sigma_z - 1j*beta*l1, 0, 1, [-1, 0, 0])
    tb_model.set_hop(t_nn + 1j*alpha * sigma_z - 1j*beta*l2, 0, 3, [-1, -1, 0])
    tb_model.set_hop(t_nn - 1j*alpha * sigma_z - 1j*beta*l3, 0, 2, [0, -1, 0])

    tb_model.set_hop(t_nn + 1j*alpha * sigma_z + 1j*beta*l3, 1, 3, [0, 0, 0])
    tb_model.set_hop(t_nn - 1j*alpha * sigma_z + 1j*beta*l2, 1, 2, [1, 0, 0])
    tb_model.set_hop(t_nn - 1j*alpha * sigma_z - 1j*beta*l3, 1, 3, [0, -1, 0])
    tb_model.set_hop(t_nn + 1j*alpha * sigma_z - 1j*beta*l2, 1, 2, [0, -1, 0])

    tb_model.set_hop(t_nn - 1j*alpha * sigma_z - 1j*beta*l1, 2, 3, [-1, 0, 0])
    tb_model.set_hop(t_nn + 1j*alpha * sigma_z + 1j*beta*l1, 2, 3, [0, 0, 0])

    tb_model.set_onsite([-h, h, h, -h])

    return tb_model

def vector_SU2(vector):
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0,-1]])
    return vector[0]*sigma_x +vector[1]*sigma_y + vector[2]*sigma_z



h = np.array([0, 0, 0.0])
model = create_CrAs2(1, 0.0, h)
# construct two-dimensional square patch covering the Dirac cone
#  parameters of the patch
square_step=401
square_center=np.array([0, 0])
square_length=1
# two-dimensional wf_array to store wavefunctions on the path
w_square= pytb.wf_array(model,[square_step,square_step])
all_kpt=np.zeros((square_step,square_step,2))
# now populate array with wavefunctions
for i in range(square_step):
    for j in range(square_step):
        # construct k-point on the square patch
        kpt=np.array([square_length*(-0.5+float(i)/float(square_step-1)),
                      square_length*(-0.5+float(j)/float(square_step-1))])
        kpt+=square_center
        # store k-points for plotting
        all_kpt[i,j,:]=kpt
        # find eigenvectors at this k-point
        (eval,evec)=model.solve_one(kpt,eig_vectors=True)
        # store eigenvector into wf_array object
        w_square[i,j]=evec

# compute Berry flux on this square patch
print("Berry flux on square patch with length: ",square_length)
print("  centered at k-point: ",square_center)
print("  for band 0 equals    : ", w_square.berry_flux([0]))
print("  for band 1 equals    : ", w_square.berry_flux([1]))
print("  for both bands equals: ", w_square.berry_flux([0,1]))
print()

# also plot Berry phase on each small plaquette of the mesh
plaq=w_square.berry_flux([0],individual_phases=True)
#
fig, ax = plt.subplots()
ax.imshow(plaq.T,origin="lower",
          extent=(all_kpt[0,0,0],all_kpt[-2, 0,0],
                  all_kpt[0,0,1],all_kpt[ 0,-2,1],))
ax.set_title("Berry curvature near Dirac cone")
ax.set_xlabel(r"$k_x$")
ax.set_ylabel(r"$k_y$")
fig.tight_layout()
fig.savefig("cone_phases.pdf")

'''
nx, ny = 50, 50
k_mesh = model.k_uniform_mesh([nx, ny])

evals = model.solve_all(k_mesh)
k_x = np.reshape(k_mesh[:, 0], (nx,ny))
k_y = np.reshape(k_mesh[:, 1], (nx,ny))

evals2d = np.reshape(evals[6, :], (nx, ny))
fig3d = plt.figure()
ax = fig3d.add_subplot(111, projection='3d')
ax.plot_surface(k_x, k_y, evals2d)
plt.show()
'''


'''
# construct circular path around Dirac cone
#   parameters of the path
circ_step=31
circ_center=np.array([-1/3,2/3])
circ_radius=0.5
# one-dimensional wf_array to store wavefunctions on the path
w_circ= pytb.wf_array(model,[circ_step])
# now populate array with wavefunctions
for i in range(circ_step):
    # construct k-point coordinate on the path
    ang=2.0*np.pi*float(i)/float(circ_step-1)
    kpt=np.array([np.cos(ang)*circ_radius,np.sin(ang)*circ_radius])
    kpt+=circ_center
    # find eigenvectors at this k-point
    (eval,evec)=model.solve_one(kpt,eig_vectors=True)
    # store eigenvector into wf_array object
    w_circ[i]=evec
# make sure that first and last points are the same
w_circ[-1]=w_circ[0]

# compute Berry phase along circular path
print("Berry phase along circle with radius: ",circ_radius)
print("  centered at k-point: ",circ_center)
print("  for band 0 equals    : ", w_circ.berry_phase([0],0))
print("  for band 1 equals    : ", w_circ.berry_phase([1],0))
print("  for band 2 equals    : ", w_circ.berry_phase([2],0))
print("  for band 3 equals    : ", w_circ.berry_phase([3],0))
print("  for band 4 equals    : ", w_circ.berry_phase([4],0))
print("  for band 5 equals    : ", w_circ.berry_phase([5],0))
print("  for band 6 equals    : ", w_circ.berry_phase([6],0))
print("  for band 7 equals    : ", w_circ.berry_phase([7],0))


print("  for both bands equals: ", w_circ.berry_phase([0,1],0))
print()
'''

'''
(fig, ax) = model.visualize(0, 1)
ax.set_title("Title goes here")
fig.savefig("python_tb_plots/toy_model/model.pdf")
'''

import pythtb as pytb
import numpy as np
import matplotlib.pyplot as plt

e_As_p = 0.5
e_Cr_xy = 0.8
e_Cr_x2_y2 = 0.9

h_aa = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 1.]])
h_ca_1u = np.array([[0.5, 0.5, 0.7],
                    [0.5, 0.5, 0.7]])
h_ca_1d = np.array([[0, 0, 0],
                    [0, 0, 0]])
h_ca_2u = np.array([[0, 0, 0],
                    [0, 0, 0]])
h_ca_2d = np.array([[0, 0, 0],
                    [0, 0, 0]])
h_ca_3u = np.array([[0, 0, 0],
                    [0, 0, 0]])
h_ca_3d = np.array([[0, 0, 0],
                    [0, 0, 0]])

h00 = np.zeros((8, 8))
h00[0:2, 0:2] = np.array([[e_Cr_xy, 0], [0, e_Cr_x2_y2]])
h00[2:8, 2:8] = e_As_p * np.identity(6)
h00[2:5, 5:8] = h_aa
h00[0:2, 2:5] = h_ca_1u
h00[0:2, 5:8] = h_ca_2d


h_10 = np.zeros_like(h00)
h_10[2:5, 5:8] = h_aa
h_10[0:2, 5:8] = h_ca_1d

h_m10 = np.zeros_like(h00)
h_m10[2:5, 5:8] = h_aa
h_m10[0:2, 2:5] = h_ca_2u

h_01 = np.zeros_like(h00)
h_01[2:5, 5:8] = h_aa
h_01[0:2, 5:8] = h_ca_3d

h_11 = np.zeros_like(h00)
h_11[2:5, 5:8] = h_aa
h_11[0:2, 2:5] = h_ca_3u

h = [h00, h_01, h_m10, h_01, h_11]


def create_CrAs2(h):
    [h00, h_10, h_m10, h_01, h_11] = h
    sq3 = np.sqrt(3)
    a1 = np.array([1, 0, 0])
    a2 = np.array([-1/2, sq3/2, 0])
    a3 = np.array([0, 0, 2/3])
    lat = [list(a1), list(a2), list(a3)]
    site_pos = [[0, 0, 0],  # cr1
                [-1/3, -1/6, 1/2],  # As+
                [1/6, -1/6, -1/2]]  # As-
    orb = [site_pos[0],
           site_pos[0],
           site_pos[1],
           site_pos[1],
           site_pos[1],
           site_pos[2],
           site_pos[2],
           site_pos[2]]
    tb_model = pytb.tb_model(2, 3, lat=lat, orb=orb, nspin=2)
    for i in range(8):
        tb_model.set_onsite(h00[i, i], i)
    for i in range(8):
        for j in range(8):
            if i < j:
                if not(np.abs(h00[i, j]) < 1e-5):
                    tb_model.set_hop(h00[i, j], i, j, [0, 0, 0])
                if not(np.abs(h_10[i, j]) < 1e-5):
                    tb_model.set_hop(h_10[i, j], i, j, [1, 0, 0])
                if not(np.abs(h_m10[i, j]) < 1e-5):
                    tb_model.set_hop(h_m10[i, j], i, j, [-1, 0, 0])
                if not(np.abs(h_01[i, j]) < 1e-5):
                    tb_model.set_hop(h_01[i, j], i, j, [0, -1, 0])
                if not(np.abs(h_11[i, j]) < 1e-5):
                    tb_model.set_hop(h_11[i, j], i, j, [1, -1, 0])
    return tb_model


model = create_CrAs2(h)

path = [[0., 0.], [2./3., 1./3.], [.5, .5], [1./3., 2./3.], [0., 0.]]
# labels of the nodes
label = (r'$\Gamma $', r'$K$', r'$M$', r'$K^\prime$', r'$\Gamma $')

# call function k_path to construct the actual path
(k_vec, k_dist, k_node) = model.k_path(path, 301)

evals = model.solve_all(k_vec)

# Plot of bands
fig, ax = plt.subplots()
ax.set_xlim(k_node[0], k_node[-1])
ax.set_xticks(k_node)
ax.set_xticklabels(label)
for n in range(len(k_node)):
    ax.axvline(x=k_node[n], linewidth=0.5, color='k')
ax.set_title("Tight binding model CrAs2")
ax.set_xlabel("Path in k-space")
ax.set_ylabel("Band energy")


ax.plot(k_dist, evals[0])
ax.plot(k_dist, evals[1])
ax.plot(k_dist, evals[2])
ax.plot(k_dist, evals[3])

ax.plot(k_dist, evals[4])  # banda cte baja
ax.plot(k_dist, evals[5])  # banda cte baja

ax.plot(k_dist, evals[6])
ax.plot(k_dist, evals[7])
ax.plot(k_dist, evals[8])
ax.plot(k_dist, evals[9])

ax.plot(k_dist, evals[10])  # banda cte alta
ax.plot(k_dist, evals[11])  # banda cte alta
ax.plot(k_dist, evals[12])
ax.plot(k_dist, evals[13])
ax.plot(k_dist, evals[14])
ax.plot(k_dist, evals[15])
ax.set_ylim(np.min(evals)-0.1, np.max(evals)+0.25)

'''
# place a text box in upper left in axes coords
str_t = "$t= $ " + str(round(t,3))
str_eps0 = "$\epsilon_0= $ " +str(round(epsilon_0,3))
str_epsI = "$\epsilon_I = $" + str(round(epsilon_I, 3))
str_Delta = "$\Delta = $" + str(round(Delta, 3))
str_lamb = "$\lambda = $" + str(round(lamb, 3))
txtbox1 = '\n'.join((str_t, str_lamb))
txtbox2 = '\n'.join((str_eps0, str_epsI, str_Delta))
ax.text(0.02, 0.95, txtbox2, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
ax.text(0.18, 0.95, txtbox1, transform=ax.transAxes, fontsize=10,
        verticalalignment='top')
'''

print(np.shape(evals))

fig.savefig("python_tb_plots/CrAs2band.pdf")


(fig, ax) = model.visualize(0, 1)
ax.set_title("Title goes here")
fig.savefig("python_tb_plots/model.pdf")

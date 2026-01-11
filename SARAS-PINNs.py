# -*- coding: utf-8 -*-
# @Time       : 2024/12/26 17:41
# @Author     : Xxxyyzin
# @File       : SARAS-PINNs.py
# @Description:
# @Software   : PyCharm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from scipy.io import loadmat


from functorch import make_functional, vmap, jacrev
from matplotlib import rc
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ----------------- GPU inspection -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # ----------------- Training Point Selection -----------------
# num_tr_pts = 8913
# Sampling_method = "Random"   # "VelGradient", "Random"
# save_training_points = True
# load_added_indices = False


# ----------------- Grid and Parameters -----------------
zmin, zmax, deltaz = 0., 2., 0.01  # km
xmin, xmax, deltax = 0., 4., 0.01

xz_nodes = loadmat('xy_nodes8913.mat')['xy']

# ----------------- velocity model -----------------
Vp_cut_down = np.fromfile("Vp_cut_down_200_400.bin", dtype=np.float32)
velmodel = (Vp_cut_down.reshape((400, 200)) / 1000)

# ----------------- Reference -----------------
T_data = np.load("traveltime.npy")

# ----------------- Source -----------------
sz, sx = 0.5, 2.0
ix = int((sx - xmin) / deltax)
iz = int((sz - zmin) / deltaz)
v_source = velmodel[ix, iz]


# ----------------- grid generation -----------------
z = np.arange(zmin, zmax, deltaz)
x = np.arange(xmin, xmax, deltax)
nz, nx = z.size, x.size

Z,X = np.meshgrid(z,x,indexing='ij')
X_star = [Z.reshape(-1,1), X.reshape(-1,1)]

# ----------------- Surface Topo -----------------
topo = np.loadtxt('../inputs/vtiseam/model/elevation.txt')[0:400, 2]/1000
topo = gaussian_filter1d(topo, 20)
topo = 2*np.gradient(np.gradient(topo)) + np.round(0.05 + 0.02*np.sin(x)*np.cos(x), 4)
base_amplitude = 0.05
variation = 0.10

topo += base_amplitude + variation * np.sin(5 * np.pi * x / (xmax - xmin))
topo += 0.005 * np.sin(6 * np.pi * x / (xmax - xmin))

plt.figure(figsize=(8,3))
plt.plot(x, topo, 'b-', linewidth=2)
plt.xlabel('X (km)')
plt.ylabel('Elevation (km)')
plt.title('平缓起伏地表')
plt.gca().invert_yaxis()
plt.grid(True)
plt.show()


Z, X = np.meshgrid(z, x, indexing='ij')  # shape (nz, nx)
mask = Z < topo[np.newaxis, :]
Z = np.where(mask, np.nan, Z)
X = np.where(mask, np.nan, X)


velmodel = np.where(np.isnan(Z), np.nan, velmodel.T)
T_data = np.where(np.isnan(Z), np.nan, T_data)


# Creating grid, calculating reference traveltimes, and prepare list of grid points for training (X_star)
z = np.arange(zmin,zmax,deltaz)
x = np.arange(xmin,xmax,deltax)

# Analytical solution for the known traveltime part
vel = velmodel[int(round(sz/deltaz)),int(round(sx/deltax))]
T0 = np.sqrt((Z-sz)**2 + (X-sx)**2)/vel


px0 = np.divide(X-sx, T0*vel**2, out=np.zeros_like(T0), where=T0!=0)      # \frac{\partial T_0(\mathbf{x})}{\partial x}
pz0 = np.divide(Z-sz, T0*vel**2, out=np.zeros_like(T0), where=T0!=0)      # \frac{\partial T_0(\mathbf{x})}{\partial y}


total_data = {
    'X': X.reshape(-1,1),
    'Z': Z.reshape(-1,1),
    'px0' : px0.reshape(-1,1),
    'pz0' : pz0.reshape(-1,1),
    'vel' : velmodel.reshape(-1,1),
    'T0'  : T0.reshape(-1,1),
    'T_true' : T_data.reshape(-1,1)
}


nrows,ncols  = Z.shape
upper_boundary_indices = []
for col in range(ncols):
    non_nan_indices = np.where(~np.isnan(velmodel[:, col]))[0]
    if len(non_nan_indices) > 0:
        upper_boundary_indices.append(non_nan_indices[0] * ncols + col)


lower_boundary_indices = np.arange(ncols * (nrows - 1), ncols * nrows)

left_column = velmodel[:, 0]
valid_left_indices = np.where(~np.isnan(left_column))[0]
left_boundary_indices = valid_left_indices * ncols


right_column = velmodel[:, -1]
valid_right_indices = np.where(~np.isnan(right_column))[0]
right_boundary_indices = valid_right_indices * ncols + (ncols - 1)


boundary_indices = np.unique(np.concatenate([
    upper_boundary_indices,
    left_boundary_indices,
    right_boundary_indices
]))


boundary_X = X.reshape(-1, 1)[boundary_indices]
boundary_Z = Z.reshape(-1, 1)[boundary_indices]


plt.figure(figsize=(10, 8))
plt.scatter(boundary_X, boundary_Z, color='red', label='Boundary Points', s=20)
plt.imshow(velmodel, extent=[xmin, xmax, zmax, zmin], cmap='viridis', alpha=0.7, aspect='auto')
plt.colorbar(label='Velocity (Km/s)')
plt.xlabel('Offset (Km)')
plt.ylabel('Depth (Km)')
plt.title('Boundary Points on Velocity Model')
plt.legend()
plt.grid()
plt.show()

boundary_data = {
    'X': X.reshape(-1,1)[boundary_indices],
    'Z': Z.reshape(-1,1)[boundary_indices],
    'px0' : px0.reshape(-1,1)[boundary_indices],
    'pz0' : pz0.reshape(-1,1)[boundary_indices],
    'vel' : velmodel.reshape(-1,1)[boundary_indices],
    'T0'  : T0.reshape(-1,1)[boundary_indices],
    'T_true' : T_data.reshape(-1,1)[boundary_indices]

}

class TrainingData:
    def __init__(self,total_datasets, boundary_datasets,vs,xs,zs,num_pde, num_bcs,Sampling_method):
        self.total_datasets = total_datasets
        self.boundary_datasets = boundary_datasets
        self.V_s = vs
        self.X_s = xs
        self.Z_s = zs

        self.num_pde = num_pde
        self.num_bcs = num_bcs
        self.sampling_method = Sampling_method

    def pde_datasets(self, num_pde):
        if self.sampling_method == "Random":
            tot_tr_pts = int(round(num_pde * (np.sum(1 + np.isnan(X)) / X.size)))
            selected_pts = np.random.choice(len(self.total_datasets["X"]), tot_tr_pts, replace=False)
            selected_pts = X.reshape(-1, 1)[selected_pts] * 0 + selected_pts.reshape(-1, 1)
            selected_pts = selected_pts[~np.isnan(selected_pts)]
            selected_pts = selected_pts.astype(int)
        elif self.sampling_method == "VelGradient":
            z_idx = np.maximum(np.round((xz_nodes[:, 1]) / deltaz).astype(int) - 1, 0)
            x_idx = np.maximum(np.round((xz_nodes[:, 0]) / deltax).astype(int) - 1, 0)
            flattened_idx = z_idx * nx + x_idx
            unique_idx, unique_positions = np.unique(flattened_idx, return_index=True)
            # 保留唯一索引
            flattened_idx_unique = flattened_idx[np.sort(unique_positions)]
            selected_pts = flattened_idx_unique
            selected_pts = X.reshape(-1, 1)[selected_pts] * 0 + selected_pts.reshape(-1, 1)
            selected_pts = selected_pts[~np.isnan(selected_pts)]
            selected_pts = selected_pts.astype(int)

        pde_datasets_dic = {
            'X': torch.tensor(self.total_datasets['X'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(
                -1, 1).to(device),
            'Z': torch.tensor(self.total_datasets['Z'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(
                -1, 1).to(device),
            'px0': torch.tensor(self.total_datasets['px0'][selected_pts], dtype=torch.float64,
                                requires_grad=True).reshape(-1, 1).to(device),
            'pz0': torch.tensor(self.total_datasets['pz0'][selected_pts], dtype=torch.float64,
                                requires_grad=True).reshape(-1, 1).to(device),
            'vel': torch.tensor(self.total_datasets['vel'][selected_pts], dtype=torch.float64,
                                requires_grad=True).reshape(-1, 1).to(device),
            'T0': torch.tensor(self.total_datasets['T0'][selected_pts], dtype=torch.float64,
                               requires_grad=True).reshape(-1, 1).to(device),
            'T_true': torch.tensor(self.total_datasets['T_true'][selected_pts], dtype=torch.float64,
                                   requires_grad=True).reshape(-1, 1).to(device)
        }
        return pde_datasets_dic

    def ibs_datasets(self):
        ibs_datasets_dic = {
            'X': torch.tensor(self.X_s, dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'Z': torch.tensor(self.Z_s, dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'px0': torch.tensor(0, dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'pz0': torch.tensor(0, dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'vel': torch.tensor(self.V_s, dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'T0': torch.tensor(0, dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'T_true': torch.tensor(0, dtype=torch.float64, requires_grad=True).reshape(-1,1).to(device)
        }
        return ibs_datasets_dic

    def bcs_datasets(self,num_bcs):
        selected_pts = np.random.choice(len(self.boundary_datasets["X"]), num_bcs, replace=False)
        bcs_datasets_dic = {
            'X': torch.tensor(self.boundary_datasets['X'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'Z': torch.tensor(self.boundary_datasets['Z'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'px0': torch.tensor(self.boundary_datasets['px0'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'pz0': torch.tensor(self.boundary_datasets['pz0'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'vel': torch.tensor(self.boundary_datasets['vel'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'T0': torch.tensor(self.boundary_datasets['T0'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1, 1).to(device),
            'T_true': torch.tensor(self.boundary_datasets['T_true'][selected_pts], dtype=torch.float64, requires_grad=True).reshape(-1,1).to(device)
        }
        return bcs_datasets_dic

### Tanh Xavier initialization
class PINNModel(nn.Module):
    def __init__(self, layers):
        super(PINNModel, self).__init__()
        self.hidden = nn.ModuleList()


        for i in range(len(layers) - 1):
            layer = nn.Linear(layers[i], layers[i + 1])
            layer.weight.data = layer.weight.data.to(torch.float64)
            layer.bias.data = layer.bias.data.to(torch.float64)
            init.xavier_normal_(layer.weight)
            init.zeros_(layer.bias)
            self.hidden.append(layer)

        self.activation = nn.Tanh()

    def forward(self, zt, xt):
        x = torch.cat((zt, xt), dim=1)
        for layer in self.hidden[:-1]:
            x = self.activation(layer(x))
        x = self.hidden[-1](x)
        return x

class PINNTrainer:
    def __init__(self, layers, total_datasets, boundary_datasets, vs, xs, zs, num_pde, num_bcs,kernel_size_pde,kernel_size_bcs,
                  device='cpu',adding_num=100,k=1,m=1,c1=1,c2=1, alpha=0.5,beta=0.5,training_epochs=[10,100,100],learning_rates=[0.001,0.001,0.001],
                 adaptiveSample_iteration=3,sample_method="Random",log_NTK = False, update_lam = False,adaptive_samples=False):
        self.layers = layers
        self.device = device
        self.model = PINNModel(self.layers).to(self.device)
        self.training_data = TrainingData(total_datasets, boundary_datasets,vs, xs, zs, num_pde, num_bcs,sample_method)
        self.pde_datasets = self.training_data.pde_datasets(num_pde)
        self.bcs_datasets = self.training_data.bcs_datasets(num_bcs)
        self.ibs_datasets = self.training_data.ibs_datasets()

        self.total_datasets = {key: torch.tensor(value, dtype=torch.float64).to(device) for key, value in
                               total_datasets.items()}

        self.total_datasets["X"] = torch.tensor(self.total_datasets["X"], requires_grad=True)
        self.total_datasets["Z"] = torch.tensor(self.total_datasets["Z"], requires_grad=True)


        self.kernel_size_pde = kernel_size_pde
        self.kernel_size_bcs = kernel_size_bcs


        # adaptive samples
        self.adding_num = adding_num
        self.k = k
        self.m = m
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta

        self.training_epochs = training_epochs
        self.learning_rates = learning_rates
        self.adaptive_samples = adaptive_samples
        self.adaptiveSample_iteration = adaptiveSample_iteration
        self.sample_method = sample_method

        if not self.adaptive_samples:
            self.adaptiveSample_iteration = 1

        # weights--lamda
        self.lam_u_val = 1
        self.lam_ibs_val = 0
        self.lam_bcs_val = 1

        self.total_loss_value = []
        self.u_loss_value = []
        self.ibs_value = []
        self.bcs_value = []
        self.datasets_loss = []

        self.K_u_log = []
        self.K_ibs_log = []
        self.K_bcs_log = []

        self.lam_u_log = []
        self.lam_ibs_log = []
        self.lam_bcs_log =[]

        self.added_indices = []

        self.log_NTK = log_NTK
        self.update_lam = update_lam
        self.adaptive_samples = adaptive_samples


    def pde_loss(self,pde_datasets):
        tau = self.model(pde_datasets["Z"], pde_datasets["X"])
        dtaudx = torch.autograd.grad(tau, pde_datasets["X"], grad_outputs=torch.ones_like(tau), create_graph=True)[0]
        dtaudz = torch.autograd.grad(tau, pde_datasets["Z"], grad_outputs=torch.ones_like(tau), create_graph=True)[0]
        L = (pde_datasets["T0"] * dtaudx + tau * pde_datasets["px0"]) ** 2 + (pde_datasets["T0"] * dtaudz +
            tau * pde_datasets["pz0"]) ** 2 - 1.0 / pde_datasets["vel"] ** 2
        pde_loss = torch.mean(L**2)

        return pde_loss,L

    def all_datasets_loss(self):
        tau = self.model(self.total_datasets["Z"], self.total_datasets["X"])
        T0 = self.total_datasets["T0"]
        pred_T = tau * T0
        T_true = self.total_datasets["T_true"]

        absolute_error = torch.abs(pred_T - T_true)
        all_datasets_loss = torch.mean(absolute_error[~torch.isnan(absolute_error)])

        return all_datasets_loss

    def ibs_loss(self):
        tau = self.model(self.ibs_datasets["Z"], self.ibs_datasets["X"])
        ibs_loss = torch.mean((tau-1)**2)
        return ibs_loss

    def bcs_loss(self,bcs_datasets):
        tau = self.model(bcs_datasets["Z"], bcs_datasets["X"])
        bcs_loss = torch.mean((tau*bcs_datasets["T0"]-bcs_datasets["T_true"])**2)
        return bcs_loss

    def compute_jacbian(self,fnet_single, params, x, z):
        jac = vmap(jacrev(fnet_single), (None, 0, 0))(params, x, z)
        jac = [j.flatten(2) for j in jac]
        return jac

    def ntk_datasets(self,datasets,kernel_size):
        select_ntk_datasets = torch.randperm(len(datasets["X"]))[:kernel_size]
        ntk_datasets = {
            "X": datasets["X"][select_ntk_datasets],
            "Z": datasets["Z"][select_ntk_datasets],
            "px0": datasets["px0"][select_ntk_datasets],
            "pz0": datasets["pz0"][select_ntk_datasets],
            "vel": datasets["vel"][select_ntk_datasets],
            "T0": datasets["T0"][select_ntk_datasets],
        }
        return ntk_datasets


    def compute_ntk(self,jac1, jac2, compute='trace'):
        # Compute J(x1) @ J(x2).T
        einsum_expr = None
        if compute == 'full':
            einsum_expr = 'Naf,Mbf->NMab'
        elif compute == 'trace':
            einsum_expr = 'Naf,Maf->NM'
        elif compute == 'diagonal':
            einsum_expr = 'Naf,Maf->NMa'
        else:
            assert False
        ntk = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
        ntk = ntk.sum(0)
        return ntk

    def adaptive_samples_RGAS(self,inital_datasets,epsilon=0):
        tau = self.model(self.total_datasets["Z"], self.total_datasets["X"])
        tau_pred = tau.reshape(nz,nx)
        T0 = self.total_datasets["T0"].reshape(nz,nx)
        T_pred = tau_pred * T0
        T_true = self.total_datasets["T_true"].reshape(nz,nx)
        dtaudx = torch.autograd.grad(tau,self.total_datasets["X"],grad_outputs=torch.ones_like(tau),create_graph=True)[0]
        dtaudz = torch.autograd.grad(tau, self.total_datasets["Z"], grad_outputs=torch.ones_like(tau), create_graph=True)[0]
        L = (self.total_datasets["T0"] * dtaudx + tau * self.total_datasets["px0"]) ** 2 + (self.total_datasets["T0"] * dtaudz + tau * self.total_datasets["pz0"]) ** 2 - 1.0 / self.total_datasets["vel"] ** 2


        # Compute epsilon(x) = |L_r(x)-epsilon|
        residual_loss = torch.abs(L)-epsilon

        # Compute domains: Safe (Omega_S) and Non-Safe (Omega_nS)
        safe_domain = residual_loss <= 0
        non_safe_domain = residual_loss > 0

        # For non-safe domain, compute sampling probability
        grad_epsilon = torch.sqrt(
            torch.sum(torch.autograd.grad(residual_loss, self.total_datasets["Z"],
                                          grad_outputs=torch.ones_like(residual_loss), create_graph=True)[0] ** 2,
                      dim=-1)
            + torch.sum(torch.autograd.grad(residual_loss, self.total_datasets["X"],
                                            grad_outputs=torch.ones_like(residual_loss), create_graph=True)[0] ** 2,
                        dim=-1)
        )


        mask_r = ~torch.isnan(residual_loss)
        residual_loss_no_nan = torch.where(mask_r, residual_loss, torch.tensor(0.0, device=residual_loss.device))
        denominator_r = torch.mean(torch.pow(residual_loss_no_nan[mask_r], self.k))


        if denominator_r == 0:
            raise ValueError("Mean of residual_loss (after ignoring NaN) is zero, which will cause a division by zero.")

        err_eq = (
                torch.pow(residual_loss_no_nan, self.k) / torch.mean(torch.pow(residual_loss_no_nan, self.k))
                + self.c1
        )

        mask_grad = ~torch.isnan(grad_epsilon)
        grad_loss_no_nan = torch.where(mask_grad, grad_epsilon, torch.tensor(0.0, device=grad_epsilon.device))


        grad_norm = (
                torch.pow(grad_loss_no_nan, self.m) / torch.mean(torch.pow(grad_loss_no_nan, self.m))
                + self.c2
        )
        grad_norm = grad_norm.unsqueeze(-1)
        sampling_prob = torch.pow(err_eq,self.alpha) * torch.pow(grad_norm,self.beta)
        sampling_prob_plot = sampling_prob / torch.sum(sampling_prob)
        # Mask the sampling probability to only include the non-safe domain
        sampling_prob = sampling_prob * non_safe_domain.float()
        # Normalize the sampling probability to sum to 1
        sampling_prob_normalized = sampling_prob / torch.sum(sampling_prob)

        sampling_prob_plot = torch.abs(sampling_prob_plot).detach().cpu().numpy().reshape(nz, nx)

        # Sample points only from the non-safe domain
        indices = torch.multinomial(
            sampling_prob_normalized.flatten(), self.adding_num, replacement=False
        )

        valid_indices = (T_true != 0) & (~torch.isnan(T_true))
        RAE = torch.mean(torch.abs(T_pred-T_true)[valid_indices] / T_true[valid_indices])
        print(f"MAE: {torch.mean(torch.abs(T_pred-T_true)[valid_indices])},  RAE: {RAE}")


        added_samples = {
            "X": self.total_datasets["X"][indices],
            "Z": self.total_datasets["Z"][indices],
            "vel": self.total_datasets["vel"][indices],
            "px0": self.total_datasets["px0"][indices],
            "pz0": self.total_datasets["pz0"][indices],
            "T0": self.total_datasets["T0"][indices],
            "T_true": self.total_datasets["T_true"][indices]
        }

        updated_samples = {
            "X": torch.cat((inital_datasets["X"], added_samples["X"].reshape(-1, 1)), dim=0),
            "Z": torch.cat((inital_datasets["Z"], added_samples["Z"].reshape(-1, 1)), dim=0),
            "vel": torch.cat((inital_datasets["vel"], added_samples["vel"].reshape(-1, 1)), dim=0),
            "px0": torch.cat((inital_datasets["px0"], added_samples["px0"].reshape(-1, 1)), dim=0),
            "pz0": torch.cat((inital_datasets["pz0"], added_samples["pz0"].reshape(-1, 1)), dim=0),
            "T0": torch.cat((inital_datasets["T0"], added_samples["T0"].reshape(-1, 1)), dim=0),
            "T_true": torch.cat((inital_datasets["T_true"], added_samples["T_true"].reshape(-1, 1)), dim=0),
        }

        return updated_samples, added_samples, inital_datasets, indices, sampling_prob_plot


    def train(self):
        start_time = time.time()

        self.sampling_prob_normalized_plot_list = []
        for num_adpativeSample in range(self.adaptiveSample_iteration):
            if num_adpativeSample != 0:
                print("=======================================================")
            print(f"Starting Adaptive Sample Iteration {num_adpativeSample}/{self.adaptiveSample_iteration}")
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rates[num_adpativeSample])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,eta_min=1e-3)

            if num_adpativeSample == 0 :
                pde_datasets = self.pde_datasets

                # # Visualization of selected training points
                # plt.figure(figsize=(6, 6))
                plt.scatter(pde_datasets["X"].detach().cpu().numpy(), pde_datasets["Z"].detach().cpu().numpy(), c='black', label='Training Points', s=1)
                plt.scatter(sx, sz, c='red', label='Source Location', s=100, edgecolor='black')
                plt.xlim(xmin, xmax)
                plt.ylim(zmax, zmin)
                plt.title('Selected Training Points')
                plt.xlabel('X Coordinate')
                plt.ylabel('Z Coordinate')
                plt.legend()
                # plt.grid()
                # plt.savefig(os.path.join(write_path, 'training_points_Random.png'))  # Save plot as image
                plt.show()  # Show the plot


                rc('font', family='Times New Roman')

                # ---------------- Figure 1: velocity Model ----------------
                fig1, ax1 = plt.subplots(figsize=(6, 6))
                im = ax1.imshow(velmodel, extent=[xmin, xmax, zmax, zmin], cmap='jet', aspect='auto')
                ax1.plot(sx, sz, 'k*', markersize=10, label="Source")
                ax1.set_xlabel('Offset (Km)', fontsize=14)
                ax1.set_ylabel('Depth (Km)', fontsize=14)
                ax1.set_aspect('equal')


                rect1 = patches.Rectangle(
                    (0.1, 0.25),
                    1.4, 1.5,
                    linewidth=1.5, edgecolor="white", facecolor="none"
                )
                ax1.add_patch(rect1)

                rect2 = patches.Rectangle(
                    (2.2, 0.5),
                    1.7, 1.4,
                    linewidth=1.5, edgecolor="white", facecolor="none"
                )
                ax1.add_patch(rect2)


                divider = make_axes_locatable(ax1)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = fig1.colorbar(im, cax=cax)
                cbar.set_label('Velocity (Km/s)', fontsize=14)

                plt.tight_layout()
                plt.savefig('Vel.png', dpi=300, bbox_inches='tight')
                plt.show()

                # ---------------- Figure 2: Sampling points ----------------
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.scatter(pde_datasets["X"].detach().cpu().numpy(), pde_datasets["Z"].detach().cpu().numpy(), c='black', label='Training Points', s=1)
                ax2.scatter(sx, sz, c='red', marker='*', label='Source Location', s=100, edgecolor='black')
                ax2.set_xlim(xmin, xmax)
                ax2.set_ylim(zmax, zmin)


                rect1 = patches.Rectangle(
                    (0.1, 0.25),
                    1.4, 1.5,
                    linewidth=1.5, edgecolor="red", facecolor="none"
                )
                ax2.add_patch(rect1)


                ax2.set_yticks(np.arange(zmin, zmax, 0.5))
                ax2.set_xlabel('Offset (Km)', fontsize=16)
                ax2.set_ylabel('Depth (Km)', fontsize=16)
                ax2.legend(loc='upper right', fontsize=12)
                ax2.set_aspect('equal')
                plt.tight_layout()
                plt.savefig('training_points_VelGradient.png', dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Initial training num is {len(pde_datasets['X'])}")



            else:
                pde_datasets, added_samples, inital_samples, added_indice,sampling_prob_plot = self.adaptive_samples_RGAS(pde_datasets)
                self.added_indices.append(added_indice)
                self.sampling_prob_normalized_plot_list.append(sampling_prob_plot)


            for epoch in range(self.training_epochs[num_adpativeSample]):
                self.model.train()  # Set the model to training mode
                optimizer.zero_grad()  # Clear gradients

                # Compute loss
                pde_loss = self.pde_loss(pde_datasets)[0]
                ibs_loss = self.ibs_loss()
                bcs_loss = self.bcs_loss(self.bcs_datasets)

                total_loss = self.lam_u_val * pde_loss + self.lam_bcs_val * bcs_loss + self.lam_ibs_val * ibs_loss
                # total_loss = self.lam_u_val * pde_loss

                if epoch % 1000 == 0:
                    datasets_loss = self.all_datasets_loss()
                    self.datasets_loss.append(datasets_loss.item())

                # total_loss = self.lam_u_val * pde_loss + self.lam_ibs_val * ibs_loss
                self.total_loss_value.append(total_loss.item())
                self.u_loss_value.append(pde_loss.item())
                self.ibs_value.append(ibs_loss.item())
                self.bcs_value.append(bcs_loss.item())


                print(f'Epoch {epoch + 1}/{self.training_epochs[num_adpativeSample]}, Loss: {total_loss.item()}, u_loss: {pde_loss.item()}, ibs_loss: {ibs_loss.item()}, bcs_loss: {bcs_loss.item()}, datasets_loss: {datasets_loss.item()}')
                # print(
                #     f'Epoch {epoch + 1}/{self.training_epochs[num_adpativeSample]}, Loss: {total_loss.item()}, u_loss: {pde_loss.item()}, ibs_loss: {ibs_loss.item()}')

                if epoch == (self.training_epochs[num_adpativeSample]-1):
                    pde_loss_matrix = torch.abs(self.pde_loss(pde_datasets)[1]).detach().cpu().numpy()


                if self.log_NTK:
                    if epoch % 500 == 0 :
                        with torch.no_grad():
                            fnet, params = make_functional(self.model)

                            def fnet_single(params, x, z):
                                return fnet(params, x.unsqueeze(0), z.unsqueeze(0)).squeeze(0)

                            u_ntk_datasets = self.ntk_datasets(pde_datasets,self.kernel_size_pde)
                            bcs_ntk_datasets = self.ntk_datasets(self.bcs_datasets,self.kernel_size_bcs)


                            J_u = self.compute_jacbian(fnet_single, params, u_ntk_datasets["X"], u_ntk_datasets["Z"])
                            K_u = self.compute_ntk(J_u, J_u)

                            J_bcs = self.compute_jacbian(fnet_single, params, bcs_ntk_datasets["X"], bcs_ntk_datasets["Z"])
                            K_bcs = self.compute_ntk(J_bcs, J_bcs)

                            self.K_u_log.append(K_u)
                            # self.K_ibs_log.append(K_ibs)
                            self.K_bcs_log.append(K_bcs)

                            if self.update_lam:
                                # if  epoch != 0:
                                # trace_K = torch.trace(K_u) + torch.trace(K_ibs) + torch.trace(K_bcs)
                                trace_K = torch.trace(K_u)  + torch.trace(K_bcs)
                                self.lam_u_val = trace_K / torch.trace(K_u)
                                # self.lam_ibs_val = trace_K / torch.trace(K_ibs)
                                self.lam_ibs_val = 0
                                self.lam_bcs_val = trace_K / torch.trace(K_bcs)
                                # self.lam_bcs_val = 0


                                self.lam_u_log.append(self.lam_u_val)
                                self.lam_ibs_log.append(self.lam_ibs_val)
                                self.lam_bcs_log.append(self.lam_bcs_val)

                                stop_update_time = time.time()
                                print(f"Time :{stop_update_time-start_time}")
                                # print(f'lam_u: {self.lam_u_val.item()}, lam_ibs: {self.lam_ibs_val.item()}, lam_bcs: {self.lam_bcs_val.item()}')
                                print(
                                    f'lam_u: {self.lam_u_val.item()}, lam_ibs: {self.lam_ibs_val}, lam_bcs: {self.lam_bcs_val}')


                total_loss.backward(retain_graph=True)
                optimizer.step()  # Update weights
                scheduler.step()  # Adjust learning rate




    def predict(self, zt_new, xt_new):
        self.model.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            zt_new = zt_new.to(self.device)
            xt_new = xt_new.to(self.device)
            tau_pred = self.model(zt_new, xt_new)
        return tau_pred


X_star = [Z.reshape(-1,1), X.reshape(-1,1)] # Grid points for prediction
X_star = torch.tensor(X_star).to(device)
X_all = X_star[1].to(torch.float64)
Z_all = X_star[0].to(torch.float64)
layers = [2] + [30] * 10 + [1]

# Initialize the trainer
trainer = PINNTrainer(
    layers = layers,
    total_datasets = total_data,
    boundary_datasets = boundary_data,
    vs = vel,
    xs = sx,
    zs = sz,
    num_pde = 8913,
    num_bcs = 200,
    kernel_size_pde=500,
    kernel_size_bcs=500,
    device=device,
    adding_num=15,
    k=2,
    m = 2,
    c1 = 0,
    c2 = 0,
    alpha = 0.5,
    beta = 1,
    training_epochs=[1000]+[500]*196+[1000],   #[1000]+[500]*196+[1000],
    learning_rates=[0.001]*200,
    adaptiveSample_iteration = 94,
    sample_method = "VelGradient",   # "VelGradient", "Random"， “Uniform”
    log_NTK = True,
    update_lam = True,
    adaptive_samples= True
)

load_PreTrained_model = False
save_T_true = False
pretrained_path = "./Results/RARD_model.pth"


# Train the model
if load_PreTrained_model == True:
    trainer.model.load_state_dict(torch.load(pretrained_path))
    print(f"Loading pre-trained model: {pretrained_path}.")
else:
    print("Non pre-trained model.")

if save_T_true == True:
    np.save("Ttrue.npy", T_data)

print(f"Sampling method is : {trainer.sample_method}.")

t0 = time.time()
# Train the model
trainer.train()
t_end = time.time()
print("======================================")
print(f"Training cost about {np.round((t_end - t0) / 60, 2)} minutes.")


torch.save(trainer.model.state_dict(), "RARD_model.pth")
loss = trainer.total_loss_value
datasets_loss_log = trainer.datasets_loss
sampling_prob_normalized_plot = trainer.sampling_prob_normalized_plot_list

np.save("sampling_prob_normalized_plot.npy",sampling_prob_normalized_plot)

if trainer.adaptive_samples and trainer.adaptiveSample_iteration!=1:
    added_indices = trainer.added_indices
    unique_indices = list(set(torch.cat(added_indices).tolist()))
    np.save("added_indices_RARD.npy",unique_indices)
u_ntk = trainer.K_u_log
bcs_ntk = trainer.K_bcs_log

lam_u = trainer.lam_u_log
lam_bcs = trainer.lam_bcs_log
lam_u_combined = [item.item() for item in lam_u]
lam_bcs_combined = [item.item() for item in lam_bcs]


pred_tau = trainer.predict(Z_all, X_all)
pred_tau = pred_tau.reshape(Z.shape)
pred_tau = pred_tau.detach().cpu().numpy()

T_pred = pred_tau*T0
PINN_error = np.abs(T_pred-T_data)
valid_indices = (T_data != 0) & (~np.isnan(T_data))
mean_error = np.mean(PINN_error[valid_indices] / T_data[valid_indices])
print(f"MSE loss is {np.mean(PINN_error[valid_indices])}, ABS-MSE loss is {mean_error}")

# plot-loss
loss_log = trainer.total_loss_value
datasets_loss_log = trainer.all_datasets_loss

np.save("loss_VGAS.npy",loss_log)
np.save("Tpred_VGAS.npy",T_pred)
np.save("PINNError_VGAS.npy",PINN_error)
np.save("datasets_loss_VGAS.npy",datasets_loss_log)


# Plot the PINN solution error
plt.style.use('default')
plt.figure(figsize=(5, 5))
ax = plt.gca()
# im = ax.imshow(PINN_error, extent=[xmin, xmax, zmax, zmin], aspect=1, cmap="jet", vmax=0.0001)
im = ax.imshow(PINN_error, extent=[xmin, xmax, zmax, zmin], aspect=1, cmap="jet")
plt.xlabel('Offset (km)', fontsize=14)
plt.xticks(fontsize=10)
plt.ylabel('Depth (km)', fontsize=14)
plt.yticks(fontsize=10)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="6%", pad=0.15)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('seconds', size=10)
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("PINNError_RARD.png", format='png', dpi=300 ,bbox_inches="tight")
plt.show()

#### Traveltime contour plots
plt.figure(figsize=(5,5))
ax = plt.gca()
im1 = ax.contour(T_data, 6, extent=[xmin,xmax,zmin,zmax], colors='r')
im2 = ax.contour(T_pred, 6, extent=[xmin,xmax,zmin,zmax], colors='k',linestyles = 'dashed')
plt.plot(x,topo,'k')

ax.plot(sx,sz,'k*',markersize=8)
plt.xlabel('Offset (km)', fontsize=14)
plt.ylabel('Depth (km)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.gca().invert_yaxis()
h1,_ = im1.legend_elements()
h2,_ = im2.legend_elements()
ax.legend([h1[0], h2[0]], ['Analytical', 'PINN'],fontsize=12)

ax.xaxis.set_major_locator(plt.MultipleLocator(0.5))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("contours.pdf", format='pdf', bbox_inches="tight")
# plt.savefig(os.path.join(figs_path, "contours.pdf"), format='pdf', bbox_inches="tight")
plt.show()



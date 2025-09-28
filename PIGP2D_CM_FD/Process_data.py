import json
import torch
import numpy as np
from models.lmgp import LMGP 
from utils.utils_general import get_tkwargs, projectDensity
import matplotlib.pyplot as plt
from gpytorch.settings import cholesky_jitter
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from models.FE import FE
from utils.utils_general import dynamic_binarize_density_3 as dynamic_binarize_density
import cv2

colors = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=128)

# set parameters for the plot
colors_density = ['#FFFFFF', '#addc30', '#5ec962', '#28ae80', '#21918c', '#2c72e8', '#3b526b', '#472d7b', '#440154']  # White to the purple end
cmap_density = LinearSegmentedColormap.from_list('custom_cmap', colors_density, N=128)

colors_contours = ["#3b4fbf", "#79a2f2", "#c2d7f2", "#f2c9bb", "#f2856d", "#bf0637"]
cmap_contours = LinearSegmentedColormap.from_list("custom_cmap", colors_contours, N=128)

cmap_curves = {'blue_dark': "#3b4cc0",'blue_light': "#8db0fe",
                 'red_dark':"#b40426",'red_light':"#f4987a",
                 'green_dark': "#33a645",'green_light': "#afd991",
                 'gray_dark': "#555555",'gray_light': "#bababa",
                 'purple_dark': "#6659a7",'purple_light': "#c2bdde",
                 'orange_dark': "#ef7e21",'orange_light': "#f6c092",
                 'yellow_dark': "#f6d31b",'yellow_light': "#fae98f",
                 'cyan_dark': "#7bddf3",'cyan_light': "#bbedf2",
                 'magenta_dark': "#fe3da7",'magenta_light': "#fe9fd3",
                 }

def moving_average(arr, window_size):
    """
    Apply a simple moving average filter to smooth the input array.
    
    Parameters:
    - arr: numpy array, the input data to be smoothed.
    - window_size: int, the size of the moving window for averaging.
    
    Returns:
    - smoothed: numpy array, the smoothed data.
    """
    return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')

def eval_elem_rho(model_list,type):
    mask_col_elem = model_list[0].mask_col_elem
    mask_nan_elem = model_list[0].mask_nan_elem
    for model in model_list:
        model.eval

    elem_x = model_list[0].X_col_elem.clone()
    m_col = model_list[0].mean_module_NN_All(elem_x)
    
    g_rho = (model_list[2].covar_module(model_list[2].train_inputs[0], elem_x)).evaluate()

    with cholesky_jitter(1e-5):
            model_list[2].chol_decomp = model_list[2].covar_module(model_list[2].train_inputs[0]).cholesky()

    K_inv_offset_rho = model_list[2].chol_decomp._cholesky_solve(model_list[2].train_targets.unsqueeze(-1) - torch.sigmoid(model_list[0].mean_module_NN_All(model_list[2].train_inputs[0])[:,2].unsqueeze(-1)))

    if type == 'proj':
        rho = projectDensity( (torch.sigmoid(m_col[:,2].unsqueeze(-1)) + g_rho.t() @ K_inv_offset_rho).squeeze(-1), model_list[0].MP['b'] )
    else:
        rho = (torch.sigmoid(m_col[:,2].unsqueeze(-1)) + g_rho.t() @ K_inv_offset_rho).squeeze(-1)
    
    # assign zero to rho
    rho[mask_nan_elem] = 0.0
    rho = rho.detach().cpu().numpy()
    rho = np.clip(rho, 0, 1)
    elem_x = elem_x.detach().cpu().numpy()

    Data = {'elem_x':elem_x,'rho':rho}
    return Data

def extract_rho(save_folder, iter, type, X_elem):
    X_col_elem = X_elem['X_col_elem']
    mask_col_elem = X_elem['mask_col_elem']
    mask_nan_elem = X_elem['mask_nan_elem']
    # load the MP
    with open(save_folder + "MP.json", "r") as file:
        MP = json.load(file)
    # load the NN_config
    with open(save_folder + "NN_config.json", "r") as file:
        NN_config = json.load(file)
    # load the training data:
    Training = torch.load(save_folder + "Training.pth",map_location=torch.device('cpu'))

    X_col = Training['X_col'].type(tkwargs["dtype"]).requires_grad_(True)
    u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False)
    u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False)
    v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False)
    rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False)
    X_col_elem = torch.tensor(X_col_elem).type(tkwargs["dtype"]).requires_grad_(True)
    mask_col_elem = torch.tensor(mask_col_elem)
    mask_nan_elem = torch.tensor(mask_nan_elem)
    
    model_u = LMGP(train_x = u_X_train, train_y = u_train, collocation_x = X_col,
                NN_config = NN_config,
                Training = Training,
                name_output='u',
                MP=MP,
                num_output = 3)

    model_v = LMGP(train_x = v_X_train, train_y = v_train, collocation_x = X_col,
                NN_config = NN_config,
                Training = Training,
                name_output='v',
                MP=MP,
                num_output = 3)

    model_rho = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                NN_config = NN_config,
                Training = Training,
                name_output='rho',
                MP=MP,
                num_output = 3)

    model_list = [model_u, model_v, model_rho]
    if iter > 0:
        model_list[0].mean_module_NN_All.load_state_dict(torch.load(save_folder + f'Trained_mean_module_NN_params_{iter}.pth',map_location=torch.device('cpu')))
    else:
        model_list[0].mean_module_NN_All.load_state_dict(torch.load(save_folder + 'Trained_mean_module_NN_params_final.pth',map_location=torch.device('cpu')))
    
    model_list[0].X_col_elem = X_col_elem
    model_list[0].mask_col_elem = mask_col_elem
    model_list[0].mask_nan_elem = mask_nan_elem
    data_elem = eval_elem_rho(model_list,type)
    return data_elem

def hex_to_rgba(hex_color, alpha=1.0):
    """
    Convert hex color to RGBA format and adjust alpha (transparency).
    Default alpha is 1.0 (fully opaque).
    
    Parameters:
    - hex_color: str, color in hex format (e.g., '#3b4cc0')
    - alpha: float, transparency level (0.0 to 1.0)
    
    Returns:
    - rgba: tuple, (r, g, b, a)
    """
    rgba = mcolors.to_rgba(hex_color)
    return (rgba[0], rgba[1], rgba[2], alpha)

def FE_eval(X_col,rho,MP,example):
    Nelx = int(MP['domain']['x'][1]) # must use element size = 1
    Nely = int(MP['domain']['y'][1]) # must use element size = 1
    Emax = MP['Emax']
    nu = MP['nu']
    p = MP['p']
    elemSize = np.array([1.0,1.0])
    mesh = {'nelx':Nelx, 'nely':Nely, 'elemSize':elemSize}
    matProp = {'E':Emax, 'nu':nu,'penal':p}; 
    exampleName = example
    physics = 'Structural'
    numDOFPerNode = 2
    ndof = numDOFPerNode*(Nelx+1)*(Nely+1)

    # define the node coordinates:
    nodeXY = np.zeros((int(ndof/numDOFPerNode),2))
    ctr = 0
    for i in range(Nelx+1):
        for j in range(Nely+1):
            nodeXY[ctr,0] = elemSize[0]*i
            nodeXY[ctr,1] = elemSize[1]*j
            ctr += 1

    force = np.zeros((ndof,1))
    
    if example == 'EX2D1':
        force[2*Nely+1, 0 ] = -0.1
        # define displacement boundary condition
        set1 = np.arange(0, 2 * (Nely + 1), 2) # fix the left edge x disp
        set2 = np.array([2*(Nely+1)*Nelx+1]) # fix the bottom right corner x and y disp
        fixed = np.union1d(set1, set2)
    
    elif example == 'EX2D2':
        indices_force = (Nely + 1) * Nelx # indices in the global node vector
        force[2*indices_force+1, 0 ] = -0.1
        # define displacement boundary condition
        set1 = np.arange(2 * (Nely + 1)) # fix the left edge x disp
        set2 = np.array([ ], dtype=int) # fix the bottom right corner y disp
        fixed = np.union1d(set1, set2)
    
    elif example == 'EX2D3':
        indices_force = np.where((nodeXY[:, 1] == 100))[0] # indices in the global node vector
        # 0 -> 0,1
        # 1 -> 2,3
        # 2 -> 4,5
        # n -> 2*n, 2*n+1
        force[2*indices_force+1, 0 ] = -0.2/len(indices_force)
        # define displacement boundary condition
        bottom_edge1 = np.where((nodeXY[:, 0] == 0) & (nodeXY[:, 1] == 0))[0] # indices in the global node vector
        bottom_edge2 = np.where((nodeXY[:, 0] == Nelx) & (nodeXY[:, 1] == 0))[0]
        fixed1 = []
        for n in bottom_edge1:
            fixed1.append(2 * n)     # x DOF
            fixed1.append(2 * n + 1) # y DOF
        fixed2 = []
        for n in bottom_edge2:
            fixed2.append(2 * n)     # x DOF
            fixed2.append(2 * n + 1) # y DOF

        fixed = np.concatenate([fixed1, fixed2])

    elif example == 'EX2D4':
        indices_force = (Nely + 1) * Nelx # indices in the global node vector
        force[2*indices_force+1, 0 ] = -0.1
        # define displacement boundary condition
        set1 = np.arange(2 * (Nely + 1)) # fix the left edge x disp
        set2 = np.array([ ], dtype=int) # fix the bottom right corner y disp
        fixed = np.union1d(set1, set2)
    elif example == 'EX2D5':
        # define the force vector
        force = np.zeros((ndof,1))
        indices_force = np.where((nodeXY[:, 0] == 100) & (nodeXY[:, 1] == 40))[0] # indices in the global node vector
        # 0 -> 0,1
        # 1 -> 2,3
        # 2 -> 4,5
        # n -> 2*n, 2*n+1
        force[2*indices_force+1, 0 ] = -0.1
        # define displacement boundary condition
        top_edge = np.where((nodeXY[:, 0] >= 0) & (nodeXY[:, 0] <= 40) & (nodeXY[:, 1] == 100))[0] # indices in the global node vector
        fixed = []
        for n in top_edge:
            fixed.append(2 * n)     # x DOF
            fixed.append(2 * n + 1) # y DOF

        fixed = np.array(fixed)
    else:
        pass

    symXAxis = {'isOn':False, 'midPt':0.5*Nely}
    symYAxis = {'isOn':False, 'midPt':0.5*Nelx}
    bc = {'exampleName':exampleName, 'physics':physics,
        'force':force, 'fixed':fixed, 'symXAxis':symXAxis, 'symYAxis':symYAxis}

    FE_force = FE(mesh, matProp, bc)
    elem_coord = FE_force.elemCenters # coordinate for element centers
    rho_elem = griddata(points=X_col, values=rho, xi=elem_coord, method='linear')
    u, c0_elem = FE_force.solve(rho_elem)
    comp = np.sum(Emax*(rho_elem**p)*c0_elem)
    return comp

blue_dark = hex_to_rgba("#3b4cc0")
blue_light = hex_to_rgba("#8db0fe")
red_dark = hex_to_rgba("#b40426")
red_light = hex_to_rgba("#f4987a")
green_dark = hex_to_rgba("#33a645")
green_light = hex_to_rgba("#afd991")

tkwargs = get_tkwargs()

pad = 2
example = 'EX2D5'
type = 'proj' # proj or no_proj
iter = [9999, 19999] # iteration to plot densities, the last one must be the final one
kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel for defining boundary pixels
Emax = 1
nu = 0.3
p = 3

# assign parameters for each example
if example == 'EX2D1':
    elem_num = [150,50] # make sure these two values are consistent with those in Step1
    x = [0,150]
    y = [0,50]
    VF = 0.5 # this is the actual volume fraction
    VF_name = 0.5 # this is the volume fraction in the mat filename
elif example == 'EX2D2':
    elem_num = [160,100]
    x = [0,160]
    y = [0,100]
    VF = 0.3 # this is the actual volume fraction
    VF_name = 0.3 # this is the volume fraction in the mat filename
elif example == 'EX2D3':
    elem_num = [200,100]
    x = [0,200]
    y = [0,100]
    VF = 0.3 # this is the actual volume fraction
    VF_name = 0.3 # this is the volume fraction in the mat filename
elif example == 'EX2D4':
    elem_num = [150,100]
    x = [0,150]
    y = [0,100]
    VF = 0.5 # this is the actual volume fraction
    VF_name = 0.5 # this is the volume fraction in the mat filename
    hole_center = [50.0,50.0]
    hole_radius = 100/3
elif example == 'EX2D5':
    elem_num = [100,100]
    x = [0,100]
    y = [0,100]
    VF = 0.32 # this is the actual volume fraction
    VF_name = 0.5 # this is the volume fraction in the mat filename
else:
    pass
domain = {'x':x,'y':y}
xmin, xmax,ymin,ymax = domain['x'][0], domain['x'][1], domain['y'][0], domain['y'][1]
xi = np.linspace(xmin, xmax, num=elem_num[0]+1)
yi = np.linspace(ymin, ymax, num=elem_num[1]+1)
dx = xi[1] - xi[0]
dy = yi[1] - yi[0]
MP = {'domain':domain,'Emax':Emax,'nu':nu,'p':p}

# get the save folder for the data
if type == 'proj':
    save_folder = 'Results/' + example + '/' + f'VF{VF_name}_proj' + '/'
else:
    save_folder = 'Results/' + example + '/' + f'VF{VF_name}' + '/'

# generate the coordinates of element centers, consistent with SIMP
Nelx = elem_num[0]
Nely = elem_num[1]
xi_simp = np.zeros((Nely, Nelx))
yi_simp = np.zeros((Nely, Nelx))

for i in range(Nelx):         # range goes from 0 to Nelx-1
    for j in range(Nely):     # range goes from 0 to Nely-1
        xi_simp[j, i] = i + 0.5
        yi_simp[j, i] = Nely - (j + 0.5)
xi_simp = xi_simp.T.flatten()
yi_simp = yi_simp.T.flatten()
X_col_elem = np.vstack([xi_simp, yi_simp]).T
xi_simp_grid = xi_simp.reshape(Nelx, Nely).T
yi_simp_grid = yi_simp.reshape(Nelx, Nely).T

# delete points in unwanted area: make sure this is also consistent with Step1
if example == 'EX2D1' or example == 'EX2D2' or example == 'EX2D3':
    distance = ((xi_simp_grid) ** 2 + (yi_simp_grid) ** 2) ** 0.5
    mask_domain =  distance >= -1
    mask_col_elem = mask_domain
    mask_nan_elem = ~mask_col_elem
elif example == 'EX2D4':
    distance = ((xi_simp_grid - hole_center[0]) ** 2 + (yi_simp_grid - hole_center[1]) ** 2) ** 0.5
    mask_domain =  distance >= hole_radius
    mask_col_elem = mask_domain
    mask_nan_elem = ~mask_col_elem
elif example == 'EX2D5':
    mask_domain =  (xi_simp_grid <= 40) | (yi_simp_grid <= 40)
    mask_col_elem = mask_domain
    mask_nan_elem = ~mask_col_elem
else:
    pass

mask_col_elem = mask_col_elem.T.flatten()
mask_nan_elem = mask_nan_elem.T.flatten()
X_elem = {'X_col_elem':X_col_elem,'mask_col_elem':mask_col_elem,
          'mask_nan_elem':mask_nan_elem,'elem_num':elem_num}

# iterate all results
boundary_elem_summary = []
grey_elem_summary = []
grey_elem_summary_wide = []
compliance_elem_summary = []
compliance_elem_bw_summary = []
VF_elem_summary = []
VF_elem_bw_summary = []
thres_elem_summary = []
for i in range(1, 11):  # Iterate from 1 to 11
    ############Process time history##############
    run_folder = save_folder + f'Run_{i}/'
    with open(run_folder + "timeHistory.json", "r") as file:
        timeHistory = json.load(file)
    
    loss_compliance = timeHistory['loss_compliance']
    loss_dem = timeHistory['loss_dem']
    loss_volConstraint = timeHistory['loss_volConstraint']
    SE = timeHistory['strain_energy']
    EW = timeHistory['external_work']
    SE_smoothed = moving_average(np.array(SE), window_size=10)
    loss_dem_smoothed = moving_average(np.array(loss_dem), window_size=10)
    loss_volConstraint_smoothed = moving_average(np.array(loss_volConstraint), window_size=10)
    vol = timeHistory['vol']
    grey = timeHistory['grey']

    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'Times New Roman',
        'axes.labelsize': 20,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,  # Set legend font size
    })
    plt.rcParams["mathtext.fontset"] = "stix"
    curve_linewidth = 2
    legend_linewidth = 1
    anno_fontsize = 20

    # Create the plot with size (8, 6)
    fig, ax = plt.subplots(figsize=(8, 6))
    anno_loc = [-0.15,0.99]
    xlim = [0,20000]
    ylim = [1e-12,100]
    # Plot 2 * SE and EW
    ax.semilogy(loss_compliance, label=r"Loss Compliance", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
    ax.semilogy(loss_dem, color=green_light, linestyle='-', linewidth=curve_linewidth)
    ax.semilogy(loss_dem_smoothed, label=r"Loss DEM", color=green_dark, linestyle='-', linewidth=curve_linewidth)
    ax.semilogy(loss_volConstraint, color=red_light, linestyle='-', linewidth=curve_linewidth)
    ax.semilogy(loss_volConstraint_smoothed, label=r"Loss Volume Fraction", color=red_dark, linestyle='-', linewidth=curve_linewidth)

    # Adding titles and labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss (mJ)")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.text(anno_loc[0], anno_loc[1], "(a)", transform=ax.transAxes, verticalalignment='top', fontsize = anno_fontsize,horizontalalignment='left')

    # Add a legend
    legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
    legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box

    # Save the plot
    file_name = "TO_evolution_a.pdf"
    file_path = f"{run_folder}/{file_name}"
    plt.savefig(file_path, format='pdf', dpi=600)


    # Create the plot (b)
    fig, ax = plt.subplots(figsize=(8, 6))
    anno_loc = [-0.1,0.99]
    xlim = [0,20000]
    ylim = [0,7]
    # Plot 2 * SE and EW
    ax.plot(2 * np.array(SE), color=blue_light, linestyle='-', linewidth=curve_linewidth)
    ax.plot(2 * SE_smoothed, label=r"$2 \times$ Strain Energy", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
    ax.plot(EW, label="External Work", color=red_dark, linestyle='-', linewidth=curve_linewidth)

    # Adding titles and labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Energy (mJ)")
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    ax.text(anno_loc[0], anno_loc[1], "(b)", transform=ax.transAxes, verticalalignment='top', fontsize = anno_fontsize,horizontalalignment='left')

    # Add a legend
    legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
    legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box

    # Save the plot
    file_name = "TO_evolution_b.pdf"
    file_path = f"{run_folder}/{file_name}"
    plt.savefig(file_path, format='pdf', dpi=600)

    # Show the plot
    # plt.show()

    # Create the plot with size (8, 6)
    fig, ax = plt.subplots(figsize=(8, 6))
    anno_loc = [-0.15,0.99]
    xlim = [0,20000]
    ylim = [0,1]
    # Plot 2 * SE and EW
    ax.plot(vol, label=r"Volume Fraction", color=blue_dark, linestyle='-', linewidth=curve_linewidth)
    ax.plot(grey, label=r"Grey Area Fraction", color=red_dark, linestyle='-', linewidth=curve_linewidth)

    # Adding titles and labels
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Fraction")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.text(anno_loc[0], anno_loc[1], "(c)", transform=ax.transAxes, verticalalignment='top', fontsize = anno_fontsize,horizontalalignment='left')

    # Add a legend
    legend = ax.legend(frameon=True, facecolor='white', edgecolor='black', framealpha=1)
    legend.get_frame().set_linewidth(legend_linewidth)  # Set the width of the legend box

    # Save the plot
    file_name = "TO_evolution_c.pdf"
    file_path = f"{run_folder}/{file_name}"
    plt.savefig(file_path, format='pdf', dpi=600)

    # Show the plot
    # plt.show()

    ############Process element density##############
    data_list = []
    for j, i in enumerate(iter):
        # Extract data for each iteration
        data_elem = extract_rho(run_folder, i, type, X_elem)
        elem_x = data_elem['elem_x']
        rho_elem = data_elem['rho']
        rho_elem_grid = rho_elem.reshape(Nelx, Nely).T

        #remove the padding layer:
        data_list.append({
            'rho_elem': rho_elem.tolist(),
            'xi_elem': elem_x[:,0].tolist(),
            'yi_elem': elem_x[:,1].tolist(),
            'xi_elem_grid':xi_simp_grid.tolist(),
            'yi_elem_grid':yi_simp_grid.tolist(),
            'rho_elem_grid':rho_elem_grid.tolist(),
        })
    
    # fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    # # axs = np.atleast_2d(axs)
    # contour = axs.contourf(xi_simp_grid, yi_simp_grid, rho_elem_grid, levels=200, cmap=cmap_density, vmin=0, vmax=1)
    # contour.set_rasterized(True)
    # axs.set_title(f'SIMP')
    # axs.set_aspect('equal')
    # axs.set_xticks([])  # Hide x ticks
    # axs.set_yticks([])  # Hide y ticks
    # axs.set_xlabel('')  # Hide x label
    # axs.set_ylabel('')  # Hide y label
    # axs.axis('off')     # Remove bounding box
    # plt.colorbar(contour, ax=axs, label="Density")
    # plt.show()

    # save the rho
    file_path = run_folder + 'rho_elem.json'        
    with open(file_path, 'w') as f:
        json.dump(data_list, f, indent=4)

    # calculate compliance using elemental data directly
    xi_elem = data_list[-1]['xi_elem']
    yi_elem = data_list[-1]['yi_elem']
    rho_elem = np.array(data_list[-1]['rho_elem'])
    elem_x = np.vstack([xi_elem, yi_elem]).T
    comp = FE_eval(elem_x,rho_elem,MP,example)
    compliance_elem_summary.append(comp)
    
    # calculate grey area fraction from element
    rho_np = rho_elem
    grey_elements = (rho_np > 0.1) & (rho_np < 0.9)
    grey_elem = np.sum(grey_elements) / len(rho_np)
    grey_elem_summary.append(grey_elem)

    grey_elements = (rho_np > 0.05) & (rho_np < 0.95)
    grey_elem = np.sum(grey_elements) / len(rho_np)
    grey_elem_summary_wide.append(grey_elem)

    # calculate the actual volume fraction
    VF_elem_summary.append(np.mean(rho_elem))

    # Create the plot with size (8, 6)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    num_levels = 100

    # Plot for the first sub-plot
    contour = axs[0].contourf(data_list[0]['xi_elem_grid'], data_list[0]['yi_elem_grid'], data_list[0]['rho_elem_grid'], levels=num_levels, cmap=cmap_density, vmin=0, vmax=1)
    contour.set_rasterized(True)
    axs[0].set_title(f'Epoch: {iter[0]}')
    axs[0].set_aspect('equal')
    axs[0].set_xticks([])  # Hide x ticks
    axs[0].set_yticks([])  # Hide y ticks
    axs[0].set_xlabel('')  # Hide x label
    axs[0].set_ylabel('')  # Hide y label
    axs[0].axis('off')     # Remove bounding box

    # Plot for the second sub-plot
    contour = axs[1].contourf(data_list[1]['xi_elem_grid'], data_list[1]['yi_elem_grid'], data_list[1]['rho_elem_grid'], levels=num_levels, cmap=cmap_density, vmin=0, vmax=1)
    contour.set_rasterized(True)
    axs[1].set_title(f'Epoch: {iter[1]}')
    axs[1].set_aspect('equal')
    axs[1].set_xticks([])  # Hide x ticks
    axs[1].set_yticks([])  # Hide y ticks
    axs[1].set_xlabel('')  # Hide x label
    axs[1].set_ylabel('')  # Hide y label
    axs[1].axis('off')     # Remove bounding box

    # Add annotation (d) in the top left corner of the last subplot
    fig.text(anno_loc[0], anno_loc[1], "(d)", transform=fig.transFigure, verticalalignment='top', fontsize=anno_fontsize, horizontalalignment='left')

    # Create the colorbar with manual positioning
    cbar_axes = fig.add_axes([0.005, 0.15, 0.03, 0.65])  # Position [left, bottom, width, height]
    cbar = fig.colorbar(contour, cax=cbar_axes)  # Use cax for colorbar
    cbar.set_ticks(np.arange(0, 1.1, 0.2))  # Set ticks on the colorbar

    # Adjust the layout manually (to prevent tight_layout from overriding your adjustments)
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Save the plot
    file_name = "TO_evolution_d.pdf"
    file_path = f"{run_folder}/{file_name}"
    plt.savefig(file_path, format='pdf', dpi=600)

    # Show the plot
    # plt.show()
    
    # binarize the field:
    rho_elem_grid = np.array(data_list[-1]['rho_elem_grid'])
    rho_elem_bw_grid, dynamic_thres_elem = dynamic_binarize_density(rho_elem_grid,VF)

    # calculate the actual volume fraction
    VF_elem_bw_summary.append(np.mean(rho_elem_bw_grid))
    thres_elem_summary.append(np.mean(dynamic_thres_elem))

    # calculate the BW compliance:
    elem_x = np.vstack([xi_elem, yi_elem]).T
    rho_elem_bw = rho_elem_bw_grid.T.flatten()
    comp = FE_eval(elem_x,rho_elem_bw,MP,example)
    compliance_elem_bw_summary.append(comp)

    fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    # axs = np.atleast_2d(axs)
    contour = axs[0].contourf(data_list[-1]['xi_elem_grid'], data_list[-1]['yi_elem_grid'], rho_elem_grid, levels=num_levels, cmap=cmap_density, vmin=0, vmax=1)
    contour.set_rasterized(True)
    axs[0].set_title(f'Original')
    axs[0].set_aspect('equal')
    axs[0].set_xticks([])  # Hide x ticks
    axs[0].set_yticks([])  # Hide y ticks
    axs[0].set_xlabel('')  # Hide x label
    axs[0].set_ylabel('')  # Hide y label
    axs[0].axis('off')     # Remove bounding box

    contour = axs[1].contourf(data_list[-1]['xi_elem_grid'], data_list[-1]['yi_elem_grid'], rho_elem_bw_grid, levels=num_levels, cmap=cmap_density, vmin=0, vmax=1)
    contour.set_rasterized(True)
    axs[1].set_title(f'BW')
    axs[1].set_aspect('equal')
    axs[1].set_xticks([])  # Hide x ticks
    axs[1].set_yticks([])  # Hide y ticks
    axs[1].set_xlabel('')  # Hide x label
    axs[1].set_ylabel('')  # Hide y label
    axs[1].axis('off')     # Remove bounding box
    # Save the plot
    file_name = "rho_BW.pdf"
    file_path = f"{run_folder}/{file_name}"
    plt.savefig(file_path, format='pdf', dpi=600)
    # plt.show()
    # plt.close()
    
    # calculate the boundary pixes
    ny, nx = rho_elem_bw_grid.shape   # element counts
    rho_bw_elem_grid_extended = np.pad(rho_elem_bw_grid, pad_width=pad, mode='constant', constant_values=0)
    rho_bw_elem_grid_extended = rho_bw_elem_grid_extended.astype(np.uint8)    
    x_extended = np.linspace(xmin - pad*dx, xmax + pad*dx, nx+2*pad)  # padd two layers
    y_extended = np.linspace(ymax + pad*dy, ymin - pad*dy, ny+2*pad)  # decreasing order
    xi_elem_extended, yi_elem_extended = np.meshgrid(x_extended, y_extended)
    rho_eroded = cv2.erode(rho_bw_elem_grid_extended, kernel, iterations=1)  # Erode solid regions
    boundary_pixels_elem = (rho_bw_elem_grid_extended - rho_eroded).astype(np.uint8)
    
    # Calculate the fraction of boundary pixels
    rho_bw_elem_grid = rho_bw_elem_grid_extended[pad:-pad, pad:-pad]
    total_solid_pixels = np.sum(rho_bw_elem_grid)  # Count total solid pixels
    boundary_count = np.sum(boundary_pixels_elem)  # Count boundary pixels
    boundary_fraction = boundary_count / total_solid_pixels if total_solid_pixels > 0 else np.nan
    boundary_elem_summary.append(boundary_fraction)

    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    # axs = np.atleast_2d(axs)
    contour = axs.contourf(xi_elem_extended, yi_elem_extended, rho_bw_elem_grid_extended, levels=100, cmap=cmap_density, vmin=0, vmax=1)
    axs.contourf(xi_elem_extended, yi_elem_extended, boundary_pixels_elem, levels=[0.5, 1], colors='red', alpha=0.7)
    contour.set_rasterized(True)
    axs.set_title(f'Boundary pixels on elements')
    axs.set_aspect('equal')
    axs.set_xticks([])  # Hide x ticks
    axs.set_yticks([])  # Hide y ticks
    axs.set_xlabel('')  # Hide x label
    axs.set_ylabel('')  # Hide y label
    axs.axis('off')     # Remove bounding box

    # Save the plot
    file_name = "rho_boundary.pdf"
    file_path = f"{run_folder}/{file_name}"
    plt.savefig(file_path, format='pdf', dpi=600)
    # plt.show()
    # plt.close()

with open(save_folder + 'boundary_elem_summary.json', 'w') as f:
    json.dump(boundary_elem_summary, f, indent=4)

with open(save_folder + 'grey_elem_summary.json', 'w') as f:
    json.dump(grey_elem_summary, f, indent=4)

with open(save_folder + 'grey_elem_summary_wide.json', 'w') as f:
    json.dump(grey_elem_summary_wide, f, indent=4)

with open(save_folder + 'compliance_elem_summary.json', 'w') as f:
    json.dump(compliance_elem_summary, f, indent=4)

with open(save_folder + 'compliance_elem_bw_summary.json', 'w') as f:
    json.dump(compliance_elem_bw_summary, f, indent=4)

with open(save_folder + 'thres_elem_summary.json', 'w') as f:
    json.dump(thres_elem_summary, f, indent=4)

with open(save_folder + 'VF_elem_bw_summary.json', 'w') as f:
    json.dump(VF_elem_bw_summary, f, indent=4)

with open(save_folder + 'VF_elem_summary.json', 'w') as f:
    json.dump(VF_elem_summary, f, indent=4)


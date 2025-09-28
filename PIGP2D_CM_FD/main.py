import os
import json
import torch
import numpy as np
import time
import shutil
from models.lmgp import LMGP 
from tqdm import tqdm
from utils.utils_general import set_seed0, get_tkwargs, calculate_TO_loss
from utils.utils_general import compute_dynamic_weight_2 as compute_dynamic_weight

tkwargs = get_tkwargs()

############################### Define Parameters ##############################################
# define material properties and
Example = 'EX2D5'
Case = 'VF0.5'# any subfolder name 
Emax = 1
Emin = 1e-3
nu = 0.3 # Poisson's ratio
p = 3 # penalty
V_star0 = 0.6 # initial volume fracton
frac_decrease = 0.5 # fraction of epoch to decrease volume fraction from VF0 to VF_f
b = 8 # sharpness parameters
thres_static = 0.5 # static threshold values to binarize the density field
rho_min = 0.1 # lower limit to define grey element (rho_min,rho_max)
rho_max = 0.9 # upper limit to define grey element (rho_min,rho_max)

#define structured elements
pad = 2 # padding layer thickness
if Example == 'EX2D1': # MBB
    from utils.get_training_data import get_data_EX2D1 as get_data
    V_star_f = 0.5
    Nelx = 150 
    Nely = 50 
    Nelx_max = 300
    Nely_max = 100
    Nelx_min = 150
    Nely_min = 50
    xmin = 0.0
    xmax = 150.0
    ymin = 0.0
    ymax = 50.0
    hole_center = None
    hole_radius = None
elif Example == 'EX2D2': # Cantilever
    from utils.get_training_data import get_data_EX2D2 as get_data
    V_star_f = 0.3
    Nelx = 160 
    Nely = 100 
    Nelx_max = 320
    Nely_max = 200
    Nelx_min = 160
    Nely_min = 100
    xmin = 0.0
    xmax = 160.0
    ymin = 0.0
    ymax = 100.0
    hole_center = None
    hole_radius = None
elif Example == 'EX2D3': # Uniformly loaded
    from utils.get_training_data import get_data_EX2D3 as get_data
    V_star_f = 0.3
    Nelx = 200  
    Nely = 100 
    Nelx_max = 400
    Nely_max = 200
    Nelx_min = 200
    Nely_min = 100
    xmin = 0.0
    xmax = 200.0
    ymin = 0.0
    ymax = 100.0
    hole_center = None
    hole_radius = None
elif Example == 'EX2D4': # Hollow beam
    from utils.get_training_data import get_data_EX2D4 as get_data
    V_star_f = 0.5
    Nelx = 150 
    Nely = 100 
    Nelx_max = 300
    Nely_max = 200
    Nelx_min = 150
    Nely_min = 100
    xmin = 0.0
    xmax = 150.0
    ymin = 0.0
    ymax = 100.0
    hole_center = [50.0,50.0]
    hole_radius = 100/3
elif Example == 'EX2D5': #L-shape
    from utils.get_training_data import get_data_EX2D5 as get_data
    V_star_f = 0.5
    Nelx = 100 
    Nely = 100 
    Nelx_max = 200
    Nely_max = 200
    Nelx_min = 100
    Nely_min = 100
    num_CP = max([Nelx_max-Nelx_min,Nely_max-Nely_min])
    xmin = 0.0
    xmax = 100.0
    ymin = 0.0
    ymax = 100.0
    hole_center = None
    hole_radius = None
else:
    raise ValueError(f"No such example: {Example}")
num_CP = max([Nelx_max-Nelx_min,Nely_max-Nely_min])

# define model parameters and training parameters
random_state = [1,3,5,7,9,11]
dynamic_weight = False
gradient_clip = False
Diff_type = 'ND2' 
omega = 0.5
learning_rate = 1e-3
nrmThreshold = 0.1; # maximum allowable norm of the gradients, [-nrmThreshold,nrmThreshold]
TO_num_iter = 20000 # number of total topology optimization iteration
plot_num = 10 # number of plots to save
delta = 1e-1
wc = 1 # weight factor for loss_compliance
wd = 1e2 # weight factor for loss_dem
wv = 1e3 # weight factor for loss_VF
basis = 'PGCAN'#'PGCAN'#'neural_network' #'M3'  
quant_correlation_class = 'Rough_RBF' 
activation = 'tanh'
if basis == 'PGCAN':
    n_features = 128 # must be a even number, only used for PGCAN
    n_cells = 3 # no more than 4, only used for PGCAN
    factor = 36
    res = [int(factor * Nely / 100), int(factor * Nelx / 100)]
    n_neurons = int(n_features/2)
    n_layers = 3 # 3 layers at most for PGCAN
    NN_arch = [n_neurons] * n_layers 
else:
    n_features = None #only used for PGCAN
    n_cells = None #only used for PGCAN
    res = None # only for PGCAN
    n_neurons = 64
    n_layers = 6
    NN_arch = [n_neurons] * n_layers 

############################### Pre-processing ##############################################
V_decrease = int(frac_decrease * TO_num_iter)
plotting_interval_TO = TO_num_iter/plot_num
V_step = (V_star0 - V_star_f)/V_decrease
domain = {'x':[xmin, xmax], 'y':[ymin, ymax]}
MP = {'pad':pad,'Nelx': Nelx, 'Nely': Nely,'Nelx_max': Nelx_max, 'Nely_max': Nely_max, 'Nelx_min': Nelx_min, 'Nely_min': Nely_min, 'num_CP':num_CP,
      'domain': domain,'thres_static':thres_static,'hole_center':hole_center,'hole_radius':hole_radius,
      'Emax': Emax, 'Emin':Emin, 'nu': nu,'p':p, 'b':b,'rho_min':rho_min,'rho_max':rho_max,
      'V_star':V_star0, 'V_star0':V_star0, 'V_star_f':V_star_f,'V_decrease':V_decrease,'V_step':V_step}
NN_config = {'random_state':random_state,'state':[],'Example':Example,'Case':Case,
             'dynamic_weight':dynamic_weight,'gradient_clip':gradient_clip,'Diff_type':Diff_type,
             'omega':omega,'learning_rate':learning_rate,'nrmThreshold':nrmThreshold,
             'TO_num_iter':TO_num_iter,'plot_num':plot_num,'plotting_interval_TO':plotting_interval_TO,
             'delta':delta,'wc':wc,'wd':wd,'wv':wv,
             'basis': basis, 'quant_correlation_class': quant_correlation_class, 'activation': activation,
             'n_features': n_features, 'n_cells': n_cells, 'res': res,'NN_arch': NN_arch, 'save_folder': []}

############################### Generate Data ##############################################
Training, X_col_all = get_data(MP)
X_col = Training['X_col'].type(tkwargs["dtype"]).requires_grad_(True).to(tkwargs['device'])
u_X_train = Training['u_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
v_X_train = Training['v_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_X_train = Training['rho_X_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
u_train = Training['u_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
v_train = Training['v_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])
rho_train = Training['rho_train'].type(tkwargs["dtype"]).requires_grad_(False).to(tkwargs['device'])

# Generate random integers
index_CP = np.random.randint(0, num_CP, size=TO_num_iter)

############################### TO Loop ##########################################
base_folder = f"Results/{Example}/{Case}/"
# Create folders for each random state
for i, state in enumerate(random_state, start=1):
    # reset V_star
    MP['V_star'] = V_star0
    
    # set the random seed number:
    set_seed0(state)
    NN_config['state'] = state

    # create the folder
    save_folder = f"{base_folder}Run_{i}/"
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)
    NN_config['save_folder'] = save_folder

    # define the models: these models shared the same NN (using model_u)
    model_u = LMGP(train_x = u_X_train, train_y = u_train, collocation_x = X_col,
                NN_config = NN_config,
                Training = Training,
                name_output='u',
                MP=MP,
                num_output = 3).to(tkwargs['device'])

    model_v = LMGP(train_x = v_X_train, train_y = v_train, collocation_x = X_col,
                NN_config = NN_config,
                Training = Training,
                name_output='v',
                MP=MP,
                num_output = 3).to(tkwargs['device'])

    model_rho = LMGP(train_x = rho_X_train, train_y = rho_train, collocation_x = X_col,
                NN_config = NN_config,
                Training = Training,
                name_output='rho',
                MP=MP,
                num_output = 3).to(tkwargs['device'])


    model_list = [model_u, model_v, model_rho]

    # define the time history dict
    timeHistory = {'loss_total':[],'loss_compliance':[],'loss_dem':[],'loss_volConstraint':[],'loss_tv':[],
               'strain_energy':[],'external_work':[],'loss_pde_1':[],'loss_pde_2':[],'vol':[],'wc':[], 'wd':[], 'wv':[],'grey':[]}  
    
    # define optimizer for just the density model
    optimizer_TO = torch.optim.Adam(model_list[0].parameters(),
            lr=learning_rate,
            amsgrad=True)

    # Learning rate scheduler
    scheduler_TO = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_TO, 
            milestones=np.linspace(0, TO_num_iter, 4).tolist(), 
            gamma=0.75)

    # Training loop
    epochs_iter = tqdm(range(TO_num_iter), desc='TO Epoch', position=0, leave=True)
    total_time = 0
    start_time = time.time()  # get the current time
    for epoch in epochs_iter:
        # zero gradients from previous iteration
        optimizer_TO.zero_grad()

        # update parameters and collocation points
        index = index_CP[epoch]
        model_list[0].dx = X_col_all[index]['dx']
        model_list[0].dy = X_col_all[index]['dy']
        model_list[0].Nx = X_col_all[index]['Nx']
        model_list[0].Ny = X_col_all[index]['Ny']
        model_list[0].mask_col = X_col_all[index]['mask_col']
        model_list[0].collocation_x = X_col_all[index]['X_col']
        model_list[0].traction_indices = X_col_all[index]['traction_indices']
        model_list[0].traction_magnitude = X_col_all[index]['traction_magnitude']
        
        # calculate DEM loss: (detach rho_element, used to monior loss_compliance_fem)
        loss_pde1, loss_pde2, strain_energy, external_work, loss_compliance, loss_volConstraint, volfrac, grey_fraction = calculate_TO_loss(model_list,Diff_type)

        offset = (1 + delta)/2 * external_work.detach()
        loss_dem = (strain_energy - external_work + offset)
        
        if dynamic_weight:
            alpha = compute_dynamic_weight(loss_dem, loss_volConstraint, model_list)
            if torch.is_tensor(alpha):
                if not torch.isnan(alpha).any() and not torch.isinf(alpha).any():
                    model_list[2].alpha = alpha.item()

        loss =  wc * loss_compliance + wd * loss_dem + wv * loss_volConstraint
        loss.backward(retain_graph=True)
        # Clip gradients to avoid NaN or inf values
        if gradient_clip:
            torch.nn.utils.clip_grad_norm_(model_list[0].parameters(), nrmThreshold)

        optimizer_TO.step()
        scheduler_TO.step()
        
        # save the loss histories and plot
        timeHistory['loss_total'].append(loss.item())
        timeHistory['loss_compliance'].append(loss_compliance.item())
        timeHistory['loss_dem'].append(loss_dem.item())
        timeHistory['loss_volConstraint'].append(loss_volConstraint.item())
        timeHistory['vol'].append(volfrac.item())
        timeHistory['strain_energy'].append(strain_energy.item())
        timeHistory['external_work'].append(external_work.item())
        timeHistory['loss_pde_1'].append(loss_pde1.item())
        timeHistory['loss_pde_2'].append(loss_pde2.item())
        timeHistory['wc'].append(wc)
        timeHistory['wd'].append(wd)
        timeHistory['wv'].append(wv)
        timeHistory['grey'].append(grey_fraction)
        
        # visualize contours and history
        if  (epoch+1) % plotting_interval_TO == 0:
            end_time = time.time() 
            torch.save(model_list[0].mean_module_NN_All.state_dict(),save_folder+f'Trained_mean_module_NN_params_{epoch}.pth')

            total_time = total_time + (end_time - start_time)
            start_time = time.time()

        model_list[0].MP['V_star'] = max(model_list[0].MP['V_star_f'], model_list[0].MP['V_star'] - model_list[0].MP['V_step'])
    
    end_time = time.time()
    total_time = total_time + (end_time - start_time)

    # save the MP to a JSON file
    with open(save_folder + "MP.json", "w") as file:
        json.dump(MP, file, indent=4)
    
    # save the NN_config to a JSON file
    with open(save_folder + "NN_config.json", "w") as file:
        json.dump(NN_config, file, indent=4)

    # save the Training to file
    torch.save(Training, save_folder + "Training.pth")

    # save the time history
    with open(save_folder + "timeHistory.json", "w") as file:
        json.dump(timeHistory, file)

    # save the final model
    torch.save(model_list[0].mean_module_NN_All.state_dict(),save_folder+'Trained_mean_module_NN_params.pth')

    # Write the loss terms to the file
    result_file_path = f"{save_folder}Results_summary.txt"
    with open(result_file_path, 'w') as f:
        f.write(f"The total training time in second is: {total_time}\n")
        f.write(f"Final value strain energy: {timeHistory['strain_energy'][-1]}\n")
        f.write(f"Final value external work: {timeHistory['external_work'][-1]}\n")
        f.write(f"Final value Loss_total: {timeHistory['loss_total'][-1]}\n\n")
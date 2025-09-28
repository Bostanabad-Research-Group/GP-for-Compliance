import torch
import numpy as np
import json
import random
from gpytorch.settings import cholesky_jitter

# def set_seed(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     np.random.seed(seed)
def set_seed0(seed):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          # PyTorch CPU seed
    if torch.cuda.is_available():    # If CUDA is available
        torch.cuda.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)

def set_seed(seed):
    random.seed(seed)                
    np.random.seed(seed)             
    torch.manual_seed(seed)          # PyTorch CPU seed
    if torch.cuda.is_available():    # If CUDA is available
        torch.cuda.manual_seed(seed)            
        torch.cuda.manual_seed_all(seed)        
    torch.backends.cudnn.deterministic = True   
    torch.backends.cudnn.benchmark = False
    # Set deterministic behavior for other PyTorch operations 
    torch.use_deterministic_algorithms(True)

def load_job_tensors_from_json(file_path, job_name):
    """
    Loads tensors for a specific job from a JSON file and converts them back to PyTorch tensors.
    
    Args:
    - file_path (str): Path to the JSON file.
    - job_name (str): The name of the job to extract tensors for (e.g., 'Job-1').

    Returns:
    - dict: A dictionary where keys are tensor names and values are PyTorch tensors.
    """
    with open(file_path, 'r') as json_file:
        jobs_dict_serializable = json.load(json_file)
    
    # Extract the dictionary for the specific job
    if job_name in jobs_dict_serializable:
        job_tensors_serializable = jobs_dict_serializable[job_name]
        # Convert lists back to tensors
        job_tensors = {key: torch.tensor(value) for key, value in job_tensors_serializable.items()}
        return job_tensors
    else:
        raise ValueError(f"No tensors found for {job_name}")

def get_tkwargs():
    """
    Returns a dictionary with the device and dtype configuration
    for PyTorch, prioritizing CUDA, then MPS (Mac GPU), and finally CPU.
    """
    # Determine the device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device("cpu")  # Use NVIDIA GPU if available
    elif torch.backends.mps.is_available():
        device = torch.device("cpu")  # Use Mac GPU (MPS) if available
    else:
        device = torch.device("cpu")  # Fallback to CPU
    
    # Define tkwargs with selected device and dtype
    tkwargs = {
        "dtype": torch.float32,
        "device": device,
    }
    
    #print(f"Using device: {device}")
    return tkwargs 

def central_diff_2nd(f, dx, dy):
    # f is a 2D torch tensor.
    # find the derivative using the 2nd order accuracy central difference method
    # First-order derivatives with second-order accuracy
    df_dx = (torch.roll(f, shifts=-1, dims=1) - torch.roll(f, shifts=1, dims=1)) / (2 * dx)
    df_dy = (torch.roll(f, shifts=-1, dims=0) - torch.roll(f, shifts=1, dims=0)) / (2 * dy)
    
    # Second-order derivatives with second-order accuracy
    d2f_dx2 = (torch.roll(f, shifts=-1, dims=1) - 2 * f + torch.roll(f, shifts=1, dims=1)) / (dx ** 2)
    d2f_dy2 = (torch.roll(f, shifts=-1, dims=0) - 2 * f + torch.roll(f, shifts=1, dims=0)) / (dy ** 2)
    
    # Mixed second derivative with second-order accuracy
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)) / (4 * dx * dy)
    
    return [df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]

def central_diff_4th(f, dx, dy):
    # f is a 2D torch tensor.
    # find the derivative using the 2nd order accuracy central difference method
    # First-order derivatives with fourth-order accuracy
    df_dx = (-torch.roll(f, shifts=-2, dims=1) 
             + 8 * torch.roll(f, shifts=-1, dims=1)
             - 8 * torch.roll(f, shifts=1, dims=1) 
             + torch.roll(f, shifts=2, dims=1)) / (12 * dx)
    
    df_dy = (-torch.roll(f, shifts=-2, dims=0) 
             + 8 * torch.roll(f, shifts=-1, dims=0)
             - 8 * torch.roll(f, shifts=1, dims=0) 
             + torch.roll(f, shifts=2, dims=0)) / (12 * dy)
    
    # Second-order derivatives with fourth-order accuracy
    d2f_dx2 = (-torch.roll(f, shifts=-2, dims=1) 
               + 16 * torch.roll(f, shifts=-1, dims=1)
               - 30 * f
               + 16 * torch.roll(f, shifts=1, dims=1) 
               - torch.roll(f, shifts=2, dims=1)) / (12 * dx**2)
    
    d2f_dy2 = (-torch.roll(f, shifts=-2, dims=0) 
               + 16 * torch.roll(f, shifts=-1, dims=0)
               - 30 * f
               + 16 * torch.roll(f, shifts=1, dims=0) 
               - torch.roll(f, shifts=2, dims=0)) / (12 * dy**2)
    
    # Mixed second derivative with fourth-order accuracy
    d2f_dxdy = (torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=-2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-2, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-2, dims=1)
                - torch.roll(torch.roll(f, shifts=2, dims=0), shifts=-2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=-1, dims=1)
                + 64 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=-1, dims=1)
                - 64 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=-1, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=2, dims=0), shifts=-1, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=1, dims=1)
                - 64 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=1, dims=1)
                + 64 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=1, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=2, dims=0), shifts=1, dims=1)
                - torch.roll(torch.roll(f, shifts=-2, dims=0), shifts=2, dims=1)
                + 8 * torch.roll(torch.roll(f, shifts=-1, dims=0), shifts=2, dims=1)
                - 8 * torch.roll(torch.roll(f, shifts=1, dims=0), shifts=2, dims=1)
                + torch.roll(torch.roll(f, shifts=2, dims=0), shifts=2, dims=1)) / (144 * dx * dy)
    
    return [df_dx, df_dy, d2f_dx2, d2f_dy2, d2f_dxdy]

def compute_dynamic_weights(ref_loss,target_loss,lambdaa,model,control):
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    delta_ref_teta = torch.autograd.grad(ref_loss, params_to_update,  retain_graph=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_ref_teta if p is not None]
    delta_ref_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    delta_target_teta = torch.autograd.grad(target_loss, params_to_update,  retain_graph=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_target_teta if p is not None]
    delta_target_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    temp1 = torch.mean(delta_ref_teta_abs) / torch.mean(delta_target_teta_abs)
    if control == 1:
        return (1.0 - lambdaa) * model.alpha + lambdaa * temp1
    else:
        return (1.0 - lambdaa) * model.beta + lambdaa * temp1

def calculate_TO_loss(model_list,Diff_type):
    MP = model_list[0].MP
    E_max = model_list[0].MP['Emax']
    E_min = model_list[0].MP['Emin']
    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    V_star = model_list[0].MP['V_star']
    dx = model_list[0].dx
    dy = model_list[0].dy
    Nx = model_list[0].Nx
    Ny = model_list[0].Ny
    mask_col = model_list[0].mask_col
    collocation_x = model_list[0].collocation_x.clone()

    for model in model_list:
        model.train

    m_col = model_list[0].mean_module_NN_All(collocation_x)
    
    g_u = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()
    g_v = (model_list[1].covar_module(model_list[1].train_inputs[0], collocation_x)).evaluate()
    g_rho = (model_list[2].covar_module(model_list[2].train_inputs[0], collocation_x)).evaluate()

    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(model_list[0].train_inputs[0]).cholesky()
            model_list[1].chol_decomp = model_list[1].covar_module(model_list[1].train_inputs[0]).cholesky()
            model_list[2].chol_decomp = model_list[2].covar_module(model_list[2].train_inputs[0]).cholesky()

    K_inv_offset_u = model_list[0].chol_decomp._cholesky_solve(model_list[0].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[0].train_inputs[0])[:,0].unsqueeze(-1))
    K_inv_offset_v = model_list[1].chol_decomp._cholesky_solve(model_list[1].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[1].train_inputs[0])[:,1].unsqueeze(-1))
    K_inv_offset_rho = model_list[2].chol_decomp._cholesky_solve(model_list[2].train_targets.unsqueeze(-1) - torch.sigmoid(model_list[0].mean_module_NN_All(model_list[2].train_inputs[0])[:,2].unsqueeze(-1)))

    u = (m_col[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u).squeeze(-1)
    v = (m_col[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v).squeeze(-1)
    rho_node = projectDensity( (torch.sigmoid(m_col[:,2].unsqueeze(-1)) + g_rho.t() @ K_inv_offset_rho).squeeze(-1), model_list[0].MP['b'] )

    rho_masked = rho_node[mask_col]
    volfrac = torch.mean(rho_masked); # get the current volume fraction
    loss_volConstraint = torch.square((volfrac / V_star) - 1.0) #+ 1e-5

    # calculate fraction of grey element
    rho_np = rho_masked.detach().cpu().numpy()
    grey_elements = (rho_np > MP['rho_min']) & (rho_np < MP['rho_max'])
    grey_fraction = np.sum(grey_elements) / len(rho_np)

    # calculate the external work
    v_load = v[model_list[0].traction_indices]
    external_work = torch.sum(v_load * model_list[0].traction_magnitude)
    
    rho_node_no_grad = rho_node.detach()
    E = E_min + rho_node_no_grad ** p * (E_max - E_min)
    E_hat = E_min + (rho_node_no_grad ** (2*p) / (rho_node + 1e-8) ** p) * (E_max - E_min)

    # use central difference method
    if Diff_type == 'AD': #autograd
        pass
    else: # use numerical differentiation
        if Diff_type == 'ND2': # central difference 2nd order
            du = central_diff_2nd(u.reshape(Nx, Ny).T, dx, dy)
            dv = central_diff_2nd(v.reshape(Nx, Ny).T, dx, dy)
        else: # central difference 4th order
            pass

        d2u_dx2 = du[2].T.flatten()
        d2u_dy2 = du[3].T.flatten()
        d2u_dxdy = du[4].T.flatten()
        d2v_dx2 = dv[2].T.flatten()
        d2v_dy2 = dv[3].T.flatten()
        d2v_dxdy = dv[4].T.flatten()

        e11 = du[0].T.flatten()
        e22 = dv[1].T.flatten()
        e12 = ((du[1] + dv[0])/2).T.flatten()
        e11_hat = e11.detach()
        e22_hat = e22.detach()
        e12_hat = e12.detach()

        s11_d = (E/(1-nu**2)) * (e11  + nu * e22)
        s22_d = (E/(1-nu**2)) * (nu * e11 + e22)
        s12_d = (E/(1+nu)) * (e12)
        s11_c = (E_hat/(1-nu**2)) * (e11_hat  + nu * e22_hat)
        s22_c = (E_hat/(1-nu**2)) * (nu * e11_hat + e22_hat)
        s12_c = (E_hat/(1+nu)) * (e12_hat)

        s11_1 = (E/(1-nu**2)) * (d2u_dx2 + nu * d2v_dxdy)
        s12_2 = (E/2/(1+nu)) * (d2u_dy2 + d2v_dxdy)
        s12_1 = (E/2/(1+nu)) * (d2u_dxdy + d2v_dx2)
        s22_2 = (E/(1-nu**2)) * (nu * d2u_dxdy + d2v_dy2)
    
    comp_vector = (s11_c * e11_hat + s22_c * e22_hat + 2 * s12_c * e12_hat)
    SE_vector = 0.5*(s11_d * e11 + s22_d * e22 + 2 * s12_d * e12)
    residual_pde1 = s11_1 + s12_2
    residual_pde2 = s12_1 + s22_2

    # get the data on valid collocation points
    comp_vector_masked = comp_vector[mask_col]
    SE_masked = SE_vector[mask_col]
    residual_pde1_masked = residual_pde1[mask_col]
    residual_pde2_masked = residual_pde2[mask_col]

    loss_pde1 = torch.mean(residual_pde1_masked**2)
    loss_pde2 = torch.mean(residual_pde2_masked**2)

    comp = comp_vector_masked.sum()
    loss_compliance =  comp / len(comp_vector_masked) * (model_list[0].domain_size[1] * model_list[0].domain_size[3])
    SE = SE_masked.sum() 
    strain_energy = SE / len(SE_masked) * (model_list[0].domain_size[1] * model_list[0].domain_size[3])

    return loss_pde1, loss_pde2, strain_energy, external_work, loss_compliance, loss_volConstraint, volfrac, grey_fraction

def eval_model(model_list,Diff_type):
    MP = model_list[0].MP
    E_max = model_list[0].MP['Emax']
    E_min = model_list[0].MP['Emin']
    nu = model_list[0].MP['nu']
    p = model_list[0].MP['p']
    mask_col = model_list[0].mask_col
    mask_nan = model_list[0].mask_nan
    xi = model_list[0].xi
    yi = model_list[0].yi
    for model in model_list:
        model.eval

    collocation_x = model_list[0].collocation_x.clone()
    m_col = model_list[0].mean_module_NN_All(collocation_x)

    g_u = (model_list[0].covar_module(model_list[0].train_inputs[0], collocation_x)).evaluate()
    g_v = (model_list[1].covar_module(model_list[1].train_inputs[0], collocation_x)).evaluate()
    g_rho = (model_list[2].covar_module(model_list[2].train_inputs[0], collocation_x)).evaluate()

    if model_list[0].chol_decomp is None:
        with cholesky_jitter(1e-5):
            model_list[0].chol_decomp = model_list[0].covar_module(model_list[0].train_inputs[0]).cholesky()
            model_list[1].chol_decomp = model_list[1].covar_module(model_list[1].train_inputs[0]).cholesky()
            model_list[2].chol_decomp = model_list[2].covar_module(model_list[2].train_inputs[0]).cholesky()

    K_inv_offset_u = model_list[0].chol_decomp._cholesky_solve(model_list[0].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[0].train_inputs[0])[:,0].unsqueeze(-1))
    K_inv_offset_v = model_list[1].chol_decomp._cholesky_solve(model_list[1].train_targets.unsqueeze(-1) - model_list[0].mean_module_NN_All(model_list[1].train_inputs[0])[:,1].unsqueeze(-1))
    K_inv_offset_rho = model_list[2].chol_decomp._cholesky_solve(model_list[2].train_targets.unsqueeze(-1) - torch.sigmoid(model_list[0].mean_module_NN_All(model_list[2].train_inputs[0])[:,2].unsqueeze(-1)))

    u = (m_col[:,0].unsqueeze(-1) + g_u.t() @ K_inv_offset_u).squeeze(-1)
    v = (m_col[:,1].unsqueeze(-1) + g_v.t() @ K_inv_offset_v).squeeze(-1)
    rho = projectDensity( (torch.sigmoid(m_col[:,2].unsqueeze(-1)) + g_rho.t() @ K_inv_offset_rho).squeeze(-1), model_list[0].MP['b'] )
    
    rho_no_grad = rho.detach()
    # make sure this line is consistent with the line in the comp
    E_hat = E_min + (rho_no_grad ** (2*p) / (rho + 1e-8) ** p) * (E_max - E_min)

    # use central difference method
    if Diff_type == 'AD': #autograd
        e11 = torch.autograd.grad(u, collocation_x, torch.ones_like(u), True, True)[0][:,0]
        e22 = torch.autograd.grad(v, collocation_x, torch.ones_like(u), True, True)[0][:,1]
        e12 = 0.5*(torch.autograd.grad(v, collocation_x, torch.ones_like(u), True, True)[0][:,0] +\
                torch.autograd.grad(u, collocation_x, torch.ones_like(u), True, True)[0][:,1])
        s11 = (E_hat/(1-nu**2)) * (e11  + nu * e22)
        s22 = (E_hat/(1-nu**2)) * (nu * e11 + e22)
        s12 = (E_hat/(1+nu)) * (e12) 
        ds11 = torch.autograd.grad(s11, collocation_x, torch.ones_like(e11), True, True)[0]
        ds22 = torch.autograd.grad(s22, collocation_x, torch.ones_like(e11), True, True)[0]
        ds12 = torch.autograd.grad(s12, collocation_x, torch.ones_like(e11), True, True)[0]
        s11_1 = ds11[:,0]
        s22_2 = ds22[:,1]
        s12_1 = ds12[:,0]
        s12_2 = ds12[:,1]
    else: # use numerical differentiation
        dx = model_list[0].dx
        dy = model_list[0].dy
        Nx = model_list[0].Nx
        Ny = model_list[0].Ny
        if Diff_type == 'ND2': # central difference 2nd order
            du = central_diff_2nd(u.reshape(Nx, Ny).T, dx, dy)
            dv = central_diff_2nd(v.reshape(Nx, Ny).T, dx, dy)
        else: # central difference 4th order
            pass

        d2u_dx2 = du[2].T.flatten()
        d2u_dy2 = du[3].T.flatten()
        d2u_dxdy = du[4].T.flatten()
        d2v_dx2 = dv[2].T.flatten()
        d2v_dy2 = dv[3].T.flatten()
        d2v_dxdy = dv[4].T.flatten()

        e11 = du[0].T.flatten()
        e22 = dv[1].T.flatten()
        e12 = ((du[1] + dv[0])/2).T.flatten()
        s11 = (E_hat/(1-nu**2)) * (e11  + nu * e22)
        s22 = (E_hat/(1-nu**2)) * (nu * e11 + e22)
        s12 = (E_hat/(1+nu)) * (e12)
        s11_1 = (E_hat/(1-nu**2)) * (d2u_dx2 + nu * d2v_dxdy)
        s12_2 = (E_hat/2/(1+nu)) * (d2u_dy2 + d2v_dxdy)
        s12_1 = (E_hat/2/(1+nu)) * (d2u_dxdy + d2v_dx2)
        s22_2 = (E_hat/(1-nu**2)) * (nu * d2u_dxdy + d2v_dy2)
    
    comp_vector = (s11 * e11 + s22 * e22 + 2 * s12 * e12)

    residual_pde1 = s11_1 + s12_2
    residual_pde2 = s12_1 + s22_2

    dC_drho = torch.autograd.grad(comp_vector, rho, torch.ones_like(comp_vector), create_graph=True)[0]

    residual_pde1 = residual_pde1[mask_col].detach().cpu().numpy()
    residual_pde2 = residual_pde2[mask_col].detach().cpu().numpy()
    u = u[mask_col].detach().cpu().numpy()
    v = v[mask_col].detach().cpu().numpy()
    e11 = e11[mask_col].detach().cpu().numpy()
    e22 = e22[mask_col].detach().cpu().numpy()
    e12 = e12[mask_col].detach().cpu().numpy()
    s11 = s11[mask_col].detach().cpu().numpy()
    s22 = s22[mask_col].detach().cpu().numpy()
    s12 = s12[mask_col].detach().cpu().numpy()
    rho = rho[mask_col].detach().cpu().numpy()
    collocation_x = collocation_x[mask_col].detach().cpu().numpy()
    comp_vector = comp_vector[mask_col].detach().cpu().numpy()
    dC_drho = dC_drho[mask_col].detach().cpu().numpy()

    Data = {'mask_nan':mask_nan,'xi':xi,'yi':yi,'collocation_x':collocation_x,'u':u,'v':v,'rho':rho,
            'residual_pde1':residual_pde1,'residual_pde2':residual_pde2,
            's11':s11,'s22':s22,'s12':s12,'e11':e11,'e22':e22,'e12':e12, 'comp_vector':comp_vector, 'dC_drho':dC_drho}
    return Data

lambdaa = 0.1
def compute_dynamic_weight_2(ref_loss,target_loss,model):
    params_to_update = [param for param in model.parameters() if param.requires_grad]

    delta_ref_teta = torch.autograd.grad(ref_loss, params_to_update,  retain_graph=True,allow_unused=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_ref_teta if p is not None]
    delta_ref_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    delta_target_teta = torch.autograd.grad(target_loss, params_to_update,  retain_graph=True,allow_unused=True)
    values = [p.reshape(-1,).cpu().tolist() for p in delta_target_teta if p is not None]
    delta_target_teta_abs = torch.abs(torch.tensor([v for val in values for v in val]))

    temp1 = torch.mean(delta_ref_teta_abs) / torch.mean(delta_target_teta_abs)
    
    return (1.0 - lambdaa) * model.alpha + lambdaa * temp1

def projectDensity(x,b=16):
    nmr = np.tanh(0.5*b) + torch.tanh(b*(x-0.5))
    x = 0.5*nmr/np.tanh(0.5*b)
    return x

def dynamic_binarize_density(density_field, tolerance=1e-6):
    # Original average density
    avg_density = np.mean(density_field)
    
    # Initialize threshold
    low, high = 0.0, 1.0
    
    while high - low > tolerance:
        threshold = (low + high) / 2
        
        # Binarize using the current threshold
        binary_field = np.where(density_field > threshold, 1, 0)

        # Compute the average density of the binary field
        binary_avg_density = np.mean(binary_field)
        
        # Adjust the threshold
        if binary_avg_density > avg_density:
            low = threshold  # Threshold is too low, increase it
        else:
            high = threshold  # Threshold is too high, decrease it
    
    # Final binarized field
    binary_field = np.where(density_field > threshold, 1, 0)
    
    return binary_field, threshold

def dynamic_binarize_density_2(density_field, target_vf, tolerance=1e-4):
    """
    Binarizes a given density field such that the final volume fraction matches the target VF.

    Parameters:
    - density_field: 2D or 3D numpy array representing the density field.
    - target_vf: Desired volume fraction (between 0 and 1).
    - tolerance: Convergence tolerance for the volume fraction.

    Returns:
    - binary_field: Binarized density field with volume fraction close to target VF.
    - threshold: Final threshold used for binarization.
    """
    # Initialize threshold search range
    low, high = 0.0, 1.0
    
    # Ensure the target VF is feasible
    # min_density = np.min(density_field)
    # max_density = np.max(density_field)
    # print(min_density)
    # print(max_density)
    # print(np.mean(density_field == min_density))
    # print(np.mean(density_field == max_density))
    # if target_vf < np.mean(density_field == min_density) or target_vf > np.mean(density_field == max_density):
    #     raise ValueError("Target volume fraction is outside achievable range.")

    while True:
        threshold = (low + high) / 2
        
        # Binarize using the current threshold
        binary_field = np.where(density_field > threshold, 1, 0)

        # Compute the actual volume fraction
        current_vf = np.mean(binary_field)
        # print(current_vf)
        # Check for convergence
        if abs(current_vf - target_vf) < tolerance:
            break
        
        # Adjust threshold based on VF difference
        if current_vf > target_vf:
            low = threshold  # Threshold is too low, increase it
        else:
            high = threshold  # Threshold is too high, decrease it
    
    return binary_field, threshold

def dynamic_binarize_density_3(density_field, target_vf, tolerance=1e-3, max_iter=2000):
    """
    Binarizes a given density field such that the final volume fraction matches the target VF.

    Parameters:
    - density_field: 2D or 3D numpy array representing the density field.
    - target_vf: Desired volume fraction (between 0 and 1).
    - tolerance: Convergence tolerance for the volume fraction.
    - max_iter: Maximum number of iterations for threshold search.

    Returns:
    - binary_field: Binarized density field with volume fraction close to target VF.
    - threshold: Final threshold used for binarization.
    """
    low, high = 0.0, 1.0
    
    for _ in range(max_iter):
        threshold = (low + high) / 2

        # Binarize using the current threshold
        binary_field = np.where(density_field > threshold, 1, 0)

        # Compute the actual volume fraction
        current_vf = np.mean(binary_field)

        # Check for convergence
        if abs(current_vf - target_vf) < tolerance:
            return binary_field, threshold
        
        # Adjust threshold based on VF difference
        if current_vf > target_vf:
            low = threshold
        else:
            high = threshold

    # If max_iter reached without convergence, return best found
    return binary_field, threshold




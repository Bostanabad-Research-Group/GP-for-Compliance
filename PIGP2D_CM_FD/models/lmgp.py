# Copyright © 2021 by Northwestern University.
# 
# LVGP-PyTorch is copyrighted by Northwestern University. It may be freely used 
# for educational and research purposes by  non-profit institutions and US government 
# agencies only. All other organizations may use LVGP-PyTorch for evaluation purposes 
# only, and any further uses will require prior written approval. This software may 
# not be sold or redistributed without prior written approval. Copies of the software 
# may be made by a user provided that copies are not sold or distributed, and provided 
# that copies are used under the same terms and conditions as agreed to in this 
# paragraph.
# 
# As research software, this code is provided on an "as is'' basis without warranty of 
# any kind, either expressed or implied. The downloading, or executing any part of this 
# software constitutes an implicit agreement to these terms. These terms and conditions 
# are subject to change at any time without prior notice.

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
import math

import gpytorch
from gpytorch.constraints import Positive
from gpytorch.priors import NormalPrior
from gpytorch.means import Mean

from utils.lmgp_utils.priors import MollifiedUniformPrior
from utils.lmgp_utils import kernels
from .gpregression import GPR

from .pgcan_utils import Model_PGCAN , Encoder , NetworkM4_fused
from utils.utils_general import get_tkwargs

control = get_tkwargs()

class LMGP(GPR):
    """The latent Map GP regression model (LMGP) which extends GPs to handle categorical inputs.

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no 
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as 
        integers 0,...,L-1 where L is the number of levels. For best performance, scale the 
        numerical variables to the unit hypercube.
    """
    def __init__(
        self,
        train_x:torch.Tensor,
        train_y:torch.Tensor,
        collocation_x:torch.Tensor,
        NN_config = {}, # parameters to config NN
        Training = {}, # Training dataset
        MP = {}, # material properties
        name_output:str='u',
        num_output = 1,
    ) -> None:
        
        tkwargs = {}  # or dict()
        tkwargs['dtype'] = control['dtype']
        tkwargs['device'] = control['device']

        quant_index = list(range(train_x.shape[-1]))

        quant_correlation_class_name = NN_config['quant_correlation_class']

        if quant_correlation_class_name == 'Rough_RBF':
            quant_correlation_class = 'RBFKernel'
        
        if quant_correlation_class_name == 'Matern32Kernel':
            quant_correlation_class = 'Matern32Kernel'
        
        if quant_correlation_class_name == 'Matern52Kernel':
            quant_correlation_class = 'Matern52Kernel'

        if quant_correlation_class_name == 'Matern12Kernel':
            quant_correlation_class = 'Matern12Kernel'

        try:
            quant_correlation_class = getattr(kernels,quant_correlation_class)
        except:
            raise RuntimeError(
                "%s not an allowed kernel" % quant_correlation_class
            )
    
        if quant_correlation_class_name == 'RBFKernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= torch.exp,inv_transform= torch.log)
                )
        elif quant_correlation_class_name == 'Rough_RBF':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform = lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
                )


        elif quant_correlation_class_name == 'Matern12Kernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))
                )
            
        elif quant_correlation_class_name == 'Matern32Kernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))             
                )

        elif quant_correlation_class_name == 'Matern52Kernel':
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=torch.arange(len(quant_index)),
                    lengthscale_constraint= Positive(transform= lambda x: 2.0**(-0.5) * torch.pow(10,-x/2),inv_transform= lambda x: -2.0*torch.log10(x/2.0))       
                )

        if quant_correlation_class_name == 'RBFKernel':
                
                quant_kernel.register_prior(
                    'lengthscale_prior', MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                )
                
        elif quant_correlation_class_name == 'Rough_RBF':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )

        elif quant_correlation_class_name == 'Matern12Kernel':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )

        elif quant_correlation_class_name == 'Matern32Kernel':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(0.5,0.5),'raw_lengthscale'
                )

        elif quant_correlation_class_name == 'Matern52Kernel':
                quant_kernel.register_prior(
                    'lengthscale_prior',NormalPrior(-3.0,3.0),'raw_lengthscale'
                )
        
        correlation_kernel = quant_kernel

        super(LMGP,self).__init__(
            train_x=train_x,train_y=train_y,correlation_kernel=correlation_kernel,noise_indices=[]
        )

        # register index and transforms
        self.register_buffer('quant_index',torch.tensor(quant_index))

        # latent variable mapping
        self.perm =[]
        self.zeta = []
        self.perm_dict = []
        self.A_matrix = []
        self.collocation_x = collocation_x ####### ADDED
        self.dx = Training['dx'] ####### ADDED
        self.dy = Training['dy'] ####### ADDED
        self.Nx = Training['Nx'] ####### ADDED
        self.Ny = Training['Ny'] ####### ADDED
        self.activation = NN_config['activation'] ####### ADDED
        self.traction_indices = Training['traction_indices'] ####### ADDED
        self.traction_magnitude = Training['traction_magnitude'] ####### ADDED
        self.mask_col = Training['mask_col'] ####### ADDED
        self.mask_nan = Training['mask_nan'] ####### ADDED
        self.xi = Training['xi'] ####### ADDED
        self.yi = Training['yi'] ####### ADDED
        self.num_output = num_output, ####### ADDED
        self.alpha = 1.0
        self.beta = 20.0
        self.covar_inv = None
        self.omega = NN_config['omega']#5.0#3.2
        self.name_output = name_output
        self.chol_decomp = None
        self.g_uvp = None
        self.basis = NN_config['basis'] 
        self.NN_layers_base = NN_config['NN_arch']
        self.MP = MP
    
        self.res = NN_config['res']
        self.n_features = NN_config['n_features']
        self.n_cells = NN_config['n_cells']
        self.domain_size = [MP['domain']['x'][0],MP['domain']['x'][1],MP['domain']['y'][0],MP['domain']['y'][1]]
        self.save_folder = NN_config['save_folder'] ####### ADDED

        if self.basis=='neural_network':
            ############################################### One NN for ALL
                setattr(self,'mean_module_NN_All', FFNN_for_Mean(self, input_size= train_x.shape[1], num_classes=num_output,layers = self.NN_layers_base, name_output = self.name_output, activation = self.activation))

        elif self.basis=='M3':
            setattr(self,'mean_module_NN_All', NetworkM4(input_dim = train_x.shape[1], output_dim=num_output, layers = self.NN_layers_base))     
        
        elif self.basis =='PGCAN': 
             network =  NetworkM4_fused(input_dim = train_x.shape[1], output_dim=num_output, layers = self.NN_layers_base, activation = self.activation)
             encoder = Encoder(
                                n_features=self.n_features,  
                                res=self.res,                      
                                n_cells=self.n_cells,         
                                domain_size=self.domain_size
             )
             setattr(self,'mean_module_NN_All', Model_PGCAN(network = network , encoder = encoder)) 

        # Fix the hyperparameter value
        self.covar_module.base_kernel.raw_lengthscale.data = torch.tensor([self.omega, self.omega])  # Set the desired value
        self.covar_module.base_kernel.raw_lengthscale.requires_grad = False  # Fix the hyperparameter

        self.covar_module.raw_outputscale.data = torch.tensor(0.541)  # Set the desired value
        self.covar_module.raw_outputscale.requires_grad = False  # Fix the hyperparameter

    def predict(self, Xtest,return_std=True, include_noise = True):
        with torch.no_grad():
            return super().predict(Xtest, return_std = return_std, include_noise= include_noise)
            
    def predict_with_grad(self, Xtest,return_std=True, include_noise = True):
        return super().predict(Xtest, return_std = return_std, include_noise= include_noise)
    


    @classmethod
    def get_params(self, name = None):
        params = {}
        print('###################Parameters###########################')
        for n, value in self.named_parameters():
             params[n] = value
        if name is None:
            print(params)
            return params
        else:
            if name == 'Mean':
                key = 'mean_module.constant'
            elif name == 'Sigma':
                key = 'covar_module.raw_outputscale'
            elif name == 'Noise':
                key = 'likelihood.noise_covar.raw_noise'
            elif name == 'Omega':
                for n in params.keys():
                    if 'raw_lengthscale' in n and params[n].numel() > 1:
                        key = n
            print(params[key])
            return params[key]

def modified_sigmoid(x, alpha=8.0):
    return 1 / (1 + torch.exp(-alpha * (x - 0.5)))

class FFNN_for_Mean(gpytorch.Module):
    def __init__(self, lmgp, input_size, num_classes, layers, name_output, activation = 'leaky_relu', x_mean=0.0, y_mean=0.0, x_std=1.0, y_std=1.0):
        super(FFNN_for_Mean, self).__init__()
        # Store normalization parameters as buffers (they don't require gradients)
        self.register_buffer('x_mean', torch.tensor(x_mean))
        self.register_buffer('y_mean', torch.tensor(y_mean))
        self.register_buffer('x_std', torch.tensor(x_std))
        self.register_buffer('y_std', torch.tensor(y_std))
        
        self.dropout = nn.Dropout(0.0)
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.hidden_num = len(layers)
        self.activation = self._get_activation(activation)
        self.name_output = name_output
        print(f"Initializing FFNN_for_Mean with name_output: {self.name_output}")
        print(f"Initializing FFNN_for_Mean with num_classes: {num_classes}")

        if self.hidden_num > 0:
            self.fci = Linear_new(input_size, layers[0], bias=True, name='fci') 
            for i in range(1,self.hidden_num):
                setattr(self, 'h' + str(i), Linear_new(layers[i-1], layers[i], bias=True,name='h' + str(i)))
            
            self.fce = Linear_new(layers[-1], num_classes, bias=True,name='fce')
        else:
            self.fci = Linear_new(input_size, num_classes, bias=True,name='fci') #Linear_MAP(input_size, num_classes, bias = True)

    def normalize_input(self, x):
        """ Normalize input coordinates. """
        # Assume x has two columns: first for x-coordinates, second for y-coordinates
        x[:, 0] = (x[:, 0] - self.x_mean) / self.x_std
        x[:, 1] = (x[:, 1] - self.y_mean) / self.y_std
        return x
    
    def forward(self, x, transform = lambda x: x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        # Normalize input coordinates before feeding into the first layer
        x = self.normalize_input(x)

        if self.hidden_num > 0:
            x = self.activation(self.fci(x))
            for i in range(1,self.hidden_num):
                x = self.activation( getattr(self, f'h{i}')(x) )
            x = self.fce(x)
        else:
            x = self.fci(x)
        return x

    def _get_activation(self, activation_name):
        # Define supported activation functions
        if activation_name == 'tanh':
            return torch.tanh
        elif activation_name == 'relu':
            return torch.relu
        elif activation_name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif activation_name == 'sigmoid':
            return torch.sigmoid
        elif activation_name == 'softplus':
            return nn.Softplus()
        elif activation_name == 'swish':
            # Swish activation function: x * sigmoid(x)
            return lambda x: x * torch.sigmoid(x)
        elif activation_name == 'sin':
            # Sin activation function for periodic tasks
            return torch.sin
        else:
            raise ValueError(f"Unsupported activation function: {activation_name}")

class NetworkM4(nn.Module):
    def __init__(self, input_dim = 2, output_dim = 1, layers = [40, 40, 40, 40], activation = 'tanh', collocation_x = []) -> None:
        super(NetworkM4, self).__init__()
        activation_list = {'tanh':nn.Tanh(), 'Silu':nn.SiLU(), 'Sigmoid':nn.Sigmoid()}
        activation = activation_list[activation]
        self.dim = layers[0]
  
        self.U = nn.Linear(input_dim, self.dim).to(control['device'])
        self.V = nn.Linear(input_dim, self.dim).to(control['device'])
        self.H1 = nn.Linear(input_dim, self.dim).to(control['device'])
        self.last= nn.Linear(self.dim, output_dim).to(control['device'])

        self.collocation_x = collocation_x
        self.alpha = 1.0
        self.beta = 1.0
        
        l = nn.ModuleList()
        for _ in range(len(layers)):
            l.append(nn.Linear(self.dim, self.dim))
            l.append(activation)
        self.layers = nn.Sequential(*l).to(control['device'])

    def forward(self, input):        
        U = nn.Tanh()(self.U(input))
        V = nn.Tanh()(self.V(input))
        H = nn.Tanh()(self.H1(input))

        for layer in self.layers:
            Z = layer(H)
            H = (1-Z)*U + Z*V
        
        out = self.last(H)
        return out

class Linear_new(Mean):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, name=None,
                 device=None, dtype=None) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        # factory_kwargs=tkwargs
        super(Linear_new, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.name=str(name)
        self.register_parameter(name=str(self.name)+'weight',  parameter= Parameter(torch.empty((out_features, in_features))))
        self.register_prior(name =str(self.name)+ 'prior_m_weight_fci', prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure=str(self.name)+'weight')
        
        if bias:
            self.register_parameter(name=str(self.name)+'bias',  parameter=Parameter(torch.empty(out_features)))
            self.register_prior(name= str(self.name)+'prior_m_bias_fci', prior=gpytorch.priors.NormalPrior(0.,1.), param_or_closure=str(self.name)+'bias')
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:                                         

        # Modify weight initialization to use a different range or method
        init.kaiming_uniform_(getattr(self, str(self.name) + 'weight'), a=math.sqrt(50))

        # You can try a different initialization like Xavier (Glorot)
        # init.xavier_uniform_(getattr(self, str(self.name) + 'weight'))

        if getattr(self,str(self.name)+'bias') is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(getattr(self,str(self.name)+'weight'))
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(getattr(self,str(self.name)+'bias'), 6*bound, 8*bound)

    def forward(self, input) -> Tensor:
        # return F.linear(input, self.weight, self.bias)
        # print(getattr(self,str(self.name)+'weight'))

        # return F.linear(input, getattr(self,str(self.name)+'weight').double(), getattr(self,str(self.name)+'bias').double())      ### Forced to Add .double() for NN in mean function
        return F.linear(input, getattr(self,str(self.name)+'weight'), getattr(self,str(self.name)+'bias'))      ### Forced to Add .double() for NN in mean function

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

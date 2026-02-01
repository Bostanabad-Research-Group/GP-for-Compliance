# Compliance Minimization via Physics-Informed Gaussian Processes
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation of our paper **"Compliance Minimization via Physics-Informed Gaussian Processes"** (PIGPs). We propose a mesh-free and simultaneous optimization framework for compliance minimization (CM), where both design and state variables are parameterized using Gaussian Process (GP) priors. Each GP employs an independent kernel while sharing a multi-output neural network (NN) as its mean function.

---

## Framework Overview
The architecture of the proposed PIGP framework is illustrated below:

![PIGP framework for CM: the mean functions are parameterized with the multi-outputs from the PGCAN having three modules: convolutional neural network-based feature encoding, feature interpolation, and decoding via a shallow MLP. GPs are employed on those mean functions to impose displacement BCs and density constraints.](figures/Schematic_TO.png)

- The mean function is parameterized with a **Parametric Grid Convolutional Attention Network (PGCAN)**, which mitigates spectral bias and provides explicit control over design complexity.  
- All parameters of the GP-based representation are estimated by **simultaneously minimizing**:
  1. Compliance,  
  2. Total potential energy,  
  3. Residual of the volume fraction constraint.  
- Importantly, our formulation excludes data-based residuals, as GPs inherently satisfy them.  

Typical results compared to the SIMP baseline are shown below:

![Comparison of final designs: For both approaches we visualize the topologies corresponding to the median compliance in each example. The cell vertices of PGCAN are also shown to demonstrate the effect of $Res$ on partitioning the design domain via PGCAN's encoder.](figures/Top_comparison.png)

---

## Key Features
Our PIGP framework enables:
1. Super-resolution topology generation with fast convergence.  
2. Comparable compliance with **reduced gray-area fraction** relative to classical methods.  
3. Direct control over **fine-scale features** through PGCAN encoding.  
4. Superior performance compared to existing ML-based approaches.  

---

## Requirements
To run the code, please set up a Python environment and install the following packages.  
We recommend creating a dedicated conda environment:

```bash
conda create --name IC python=3.12
conda activate IC
```

Install core dependencies:
```bash
# PyTorch with CUDA
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# BoTorch (install first)
pip install botorch==0.11.3

# GPyTorch (install after BoTorch, ignore mismatch warnings)
pip install gpytorch==1.9.1

# Scientific stack
pip install numpy==2.2.6 scipy==1.16.0 scikit-learn==1.7.0 pandas==2.3.2

# Visualization
pip install matplotlib==3.10.0 vtk==9.4.2 pyvista==0.45.3

# Utilities
pip install tqdm==4.67.1 dill sympy plotly
```

---

## Usage
1. Clone this repository.  
2. Navigate to the **PIGP2D_CM_FD** folder.  
3. Run the example scripts via:
   ```bash
   python main.py
   ```
   You can customize:
   - Example selection,  
   - Network architecture,  
   - Kernel type,  
   - Number of epochs,  
   - Other parser options.  
4. After training, run:
   ```bash
   python Process_data.py
   ```
   to generate figures and statistical results.  

---

## Contributions and Support
Contributions are welcome!  
- If you find a bug, mistake, or unclear documentation, please [open an issue](../../issues) and label it with the relevant module/feature.  
- For major contributions, please submit a pull request.  

---

## Citation
If you use this code or find our work useful, please cite:  

```bibtex
@article{sun_compliance_2025,
	title = {Compliance minimization via physics-informed {Gaussian} processes},
	volume = {68},
	issn = {1615-147X, 1615-1488},
	url = {https://link.springer.com/10.1007/s00158-025-04179-5},
	doi = {10.1007/s00158-025-04179-5},
	language = {en},
	number = {12},
	urldate = {2025-11-27},
	journal = {Structural and Multidisciplinary Optimization},
	author = {Sun, Xiangyu and Yousefpour, Amin and Hosseinmardi, Shirin and Bostanabad, Ramin},
	month = dec,
	year = {2025},
	pages = {259},
}
```

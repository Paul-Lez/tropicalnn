### Tropical NN

This repository contains the code used for the experiments in the paper *Tropical Expressivity of Neural Networks*. 

Languages used: 
- Julia with Oscar. Follow installation instructions here: https://www.oscar-system.org/install/
- Matlab (archived only)
- Python

# Instructions for running the experiments 

## Julia experiments
- Width-depth separation: run the file `paper/characterising_width_depth/launcher.jl`
- Linear regions of a network trained on MNIST: run the file `paper/mnist/launcher.jl`
- Symbolic calculations of linear regions: run the file `paper/num_linear_regions/launcher.jl`
- Redundant monomials: `paper/redundant_monomials/launcher.jl`
- Hoffman constants: `paper/hoffman_constants/trop_rat_map.jl`

## Python experiment 
- Numerical computation of linear regions of invariant neural networks (section 5.2). Run the Jupyter notebook `paper/numerical_linear_regions.ipynb`.

## Matlab experiments
For the sake of completness, we include the MATLAB code used to compute Hoffman constants using the PVZ algorithm. We found this version to be unstable, occasionally yielding incorrect numerical values.
The old version of MATLAB code for computing and estimating Hoffman constants of tropical Puiseux rational maps is based on the MATLAB code and scripts from [this page](https://www.andrew.cmu.edu/user/jfp/hoffman.html).
The MATLAB code requires the Optimization Toolbox is needed. A version of MATLAB 2023a or higher is recommended.
The computation of the Hoffman constant, estimation of lower/upper bounds, comparison of computational time is bundled in function `trop_test`. To run the function, in MATLAB console do
`trop_test(m_p,m_q,n)`
where `m_p` is the number of monomials in the numerator, `m_q` is the number of monomials in the denominator, and `n` is the dimension of variables.
To display the results, in MATLAB console do
`disp_results`
All results shown in the paper are stored in `.mat` files and can be shown by `disp_results` function. 

The MATLAB code for computing and estimating Hoffman constants of tropical Puiseux rational maps is based on the MATLAB code and scripts from https://www.andrew.cmu.edu/user/jfp/hoffman.html

The MATLAB code requires the Optimization Toolbox is needed. A version of MATLAB 2023a or higher is recommended.

The computation of the Hoffman constant, estimation of lower/upper bounds, comparison of computational time is bundled in function ```trop_test'''. To run the function, in MATLAB console do

```trop_test(m_p,m_q,n)'''

where ```m_p''' is the number of monomials in the nominator, ```m_q''' is the number of monomials in the denominator, and ```n''' is the dimension of variables.

To display the results, in MATLAB console do

```disp_results'''

All results shown in the paper are stored in ```.mat''' files and can be shown by ```disp_results''' function.


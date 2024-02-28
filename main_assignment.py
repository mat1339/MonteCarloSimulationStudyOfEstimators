 """
Data Analytics II: Causal Econometrics

Individual Assignment - Monte Carlo Simulation

Main File 

@author: Matthias Lukosch
MEcon Student at the University of St.Gallen
"""


# import modules
import sys
import numpy as np
import pandas as pd
from numpy.random import seed


# set working directory
PATH = '/Users/matthias/Documents/DA 2/Individual Assignment/'
sys.path.append(PATH)


# load own functions
import func_assignment as func


###
###
### DGP_1: y = beta0 + beta1 * d + beta2 * x + u
# define true coefficients
beta = np.array([[30],[5],[8]])

# number of observation to be drawn from population
n = 200

# define population moments
normal_mean = 0
normal_std = 40
u_means = np.zeros(n)
u_cov = 50 * np.identity(n)
degree_overlap = 50

# define number of simulations
sim_num = 500

# set seed for replicability
seed(42)

# draw a first sample from the population
y, cov = func.DGP_1(beta, 
                    n, 
                    normal_mean, 
                    normal_std, 
                    degree_overlap, 
                    u_means, 
                    u_cov)

# vizualize common support
func.my_CSC(cov)

# correlation matrix of treatment and confounder
print(np.corrcoef(cov[:,1], cov[:,2]))

# set seed for replicability
seed(42)

# OLS simulation
OLS_results_DGP_1 = func.my_OLS_Simu(sim_num, 
                                     DGP_1, 
                                     beta, 
                                     n, 
                                     normal_mean, 
                                     normal_std, 
                                     degree_overlap, 
                                     u_means, 
                                     u_cov)

# IPW simulation
IPW_results_DGP_1 = func.my_IPW_Simu(sim_num, 
                                     DGP_1, 
                                     my_IPW, 
                                     beta, 
                                     n, 
                                     normal_mean, 
                                     normal_std, 
                                     degree_overlap, 
                                     u_means, 
                                     u_cov)


# create table for comparison
DGP_1_table = pd.DataFrame(list(zip(OLS_results_DGP_1, IPW_results_DGP_1)),
                           index = ['Bias', 'Variance', 'MSE'],
                           columns = ['OLS', 'IPW']) 

# print results
print(DGP_1_table)


###
###
### DGP_2: y = beta0 + beta1 * d + beta2 * x1 + beta3 * x2 + beta4 * x3 (imperfect multicollinearity)
# define true coefficients
beta = np.array([[30],[5],[8],[45],[0]])

# number of observation to be drawn from population
n = 500

# define population moments
mv_means = (4,9)
mv_covariances = np.array([[300,80],[80,150]])
u_means = np.zeros(n)
u_cov = 50 * np.identity(n)
degree_overlap = 50

# set seed for replicability
seed(42)

# OLS simulation
OLS_results_DGP_2 = func.my_OLS_Simu(sim_num, 
                                     DGP_2, 
                                     beta, 
                                     n,
                                     mv_means, 
                                     mv_covariances, 
                                     degree_overlap, 
                                     u_means,
                                     u_cov)


# IPW simulation
IPW_results_DGP_2 = func.my_IPW_Simu(sim_num, 
                                     DGP_2, 
                                     my_IPW, 
                                     beta, 
                                     n, 
                                     mv_means, 
                                     mv_covariances, 
                                     degree_overlap, 
                                     u_means, 
                                     u_cov)


# create table for comparison
DGP_2_table = pd.DataFrame(list(zip(OLS_results_DGP_2, IPW_results_DGP_2)),
                           index = ['Bias', 'Variance', 'MSE'], 
                           columns = ['OLS', 'IPW']) 

# print results
print(DGP_2_table)



###
###
### DGP_3: y = beta0 + beta1 * d + beta2 * x1 + beta3 * x2 (x1 is unobservable, CIA violated)
# define true coefficients
beta = np.array([[30],[5],[8],[45]])

# number of observation to be drawn from population
n = 500

# define population moments
mv_means = (4,9)
mv_covariances = np.array([[300,80],[80,150]])
u_means = np.zeros(n)
u_cov = 50 * np.identity(n)
degree_overlap = 50

# set seed for replicability
seed(42)

# OLS simulation
OLS_results_DGP_3 = func.my_OLS_Simu_OVB(sim_num, 
                                         DGP_3, 
                                         beta, 
                                         n, 
                                         mv_means,
                                         mv_covariances,
                                         degree_overlap,
                                         u_means,
                                         u_cov)


# IPW simulation
IPW_results_DGP_3 = func.my_IPW_Simu(sim_num, 
                                     DGP_3, 
                                     my_IPW_SB, 
                                     beta, 
                                     n,
                                     mv_means, 
                                     mv_covariances, 
                                     degree_overlap, 
                                     u_means, 
                                     u_cov)


# create table for comparison
DGP_3_table = pd.DataFrame(list(zip(OLS_results_DGP_3, IPW_results_DGP_3)),
                           index = ['Bias', 'Variance', 'MSE'], 
                           columns = ['OLS', 'IPW']) 

# print results
print(DGP_3_table)












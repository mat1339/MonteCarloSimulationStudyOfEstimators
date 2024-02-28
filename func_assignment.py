"""
Data Analytics II: Causal Econometrics

Individual Assignment - Monte Carlo Simulation

Functions File

@author: Matthias Lukosch
MEcon Student at the University of St.Gallen
"""

### import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import multivariate_normal
from numpy.random import normal
from statsmodels.discrete.discrete_model import Probit



### function DGP1
def DGP_1(beta, n, normal_mean, normal_std, degree_overlap ,u_means, u_cov):
    """
    

    Parameters
    ----------
    beta : Type: np.array
        DESCRITPION: Vector that contains the true population coefficients.
    n : TYPE: int
        DESCRIPTION: Number of observations to be drawn from the population.
    normal_mean : TYPE: int
        DESCRIPTION: Mean of the confounder in the population.
    normal_std : TYPE: int
        DESCRIPTION: Standard Deviation of the confounder in the population.
    degree_overlap: Type: int
        DESCRIPTION: Standard deviation of the threshold error.
    u_means: Type: np.array
        DESCRIPTION: Means of the error terms of the DGP.
    u_cov : TYPE: np.array
        DESCRIPTION: Variance-Covariance Matrix of the error terms.

    Returns
    -------
    Sample drawn from the population.

    """
    
    # draw data
    # confounder drawn from normal distribution
    x = np.array([normal(normal_mean, normal_std, n)]).T
    # specify threshold level
    threshold = np.array([x[:,0]]).T + np.array([normal(0,degree_overlap,n)]).T > normal_mean 
    # create dummy storage
    d = np.zeros((n,1))
    # assign treatment
    d[threshold[:,0] == True , 0] = 1
    # error terms drawn from multivariate normal distribution
    u = np.array([multivariate_normal(u_means, u_cov)]).T
    
    # concatenate covariates
    cov = np.c_[np.ones(n), d, x]
    
    # calculate dependent variable y
    y = cov @ beta + u
    
    return(y,cov)

### function DGP2
def DGP_2(beta, n, mv_means, mv_covariances, degree_overlap, u_means, u_cov):
    """
    

    Parameters
    ----------
    beta : Type: np.array
        DESCRITPION: Vector that contains the true population coefficients.
    n : TYPE: int
        DESCRIPTION: Number of observations to be drawn from the population.
    mv_means : TYPE: tuple
        DESCRIPTION: Means of the confounders that are to be drawn from multivariate normal distribution.
    mv_covariances : TYPE: np.array
        DESCRIPTION: Variance-Covariance Matrix of confounders.
    degree_overlap: Type: int
        DESCRIPTION: Standard deviation of the threshold error.
    u_means: Type: np.array
        DESCRIPTION: Means of the error terms of the DGP.
    u_cov : TYPE: np.array
        DESCRIPTION: Variance-Covariance Matrix of the error terms.

    Returns
    -------
    Sample drawn from the population.

    """
    
    ## draw data
    # confounders drawn from the multivariate normal distribution
    x = multivariate_normal(mv_means, mv_covariances, n)
    # specify threshold levels
    threshold1 = np.array([x[:,0]]).T + np.array([normal(0,degree_overlap,n)]).T > mv_means[0] 
    threshold2 = np.array([x[:,1]]).T + np.array([normal(0,degree_overlap,n)]).T > mv_means[1] 
    # create dummy storage
    d = np.zeros((n,1))
    # assign treatment
    d[threshold1[:,0] & threshold2[:,0] == True , 0] = 1
    # error terms drawn from the multivariate normal distribution
    u = np.array([multivariate_normal(u_means, u_cov)]).T
    # third confounder as linear combination of the other two + noice
    x3 = np.array([x[:,1]]).T * 0.8 + 0.2 * np.array([x[:,0]]).T + normal(0,500,1)
    
    
    # concatenate covariates
    cov = np.c_[np.ones(n), d, x, x3]
    
    # calculate dependent variable y
    y = cov @ beta + u
    
    return(y,cov)


### function DGP3
def DGP_3(beta, n, mv_means, mv_covariances, degree_overlap, u_means, u_cov):
    """
    

    Parameters
    ----------
    beta : Type: np.array
        DESCRITPION: Vector that contains the true population coefficients.
    n : TYPE: int
        DESCRIPTION: Number of observations to be drawn from the population.
    mv_means : TYPE: tuple
        DESCRIPTION: Means of the confounders that are to be drawn from multivariate normal distribution.
    mv_covariances : TYPE: np.array
        DESCRIPTION: Variance-Covariance Matrix of confounders.
    degree_overlap: Type: int
        DESCRIPTION: Standard deviation of the threshold error.
    u_means: Type: np.array
        DESCRIPTION: Means of the error terms of the DGP.
    u_cov : TYPE: np.array
        DESCRIPTION: Variance-Covariance Matrix of the error terms.

    Returns
    -------
    Sample drawn from the population.

    """
    
    ## draw data
    # confounders drawn from the multivariate normal distribution
    x = multivariate_normal(mv_means, mv_covariances, n)
    # specify threshold level
    threshold = np.array([x[:,0]]).T + np.array([normal(0,degree_overlap,n)]).T > mv_means[0]
    # create dummy storage
    d = np.zeros((n,1))
    # assign treatment
    d[threshold[:,0] == True , 0] = 1
    # error terms drawn from the multivariate normal distribution
    u = np.array([multivariate_normal(u_means, u_cov)]).T
    
    # concatenate covariates
    cov = np.c_[np.ones(n), d, x]
    
    # calculate dependent variable y
    y = cov @ beta + u
    
    return(y,cov)


### OLS function
def my_OLS(y, cov):
    """
    

    Parameters
    ----------
    y : TYPE: np.array
        DESCRIPTION: Dependent variable y.
    cov : TYPE: np.array
        DESCRIPTION: Covariates.

    Returns
    -------
    Vector that contains the estimated coefficients.

    """
    # calculate beta head
    beta_head = np.linalg.inv(cov.T @ cov) @ cov.T @ y
    
    return(beta_head)

### IPW function
def my_IPW(y,cov):
    """
    

    Parameters
    ----------
    y : TYPE: np.array
        DESCRIPTION: Dependent variable y.
    cov : TYPE: np.array
        DESCRIPTION: Covariates.

    Returns
    -------
    ate : TYPE: int
        DESCRIPTION: Average treatment effect.

    """
    
    ## 1st step
    # store treatment variable separately
    d = np.array([cov[:,1]]).T
    # adjust covariates matrix
    x = np.delete(cov,1,1)
    # probit model
    model = Probit(d,x)
    probit_model = model.fit()
    # predict propensity scores
    p = np.array([probit_model.predict(x)]).T
    
    ## 2nd step
    # ate calculation
    ate = np.mean(d*y/p - ((1-d)*y)/(1-p))
    
    return ate

### IPW function in case of selection bias
def my_IPW_SB(y,cov):
    """
    

    Parameters
    ----------
    y : TYPE: np.array
        DESCRIPTION: Dependent variable y.
    cov : TYPE: np.array
        DESCRIPTION: Covariates.

    Returns
    -------
    ate : TYPE: int
        DESCRIPTION: Average treatment effect.

    """
    
    ## 1st step
    # store treatment variable separately
    d = np.array([cov[:,1]]).T
    # adjust covariate matrix
    delete = np.array([[1,2]])
    x = np.delete(cov, delete,1)
    # probit model
    model = Probit(d,x)
    probit_model = model.fit()
    # predict propensity scores
    p = np.array([probit_model.predict(x)]).T
    
    ## 2nd step
    # calculate ATE
    ate = np.mean(d*y/p - ((1-d)*y)/(1-p))
    
    return ate 

### DR function
def my_DR(y,cov):
    """
    

    Parameters
    ----------
    y : TYPE: np.array
        DESCRIPTION: Dependent variable y.
    cov : TYPE: np.array
        DESCRIPTION: Covariates.

    Returns
    -------
    ate : TYPE: int
        DESCRIPTION: Average treatment effect.

    """
    
    ## 1st step
    # store treatment variable separately
    d = np.array([cov[:,1]]).T
    # adjust covariate matrix
    x = np.delete(cov,1,1)
    # probit model
    model = Probit(d,x)
    probit_model = model.fit()
    # predict propensity scores
    p = np.array([probit_model.predict(x)]).T
    
    ## 2nd step
    # outcome regression parts (not correctly coded so far)
    mu_treat = np.mean(d*y/p)
    mu_control = np.mean(((1-d)*y)/(1-p))
    
    ## 3rd step
    # calculate ATE
    ate = np.mean(mu_treat - mu_control + (d * (y - mu_treat))/p - ((1-d)*(y - mu_control))/ (1-p))
    
    return ate

### histogram function
def my_hist(results, true_value, my_title):
    """
    

    Parameters
    ----------
    results : TYPE: np.array
        DESCRIPTION: Vector that contains the estimates.
    true_value : TYPE: int
        DESCRIPTION: True population parameter of the ATE.
    my_title : TYPE: string
        DESCRIPTION: Title of Histogram.

    Returns
    -------
    Histogram of the estimates.

    """
    
    # create figure
    plt.figure()
    # create histogram
    plt.hist(x = results[:,0], bins = 'auto', color = 'blue', edgecolor = 'black')
    # add line for true value
    plt.axvline(x= true_value, color = 'red', label="true value")
    # add legend
    plt.legend(loc='best')
    # add title
    plt.title(my_title)
    # display figure
    plt.show()
 
### common support condition function
def my_CSC(cov):
    """
    

    Parameters
    ----------
    cov : TYPE: np.array
        DESCRIPTION: Cavariates matrix.

    Returns
    -------
    Histogram of the confounder values grouped by treated and non-treated.

    """
    # find indices of treated and controls
    cov_treated = cov[np.where(cov[:,1] == 1)]
    cov_control = cov[np.where(cov[:,1] == 0)]

    # create figure
    plt.figure()
    # create histograms
    plt.hist(x = cov_treated[:,2], bins = 'auto', color = 'blue', alpha = 0.5, edgecolor = 'black', label = 'Covariate X Treated')
    plt.hist(x = cov_control[:,2], bins = 'auto', color = 'green', alpha = 0.5, edgecolor = 'black', label = 'Covariate X Control')
    # add legend
    plt.legend(loc = 'best')
    # display figure
    plt.show()
    
### OLS simulation function  
def my_OLS_Simu(sim_num, DGP_count, beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov):
    """
    

    Parameters
    ----------
    sim_num : TYPE: int
        DESCRIPTION: Number of simulations to be done.
    DGP_count : TYPE: string
        DESCRIPTION: Name of the particular DGP function (choose either 'DGP_1', 'DGP_2', or 'DGP_3').
    
    For the remaining input parameters, please see the documentation on the particular DGP process chosen.

    Returns
    -------
    OLS_column : TYPE: np.array
        DESCRIPTION: Vector that contains bias, variance, and MSE of OLS estimates.

    """
    # create storage
    results_OLS = np.zeros((sim_num,1))

    # loop over repeatedly drawn samples from the population
    for i in range(0,sim_num):
        y,cov = DGP_count(beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov)
        # calulate coefficients
        beta_head = my_OLS(y,cov)
        # store the coefficient on the dummy variable into an array
        results_OLS[i,0] = beta_head[1]


    # graphical presentation
    my_hist(results_OLS, beta[1], 'Histogram of OLS estimates')

    # performance measures
    # bias
    bias_OLS = np.round(np.float(np.mean(results_OLS[:,0]) - beta[1]),4)
    # variance
    variance_OLS = np.round(np.mean((results_OLS[:,0] - np.mean(results_OLS[:,0]))**2),4)
    # mean-squared error
    MSE_OLS = np.round(variance_OLS + bias_OLS**2,4)
    
    # combine performance measures
    OLS_column = [bias_OLS, variance_OLS, MSE_OLS]
    
    return OLS_column

### IPW simulation function
def my_IPW_Simu(sim_num, DGP_count, name_IPW, beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov):
    """
    

    Parameters
    ----------
    sim_num : TYPE: int
        DESCRIPTION: Number of simulations to be done.
    DGP_count : TYPE: string
        DESCRIPTION: Name of the particular DGP function (choose either 'DGP_1', 'DGP_2', or 'DGP_3').
    name_IPW : TYPE: string
        DESCRIPTION: Name of the IPW estimator function (choose eihter 'my_IPW' or 'my_IPW_SB').
    
    For the remaining input parameters, please see the documentation on the particular DGP process chosen.

    Returns
    -------
    IPW_column : TYPE: np.array
        DESCRIPTION: Vector that contains bias, variance, and MSE of IPW estimates.

    """
    
    # create storage
    results_IPW = np.zeros((sim_num,1))

    # loop over repeatedly drawn samples from the population
    for i in range(0,sim_num):
        y, cov = DGP_count(beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov)
        # estimate and store ate
        results_IPW[i,0]  = name_IPW(y,cov)

    # graphical presentation
    my_hist(results_IPW, beta[1], 'Histogram of IPW estimates')

    # performance measures
    # bias
    bias_IPW = np.round(np.float(np.mean(results_IPW[:,0]) - beta[1]),4)
    # variance
    variance_IPW = np.round(np.mean((results_IPW[:,0] - np.mean(results_IPW[:,0]))**2),4)
    # mean squared error
    MSE_IPW = np.round(variance_IPW + bias_IPW**2,4)
    
    IPW_column = [bias_IPW, variance_IPW, MSE_IPW]
    
    return IPW_column

### DR simulation function
def my_DR_Simu(sim_num, DGP_count, beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov):
    """
    

    Parameters
    ----------
    sim_num : TYPE: int
        DESCRIPTION: Number of simulations to be done.
    DGP_count : TYPE: string
        DESCRIPTION: Name of the particular DGP function (choose either 'DGP_1', 'DGP_2', or 'DGP_3').
    
    For the remaining input parameters, please see the documentation on the particular DGP process chosen.

    Returns
    -------
    DR_column : TYPE: np.array
        DESCRIPTION: Vector that contains bias, variance, and MSE of DR estimates.

    """
    # create storage
    results_DR = np.zeros((sim_num,1))

    # loop over repeatedly drawn samples from the population
    for i in range(0,sim_num):
        y, cov = DGP_count(beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov)
        # estimate and store ate
        results_DR[i,0] = my_DR(y,cov)
    

    # graphical presentation
    my_hist(results_DR, beta[1], 'Histogram of DR estimates')

    # performance measures
    # bias
    bias_DR = np.round(np.float(np.mean(results_DR[:,0]) - beta[1]),4)
    # variance
    variance_DR = np.round(np.mean((results_DR[:,0] - np.mean(results_DR[:,0]))**2),4)
    # mean-squared error
    MSE_DR = np.round(variance_DR + bias_DR**2,4)
    
    DR_column = [bias_DR, variance_DR, MSE_DR]
    
    return DR_column


### OLS simulation function in case of OVB
def my_OLS_Simu_OVB(sim_num, DGP_count, beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov):
    """
    

    Parameters
    ----------
    sim_num : TYPE: int
        DESCRIPTION: Number of simulations to be done.
    DGP_count : TYPE: string
        DESCRIPTION: Name of the particular DGP function (choose either 'DGP_1', 'DGP_2', or 'DGP_3').
    
    For the remaining input parameters, please see the documentation on the particular DGP process chosen.

    Returns
    -------
    OLS_column : TYPE: np.array
        DESCRIPTION: Vector that contains bias, variance, and MSE of OLS estimates.

    """
    # create storage
    results_OLS = np.zeros((sim_num,1))

    # loop over repeatedly drawn samples from the population
    for i in range(0,sim_num):
        y,cov = DGP_count(beta, n, mv_mean, mv_covariances, degree_overlap, u_means, u_cov)
        # prepare OVB
        cov = np.delete(cov,2,1)
        # calulate coefficients
        beta_head = my_OLS(y,cov)
        # store the coefficient on the dummy variable into an array
        results_OLS[i,0] = beta_head[1]


    # graphical presentation
    my_hist(results_OLS, beta[1], 'Histogram of OLS estimates')

    # performance measures
    # bias
    bias_OLS = np.round(np.float(np.mean(results_OLS[:,0]) - beta[1]),4)
    # variance
    variance_OLS = np.round(np.mean((results_OLS[:,0] - np.mean(results_OLS[:,0]))**2),4)
    # mean-squared error
    MSE_OLS = np.round(variance_OLS + bias_OLS**2,4)
    
    # combine performance measures
    OLS_column = [bias_OLS, variance_OLS, MSE_OLS]
    
    return OLS_column
    



    
    
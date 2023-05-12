import matplotlib.pyplot as plt
import numpy as np
import os
from iminuit import Minuit
from scipy.special import gamma, factorial

"""
Essentially identical to HiggsProblem2, but with minor corrections to expressions (an more explicit maths to help Python a bit)
"""

################            Generic Functions            ################
################            -----------------            ################
def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]    
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    """
    Finds all of the intersect coordinates between two lines (datasets) and returns them as two lists of corresponding coordinates.
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


################            Experiment Data            ################
################            ---------------            ################

N_obs_SR1 = 24
n_ggF_SR1 = 16.2
n_VBF_SR1 = 0.9
n_b_SR1 = 5.2

N_obs_SR1_D = n_ggF_SR1 + n_VBF_SR1 + n_b_SR1

N_obs_SR2 = 8
n_ggF_SR2 = 2.1
n_VBF_SR2 = 4.2
n_b_SR2 = 0.9

N_obs_SR2_D = n_ggF_SR2 + n_VBF_SR2 + n_b_SR2



################            Part a) Likelihood Equation            ################
################            ---------------------------            ################

def Poisson(x, mu):
    """
    The actual Poisson distribution function.
    """
    return ((mu**x) * np.exp((-1 * mu))) / factorial(x)


################            Part b) Simultaneous Measurements of mu_ggF, mu_VBF            ################
################            ---------------------------------------------------            ################

def Neg_L_LH(mu_ggF, mu_VBF):
    """
    Will include if statement to check for negative mu
    """
    mu_SR1 = (mu_ggF*n_ggF_SR1) + (mu_VBF*n_VBF_SR1) + n_b_SR1
    mu_SR2 = (mu_ggF*n_ggF_SR2) + (mu_VBF*n_VBF_SR2) + n_b_SR2
    if mu_SR1 < 0 or mu_SR2 < 0:
        return 1e8
    else:
        a = np.log(factorial(N_obs_SR1)) + np.log(factorial(N_obs_SR2))
        b = N_obs_SR1 * np.log(mu_SR1)
        c = N_obs_SR2 * np.log(mu_SR2)
        #return -1 * (np.log(Poisson(N_obs_SR1, mu_SR1)) + np.log(Poisson(N_obs_SR2, mu_SR2)))
        return mu_SR1 + mu_SR2 + a - b - c

NLLH = Minuit(Neg_L_LH, mu_ggF=1.0, mu_VBF=1.0)

NLLH.migrad()

NLLH.hesse()

NLLH.minos()

####    Manual 1D Profile Plots    ####
####    -----------------------    ####

Min_LH = Neg_L_LH(NLLH.values[0], NLLH.values[1])

sweep = np.linspace(0.5, 2.0, 100)

ggF_LH = []

VBF_LH = []

for i in sweep:
    NLLH_muggF = Minuit(Neg_L_LH, mu_ggF=i, mu_VBF=1.0)

    NLLH_muggF.fixed['mu_ggF'] = True

    NLLH_muggF.migrad()
    NLLH_muggF.hesse()
    NLLH_muggF.minos()

    ggF_LH.append(Neg_L_LH(NLLH_muggF.values[0], NLLH_muggF.values[1]))


    NLLH_muVBF = Minuit(Neg_L_LH, mu_ggF=1.0, mu_VBF=i)

    NLLH_muVBF.fixed['mu_VBF'] = True

    NLLH_muVBF.migrad()
    NLLH_muVBF.hesse()
    NLLH_muVBF.minos()

    VBF_LH.append(Neg_L_LH(NLLH_muVBF.values[0], NLLH_muVBF.values[1]))

Delta_muggF_NLLH = [val - Min_LH for val in ggF_LH]

Delta_muVBF_NLLH = [val - Min_LH for val in VBF_LH]

"""
Intersection Points
"""
ggF_x, ggF_y = intersection(sweep, 0.5*np.ones(len(sweep)), sweep, np.array(Delta_muggF_NLLH))

VBF_x, VBF_y = intersection(sweep, 0.5*np.ones(len(sweep)), sweep, np.array(Delta_muVBF_NLLH))


plt.figure(figsize=(15.0, 15.0))
plt.title('Profile Likelihood Plot for $\mu_{ggF}$', fontsize=20)
plt.xlabel('$\mu_{ggF}$', fontsize=20)
plt.ylabel('-$\Delta$ln($\mathcal{L}$)', fontsize=20)
plt.plot(sweep, Delta_muggF_NLLH, '-', linewidth=2, color='blue')
plt.vlines(NLLH.values[0], min(Delta_muggF_NLLH), max(Delta_muggF_NLLH), linestyles='dotted', colors='black')
plt.hlines(0.5, min(sweep), max(sweep), linestyles='dotted', colors='black')
plt.vlines(ggF_x, ymin=0, ymax=0.5, linestyles='dotted', colors='black')
plt.ylim(0, None)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
plt.show()

plt.figure(figsize=(15.0, 15.0))
plt.title('Profile Likelihood Plot for $\mu_{VBF}$', fontsize=20)
plt.xlabel('$\mu_{VBF}$', fontsize=20)
plt.ylabel('-$\Delta$ln($\mathcal{L}$)', fontsize=20)
plt.plot(sweep, Delta_muVBF_NLLH, '-', linewidth=2, color='blue')
plt.vlines(NLLH.values[1], min(Delta_muVBF_NLLH), max(Delta_muVBF_NLLH), linestyles='dotted', colors='black')
plt.hlines(0.5, min(sweep), max(sweep), linestyles='dotted', colors='black')
plt.vlines(VBF_x, ymin=0, ymax=0.5, linestyles='dotted', colors='black')
plt.ylim(0, None)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
plt.show()


####    Manual Contour Plots    ####
####    --------------------    ####

pts_68 = NLLH.mncontour('mu_ggF', 'mu_VBF', cl=0.68, size=100)
#pts_90 = NLLH.mncontour('mu_ggF', 'mu_VBF', cl=0.90, size=100)
pts_95 = NLLH.mncontour('mu_ggF', 'mu_VBF', cl=0.95, size=100)
#pts_99 = NLLH.mncontour('mu_ggF', 'mu_VBF', cl=0.99, size=100)
x_68, y_68 = np.transpose(pts_68)
#x_90, y_90 = np.transpose(pts_90)
x_95, y_95 = np.transpose(pts_95)
#x_99, y_99 = np.transpose(pts_99)

plt.figure(figsize=(13.3, 10.0))
plt.title('Contour Plot for $\mu_{ggF}$, $\mu_{VBF}$', fontsize=20)
plt.xlabel('$\mu_{ggF}$', fontsize=20)
plt.ylabel('$\mu_{VBF}$', fontsize=20)
plt.plot(x_68, y_68, '-', linewidth=2, color='blue', label='68% CL')
#plt.plot(x_90, y_90, '-', linewidth=2, color='darkgreen', label='90% CL')
plt.plot(x_95, y_95, '-', linewidth=2, color='red', label='95% CL')
#plt.plot(x_99, y_99, '-', linewidth=2, color='cyan', label='99% CL')
plt.plot(NLLH.values[0], NLLH.values[1], '+', markersize=6, color='black')
plt.grid()
plt.legend(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.94, 0.94)
plt.show()



################            Part c) Separate Measurements of mu_ggF, mu_VBF            ################
################            -----------------------------------------------            ################
"""
In interpreting 'separate', will fixed the other signal strength to it's SM predicted value (1.0) and perform the mnimisation for the signal strength of interest.

This will have the same effect as defining two new log-likelihoods where the background is now the combined counts of all that is not the count of interest.
-> E.g. if measuring mu_ggF, n_b_1_new = n_b_1 + n_VBF_1 ..., which is exactly the same as fixing mu_VBF to 1.0. 
"""
NLLH_ggF = Minuit(Neg_L_LH, mu_ggF=1.0, mu_VBF=1.0)

NLLH_VBF = Minuit(Neg_L_LH, mu_ggF=1.0, mu_VBF=1.0)

NLLH_ggF.fixed['mu_VBF'] = True

NLLH_ggF.migrad()
NLLH_ggF.hesse()
NLLH_ggF.minos()

NLLH_VBF.fixed['mu_ggF'] = True

NLLH_VBF.migrad()
NLLH_VBF.hesse()
NLLH_VBF.minos()


####    Manual 1D Profile Scans    ####
####    -----------------------    ####

Min_LH_ggF = Neg_L_LH(NLLH_ggF.values[0], NLLH_ggF.values[1])

Min_LH_VBF = Neg_L_LH(NLLH_VBF.values[0], NLLH_VBF.values[1])

ggF_LH_c = []

VBF_LH_c = []

for i in sweep:
    NLLH_muggF_c = Minuit(Neg_L_LH, mu_ggF=i, mu_VBF=1.0)

    NLLH_muggF_c.fixed['mu_ggF'] = True
    NLLH_muggF_c.fixed['mu_VBF'] = True

    NLLH_muggF_c.migrad()
    #NLLH_muggF_c.hesse()
    NLLH_muggF_c.minos()

    ggF_LH_c.append(Neg_L_LH(NLLH_muggF_c.values[0], NLLH_muggF_c.values[1]))


    NLLH_muVBF_c = Minuit(Neg_L_LH, mu_ggF=1.0, mu_VBF=i)

    NLLH_muVBF_c.fixed['mu_VBF'] = True
    NLLH_muVBF_c.fixed['mu_ggF'] = True

    NLLH_muVBF_c.migrad()
    #NLLH_muVBF_c.hesse()
    NLLH_muVBF_c.minos()

    VBF_LH_c.append(Neg_L_LH(NLLH_muVBF_c.values[0], NLLH_muVBF_c.values[1]))


Delta_muggF_NLLH_c = [val - Min_LH_ggF for val in ggF_LH_c]

Delta_muVBF_NLLH_c = [val - Min_LH_VBF for val in VBF_LH_c]

"""
Intersection Points
"""
ggF_c_x, ggF_c_y = intersection(sweep, 0.5*np.ones(len(sweep)), sweep, np.array(Delta_muggF_NLLH_c))

VBF_c_x, VBF_c_y = intersection(sweep, 0.5*np.ones(len(sweep)), sweep, np.array(Delta_muVBF_NLLH_c))

plt.figure(figsize=(15.0, 15.0))
plt.title('Profile Likelihood Plot for $\mu_{ggF}$ (Part C)', fontsize=20)
plt.xlabel('$\mu_{ggF}$', fontsize=20)
plt.ylabel('-$\Delta$ln($\mathcal{L}$)', fontsize=20)
plt.plot(sweep, Delta_muggF_NLLH_c, '-', linewidth=3, color='blue')
plt.vlines(NLLH_ggF.values[0], min(Delta_muggF_NLLH_c), max(Delta_muggF_NLLH_c), linestyles='dotted', colors='black')
plt.hlines(0.5, min(sweep), max(sweep), linestyles='dotted', colors='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.2, 0.2, 0.8, 0.8)
plt.show()

plt.figure(figsize=(15.0, 15.0))
plt.title('Profile Likelihood Plot for $\mu_{VBF}$ (Part C)', fontsize=20)
plt.xlabel('$\mu_{VBF}$', fontsize=20)
plt.ylabel('-$\Delta$ln($\mathcal{L}$)', fontsize=20)
plt.plot(sweep, Delta_muVBF_NLLH_c, '-', linewidth=3, color='blue')
plt.vlines(NLLH_VBF.values[1], min(Delta_muVBF_NLLH_c), max(Delta_muVBF_NLLH_c), linestyles='dotted', colors='black')
plt.hlines(0.5, min(sweep), max(sweep), linestyles='dotted', colors='black')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.2, 0.2, 0.8, 0.8)
plt.show()



################            Part d) Expected Uncertainties            ################
################            ------------------------------            ################
def Neg_L_LH_D(mu_ggF, mu_VBF):
    """
    Will include if statement to check for negative mu
    """
    mu_SR1 = (mu_ggF*n_ggF_SR1) + (mu_VBF*n_VBF_SR1) + n_b_SR1
    mu_SR2 = (mu_ggF*n_ggF_SR2) + (mu_VBF*n_VBF_SR2) + n_b_SR2
    if mu_SR1 < 0 or mu_SR2 < 0:
        return 1e8
    else:
        a = np.log(gamma(N_obs_SR1_D + 1)) + np.log(gamma(N_obs_SR2_D + 1))
        b = N_obs_SR1_D * np.log(mu_SR1)
        c = N_obs_SR2_D * np.log(mu_SR2)
        return mu_SR1 + mu_SR2 + a - b - c


NLLH_D = Minuit(Neg_L_LH_D, mu_ggF=1.0, mu_VBF=1.0)

NLLH_D.migrad()
NLLH_D.hesse()
NLLH_D.minos()

####    Manual 1D Profile Plots    ####
####    -----------------------    ####

Min_LH_D = Neg_L_LH_D(NLLH_D.values[0], NLLH_D.values[1])

sweep_D = np.linspace(0, 2.0, 100)

ggF_LH_D = []

VBF_LH_D = []

for i in sweep_D:
    NLLH_muggF_D = Minuit(Neg_L_LH_D, mu_ggF=i, mu_VBF=1.0)

    NLLH_muggF_D.fixed['mu_ggF'] = True

    NLLH_muggF_D.migrad()
    NLLH_muggF_D.hesse()
    NLLH_muggF_D.minos()

    ggF_LH_D.append(Neg_L_LH_D(NLLH_muggF_D.values[0], NLLH_muggF_D.values[1]))


    NLLH_muVBF_D = Minuit(Neg_L_LH_D, mu_ggF=1.0, mu_VBF=i)

    NLLH_muVBF_D.fixed['mu_VBF'] = True

    NLLH_muVBF_D.migrad()
    NLLH_muVBF_D.hesse()
    NLLH_muVBF_D.minos()

    VBF_LH_D.append(Neg_L_LH_D(NLLH_muVBF_D.values[0], NLLH_muVBF_D.values[1]))

Delta_muggF_NLLH_D = [val - Min_LH_D for val in ggF_LH_D]

Delta_muVBF_NLLH_D = [val - Min_LH_D for val in VBF_LH_D]


"""
Intersection Points
"""
ggF_D_x, ggF_D_y = intersection(sweep_D, 0.5*np.ones(len(sweep_D)), sweep_D, np.array(Delta_muggF_NLLH_D))

VBF_D_x, VBF_D_y = intersection(sweep_D, 0.5*np.ones(len(sweep_D)), sweep_D, np.array(Delta_muVBF_NLLH_D))


plt.figure(figsize=(15.0, 15.0))
plt.title('Profile Likelihood Plot for $\mu_{ggF}$ (Exp. Design)', fontsize=20)
plt.xlabel('$\mu_{ggF}$', fontsize=20)
plt.ylabel('-$\Delta$ln($\mathcal{L}$)', fontsize=20)
plt.plot(sweep_D, Delta_muggF_NLLH_D, '-', linewidth=2, color='blue')
plt.vlines(NLLH_D.values[0], min(Delta_muggF_NLLH_D), max(Delta_muggF_NLLH_D), linestyles='dotted', colors='black')
plt.hlines(0.5, min(sweep_D), max(sweep_D), linestyles='dotted', colors='black')
plt.vlines(ggF_D_x, ymin=0, ymax=0.5, linestyles='dotted', colors='black')
plt.ylim(0, None)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
plt.show()

plt.figure(figsize=(15.0, 15.0))
plt.title('Profile Likelihood Plot for $\mu_{VBF}$ (Exp. Design)', fontsize=20)
plt.xlabel('$\mu_{VBF}$', fontsize=20)
plt.ylabel('-$\Delta$ln($\mathcal{L}$)', fontsize=20)
plt.plot(sweep_D, Delta_muVBF_NLLH_D, '-', linewidth=2, color='blue')
plt.vlines(NLLH_D.values[1], min(Delta_muVBF_NLLH_D), max(Delta_muVBF_NLLH_D), linestyles='dotted', colors='black')
plt.hlines(0.5, min(sweep_D), max(sweep_D), linestyles='dotted', colors='black')
plt.vlines(VBF_D_x, ymin=0, ymax=0.5, linestyles='dotted', colors='black')
plt.ylim(0, None)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.1, 0.1, 0.9, 0.9)
plt.show()


####    Manual Contour Plots    ####
####    --------------------    ####

pts_68 = NLLH_D.mncontour('mu_ggF', 'mu_VBF', cl=0.68, size=100)
#pts_90 = NLLH_D.mncontour('mu_ggF', 'mu_VBF', cl=0.90, size=100)
pts_95 = NLLH_D.mncontour('mu_ggF', 'mu_VBF', cl=0.95, size=100)
#pts_99 = NLLH.mncontour('mu_ggF', 'mu_VBF', cl=0.99, size=100)
x_68, y_68 = np.transpose(pts_68)
#x_90, y_90 = np.transpose(pts_90)
x_95, y_95 = np.transpose(pts_95)
#x_99, y_99 = np.transpose(pts_99)

plt.figure(figsize=(13.3, 10.0))
plt.title('Contour Plot for $\mu_{ggF}$, $\mu_{VBF}$ (Exp. Design)', fontsize=20)
plt.xlabel('$\mu_{ggF}$', fontsize=20)
plt.ylabel('$\mu_{VBF}$', fontsize=20)
plt.plot(x_68, y_68, '-', linewidth=2, color='blue', label='68% CL')
#plt.plot(x_90, y_90, '-', linewidth=2, color='darkgreen', label='90% CL')
plt.plot(x_95, y_95, '-', linewidth=2, color='red', label='95% CL')
#plt.plot(x_99, y_99, '-', linewidth=2, color='cyan', label='99% CL')
plt.plot(NLLH_D.values[0], NLLH_D.values[1], '+', markersize=6, color='black')
plt.grid()
plt.legend(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.94, 0.94)
plt.show()



################            Printing to Terminal            ################
################            --------------------            ################
print('')
print('Problem 2 Part b)')
print('-----------------')
print('')
print(NLLH.params)
print('')
print('mu_ggF = ', NLLH.values[0], ' +', ggF_x[1] - NLLH.values[0], ' -', NLLH.values[0] - ggF_x[0])
print('')
print('mu_VBF = ', NLLH.values[1], ' +', VBF_x[1] - NLLH.values[1], ' -', NLLH.values[1] - VBF_x[0])
print('')
print('Correlation Matrix')
print('------------------')
print('')
print(NLLH.covariance.correlation())
print('')
print('Correlation Matrix')
print('------------------')
print('')
print(repr(NLLH.covariance.correlation()))
print('')


print('')
print('Problem 2 Part c)')
print('-----------------')
print('')
print('mu_ggF')
print('------')
print(NLLH_ggF.params)
print(NLLH_ggF.covariance.correlation())
print(repr(NLLH_ggF.covariance.correlation()))
print('')
print('mu_ggF = ', NLLH_ggF.values[0], ' +', ggF_c_x[1] - NLLH_ggF.values[0], ' -', NLLH_ggF.values[0] - ggF_c_x[0])
print('')
print('')
print('mu_VBF')
print('------')
print(NLLH_VBF.params)
print(NLLH_VBF.covariance.correlation())
print(repr(NLLH_VBF.covariance.correlation()))
print('')
print('mu_VBF = ', NLLH_VBF.values[1], ' +', VBF_c_x[1] - NLLH_VBF.values[1], ' -', NLLH_VBF.values[1] - VBF_c_x[0])
print('')


print('')
print('Problem 2 Part d)')
print('-----------------')
print('')
print(NLLH_D.params)
print('')
print('mu_ggF = ', NLLH_D.values[0], ' +', ggF_D_x[1] - NLLH_D.values[0], ' -', NLLH_D.values[0] - ggF_D_x[0])
print('')
print('mu_VBF = ', NLLH_D.values[1], ' +', VBF_D_x[1] - NLLH_D.values[1], ' -', NLLH_D.values[1] - VBF_D_x[0])
print('')
print('Correlation Matrix')
print('------------------')
print('')
print(NLLH_D.covariance.correlation())
print('')
print('Correlation Matrix')
print('------------------')
print('')
print(repr(NLLH_D.covariance.correlation()))
print('')
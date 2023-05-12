import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
import os

"""
Global Electroweak Fit - Very, Very Dodgy -> Maths and stuff look ok, results look a bit weird. ALso, how to incorprate Mt?

NB: This is the cleaner version of the code, with better defined variables and looping. Still need to implement autosave.

TODO: plt.ylim(-0.1, 9), M.errordef = Minuit.LIKELIHOOD

--> Since using -2ln(L), should be 1, 4, 9
"""

################            Global Parameters            ################
################            -----------------            ################
MW_Meas = 80.360 #From Latest ATLAS # 80.399 #GeV

Mt_Meas = 172.4 #GeV # Redundant

Mt_Exp = 172.8##174.3 #GeV

MZ_Meas = 91.1875 #GeV

s_Meas = 0.23148 # 0.23140 #From Online

MW_Uncerts = [0.01, 0.001, 0.0001] #*100%

Mt_Uncerts = [0.01, 0.001]

s_Rel_Uncerts = [1e-3, 1e-5]

##############          Part a) Likelihood Function only Mw            ################
##############          -----------------------------------            ################
def Mw(MH):
    MW_i, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = [80.3799, 0.05429, 0.008939, 0.0000890, 0.000161, 1.070, 0.5256, 0.0687, 0.00179, 0.0000659, 0.0737, 114.9]

    dH = np.log(MH / 100)
    dh = np.power((MH / 100), 2)
    dt = np.power((Mt_Exp / 174.3), 2) - 1
    dZ = (MZ_Meas / 91.1875) - 1
    d_a = ((314.19e-4 + 276.8e-4 - 0.7e-4) / 0.05907) - 1
    d_a_s = (0.1176 / 0.119) - 1

    mw = MW_i - c1*dH - c2*np.power(dH, 2) + c3*np.power(dH, 4) + c4*(dh - 1) - c5*d_a + c6*dt - c7*np.power(dt, 2) - c8*dH*dt + c9*dh*dt - c10*d_a_s + c11*dZ 

    return mw

MH_Scan = np.linspace(1, 200, 200)
ML_Scan_ALL = []

print('Part a)')
print('-------')

for i in MW_Uncerts:
    def NLLH(MH):
        return np.power((Mw(MH) - MW_Meas), 2) / np.power((i*MW_Meas), 2)

    M = Minuit(NLLH, MH=100)
    #M.errordef = Minuit.LIKELIHOOD
    M.limits['MH'] = (10, 1000)
    #M.errors = (10)
    M.migrad()
    M.hesse()
    #M.minos()

    print(f'{i*100}% Uncertainty on MW Measurement')
    print(M.params)
    print('')
    print('MH = ', M.values['MH'], ' +/- ', M.errors['MH'], ' GeV')
    print('')
    print('')
    print('')

    ML_Scan_ALL.append([NLLH(mh) for mh in MH_Scan])

plt.figure(figsize=(13.3, 10.0))
plt.title('Negative Log-Likelihood Scan vs M$_{H}$', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_ALL[0], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_ALL[1], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_ALL[2], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

print('')
print('')
print('')
print('')
print('')


##############          Part b) Likelihood Function but with Precision on mt            ################
##############          ----------------------------------------------------            ################
def Mw_b(MH, Mt):
    MW_i, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = [80.3799, 0.05429, 0.008939, 0.0000890, 0.000161, 1.070, 0.5256, 0.0687, 0.00179, 0.0000659, 0.0737, 114.9]

    dH = np.log(MH / 100)
    dh = np.power((MH / 100), 2)
    dt = np.power((Mt / 174.3), 2) - 1
    dZ = (MZ_Meas / 91.1875) - 1
    d_a = ((314.19e-4 + 276.8e-4 - 0.7e-4) / 0.05907) - 1
    d_a_s = (0.1176 / 0.119) - 1

    mw = MW_i - c1*dH - c2*np.power(dH, 2) + c3*np.power(dH, 4) + c4*(dh - 1) - c5*d_a + c6*dt - c7*np.power(dt, 2) - c8*dH*dt + c9*dh*dt - c10*d_a_s + c11*dZ 

    return mw

ML_Scan_b_ALL = []

print('Part b)')
print('-------')

for i in Mt_Uncerts:
    for j in MW_Uncerts:
        def NLLH(MH, Mt):
            return (np.power((MW_Meas - Mw_b(MH, Mt)), 2) / np.power((MW_Meas*j), 2)) + (np.power((Mt - Mt_Exp), 2) / np.power((Mt*i), 2))

        M = Minuit(NLLH, MH=100, Mt=Mt_Meas)

        #M.fixed['Mt'] = True
        M.limits['MH'] = (10, 1000)
        M.migrad()
        M.hesse()

        print(f'{i*100}% Uncertainty on Mt Measurement, {j*100}% Uncertainty on MW Measurement')
        print(M.params)
        print('')
        print('MH = ', M.values['MH'], ' +/- ', M.errors['MH'], ' GeV')
        print('')
        print('Mt = ', M.values['Mt'], ' +/- ', M.errors['Mt'], ' GeV')
        print('')
        print('')
        #print(Mw_b(M.values['MH'], M.values['Mt']))

        ## Shows as displayed.
        #M.draw_mnprofile('MH')
        #plt.show()

        temp_scan = []
        for k in MH_Scan:
            M_Scan = Minuit(NLLH, MH=k, Mt=Mt_Meas)
            M_Scan.fixed['MH'] = True
            M_Scan.migrad()
            M_Scan.hesse()
            temp_scan.append(NLLH(M_Scan.values['MH'], M_Scan.values['Mt']))
        
        ML_Scan_b_ALL.append(temp_scan)



plt.figure(figsize=(13.3, 10.0))
plt.title('Negative Log-Likelihood Scan vs M$_{H}$ (1% M$_{t}$ Uncert.)', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_b_ALL[0], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_b_ALL[1], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_b_ALL[2], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

plt.figure(figsize=(13.3, 10.0))
plt.title('Negative Log-Likelihood Scan vs M$_{H}$ (0.1% M$_{t}$ Uncert.)', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_b_ALL[3], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_b_ALL[4], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_b_ALL[5], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

## Shows disparity between Minuit and manual profiling
#print(ML_Scan_b_ALL[2])
#min_value = min(ML_Scan_b_ALL[2])
#min_index = ML_Scan_b_ALL[2].index(min_value)
#print(MH_Scan[min_index])



print('')
print('')
print('')
print('')
print('')



##############          Part c) Likelihood Function for sin^2(theta_eff)            ################
##############          ------------------------------------------------            ################
def sin_2_theta(MH):
    s0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = [0.2312527, 4.729e-4, 2.07e-5, 3.85e-6, -1.85e-6, 0.0207, -0.002851, 1.82e-4, -9.74e-6, 3.98e-4, -0.655]

    LH = np.log(MH / 100)
    delta_H = MH / 100
    delta_alpha = ((314.19e-4 + 276.8e-4 - 0.7e-4) / 0.05907) - 1  #(314.19e-4 + 276.8e-4 - 0.7e-4)
    delta_t = ((Mt_Exp / 178.0)**2) - 1
    delta_alpha_s = (0.1176 / 0.117) - 1
    delta_Z = (MZ_Meas / 91.1876) - 1
    
    sin_2_theta_c = s0 + d1*LH + d2*np.power(LH, 2) + d3*np.power(LH, 4) + d4*(np.power(delta_H, 2) - 1) + d5*delta_alpha + d6*delta_t + d7*np.power(delta_t, 2) + d8*delta_t*(delta_H - 1) + d9*delta_alpha_s + d10*delta_Z

    return sin_2_theta_c

print('Part c)')
print('-------')

ML_Scan_s_ALL = []

for i in s_Rel_Uncerts:
    def NLLH(MH):
        return np.power((sin_2_theta(MH) - s_Meas), 2) / np.power(i, 2)

    M = Minuit(NLLH, MH=100)
    M.limits['MH'] = (10, 1000)
    M.migrad()
    M.hesse()

    print(f'{i} Relative Uncertainty on sin^2(theta_eff) Measurements')
    print(M.params)
    print('')
    print('MH = ', M.values['MH'], ' +/- ', M.errors['MH'], ' GeV')
    print('')
    print('')
    print('')

    ML_Scan_s_ALL.append([NLLH(mh) for mh in MH_Scan])

plt.figure(figsize=(13.3, 10.0))
plt.title(r'Negative Log-Likelihood Scan sin$^{2}$($\theta_{eff}$) vs M$_{H}$', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_s_ALL[0], '-', color='blue', linewidth=2, label=r'10$^{-3}$ Rel. Precision on sin$^{2}$($\theta_{eff}$)')
plt.plot(MH_Scan, ML_Scan_s_ALL[1], '-', color='darkgreen', linewidth=2, label=r'10$^{-5}$ Rel. Precision on sin$^{2}$($\theta_{eff}$)')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22, loc=2)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

print('')
print('')
print('')
print('')
print('')



##############          Part d) Everything Everywhere All At Once            ################
##############          -----------------------------------------            ################
# Precise Mt
MW_Uncerts = [0.01, 0.001, 0.0001]
s_rel_uncert = [1e-3, 1e-5]

print('Part d)')
print('-------')
print('')

ML_Scan_Mt_Precise = []

for j in s_rel_uncert:
    for i in MW_Uncerts:
        def NLLH(MH):
            return (np.power((Mw(MH) - MW_Meas), 2) / np.power((MW_Meas*i), 2)) + (np.power((sin_2_theta(MH) - s_Meas), 2) / np.power((j), 2))

        M = Minuit(NLLH, MH=100)

        M.limits['MH'] = (10, 1000)
        M.migrad()
        M.hesse()

        print(f'Precise Mt, {i*100}% Uncertainty in MW, {j} Relative Uncertainty in sin^2(theta_eff)')
        print(M.params)
        print('')
        print('MH = ', M.values['MH'], ' +/- ', M.errors['MH'], ' GeV')
        print('')
        print('')
        print('')

        ML_Scan_Mt_Precise.append([NLLH(mh) for mh in MH_Scan])

print('')
print('')
print('')
print('')
print('')

ML_Scan_Mt_1 = []

# Uncert on Mt  ---  Here, need to write another sin_2_theta function with a free mt parameter as we are now allowing it to vary. ffs
def sin_2_theta_D(MH, Mt):
    s0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10 = [0.2312527, 4.729e-4, 2.07e-5, 3.85e-6, -1.85e-6, 0.0207, -0.002851, 1.82e-4, -9.74e-6, 3.98e-4, -0.655]

    LH = np.log(MH / 100)
    delta_H = MH / 100
    delta_alpha = ((314.19e-4 + 276.8e-4 - 0.7e-4) / 0.05907) - 1
    delta_t = ((Mt / 178.0)**2) - 1
    delta_alpha_s = (0.1176 / 0.117) - 1
    delta_Z = (MZ_Meas / 91.1876) - 1
    
    sin_2_theta_d = s0 + d1*LH + d2*np.power(LH, 2) + d3*np.power(LH, 4) + d4*(np.power(delta_H, 2) - 1) + d5*delta_alpha + d6*delta_t + d7*np.power(delta_t, 2) + d8*delta_t*(delta_H - 1) + d9*delta_alpha_s + d10*delta_Z

    return sin_2_theta_d



for j in s_rel_uncert:
    for i in MW_Uncerts:
        def NLLH(MH, Mt):
            return (np.power((Mw_b(MH, Mt) - MW_Meas), 2) / np.power((MW_Meas*i), 2)) + (np.power((sin_2_theta_D(MH, Mt) - s_Meas), 2) / np.power((j), 2)) + (np.power((Mt - Mt_Exp), 2) / np.power((Mt_Exp*0.01), 2))

        M = Minuit(NLLH, MH=100, Mt=172.8)

        #M.fixed['Mt'] = True
        M.limits['MH'] = (10, 1000)
        #M.errors = (100, 1)
        M.migrad()
        M.hesse()

        print(f'1% Uncertainty in Mt, {i*100}% Uncertainty in MW, {j} Relative Uncertainty in sin^2(theta_eff)')
        print(M.params)
        print('')
        print('MH = ', M.values['MH'], ' +/- ', M.errors['MH'], ' GeV')
        print('')
        print('Mt = ', M.values['Mt'], ' +/- ', M.errors['Mt'], ' GeV')
        print('')
        print('')

        temp_scan = []
        for k in MH_Scan:
            M_Scan = Minuit(NLLH, MH=k, Mt=Mt_Meas)
            M_Scan.fixed['MH'] = True
            M_Scan.migrad()
            M_Scan.hesse()
            temp_scan.append(NLLH(M_Scan.values['MH'], M_Scan.values['Mt']))
        
        ML_Scan_Mt_1.append(temp_scan)

        #ML_Scan_Mt_1.append([NLLH(mh, Mt=Mt_Meas) for mh in MH_Scan])


plt.figure(figsize=(13.3, 10.0))
plt.title(r'Negative Log-Likelihood Scan vs M$_{H}$ (Mt Precise, 10$^{-3}$ sin$^{2}$($\theta_{eff}$))', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_Mt_Precise[0], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_Precise[1], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_Precise[2], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

plt.figure(figsize=(13.3, 10.0))
plt.title(r'Negative Log-Likelihood Scan vs M$_{H}$ (Mt Precise, 10$^{-5}$ sin$^{2}$($\theta_{eff}$))', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_Mt_Precise[3], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_Precise[4], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_Precise[5], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

plt.figure(figsize=(13.3, 10.0))
plt.title(r'Negative Log-Likelihood Scan vs M$_{H}$ (1% Mt, 10$^{-3}$ sin$^{2}$($\theta_{eff}$))', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_Mt_1[0], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_1[1], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_1[2], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()

plt.figure(figsize=(13.3, 10.0))
plt.title(r'Negative Log-Likelihood Scan vs M$_{H}$ (1% Mt, 10$^{-5}$ sin$^{2}$($\theta_{eff}$))', fontsize=24)
plt.xlabel('M$_{H}$ [GeV]', fontsize=22, loc='right')
plt.ylabel('$\chi ^{2}$', fontsize=22, loc='top')
plt.plot(MH_Scan, ML_Scan_Mt_1[3], '-', color='blue', linewidth=2, label='1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_1[4], '-', color='darkgreen', linewidth=2, label='0.1% Precision on M$_{W}$')
plt.plot(MH_Scan, ML_Scan_Mt_1[5], '-', color='red', linewidth=2, label='0.01% Precision on M$_{W}$')
plt.ylim(-0.1, 9.5)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=22)
plt.subplots_adjust(0.1, 0.11, 0.9, 0.9)
plt.show()


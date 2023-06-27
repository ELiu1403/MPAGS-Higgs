import matplotlib.pyplot as plt
import numpy as np
import os
from iminuit import Minuit
import scipy
from scipy import integrate
import ROOT

"""
The Orsay Experiment
--------------------

TODO: Auto Save all plots
"""

################            Constants           ##################
################            ---------           ##################

Gf = 1.1664e-5 #GeV

m_e_GeV = 0.511e-3 #GeV

m_e_MeV = 0.511 #MeV

a_H = (m_e_GeV**2 * Gf * np.sqrt(2)) / (4 * np.pi)

FSC = 1 / (4*np.pi)**2

L = 2 #m - Length of the Experimental Hall


################            Part a) Higgs Acceptance vs Lifetime            ################
################            ------------------------------------            ################
H_mass = np.linspace(1, 52, 52) # Higgs masses in MeV (range as defined by paper)

def H_Lifetime(mass):
    """
    Returns the lifetime of a Higgs boson of specific mass, as given in the paper.
    """
    return ( 1 / (0.5 * a_H * mass * (1 - ((4 * m_e_MeV**2) / (mass**2)))**(1.5))) * 6.582e-22

H_time = [H_Lifetime(m) for m in H_mass]

def Avg_R(mass):
    """
    The classic <R> = bamma*beta*c*tau
    """
    tau = H_Lifetime(mass)

    b_y = np.sqrt(1600**2 - mass**2) / mass

    return b_y * 2.997e8 * tau

def Survive_Frac(mass):
    """
    N(length of cavern) / N0
    """
    return np.exp(-(L / Avg_R(mass)))

surviv_frac = [Survive_Frac(m) for m in H_mass]

plt.figure(figsize=(13.3, 10.0))
plt.title('Higgs Acceptance vs Lifetime', fontsize=20)
plt.xlabel(r'$\tau_{H}$' + ' [s]', fontsize=20, loc='right')
plt.ylabel('Acceptance', fontsize=20, loc='top')
plt.plot(H_time, surviv_frac, '-', linewidth=2, color='blue')
plt.xscale('log')
plt.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.9, 0.9)
plt.savefig(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part A.png')
plt.show()

################            Part b) Higgs Acceptance vs Mass            ################
################            --------------------------------            ################
plt.figure(figsize=(13.3, 10.0))
plt.title('Higgs Acceptance vs Mass', fontsize=20)
plt.xlabel(r'm$_{H}$' + ' [MeV]', fontsize=20, loc='right')
plt.ylabel('Acceptance', fontsize=20, loc='top')
plt.plot(H_mass, surviv_frac, '-', linewidth=2, color='blue')
plt.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.9, 0.9)
plt.savefig(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part B.png')
plt.show()


################            Part c) <Number of Events> vs Higgs Mass            ################
################            ----------------------------------------            ################
E0 = 1600 # Electon Beam Energy, MeV
EH = 1600 # Higgs Energy from Higgs Brenstrahlung, MeV
z = EH / E0 # Normalised Higgs Energy - Almost always 1

Z = 74 # Target Atomic Number

N_e = 2e16 # Number of electons delivered to dump

def f(mass, NHE):
    return ((mass**2) * (1-NHE)) / (m_e_MeV**2 * NHE**2)

def F(mass, screening):
    if screening == True:
        return np.log(184 * (Z)**(- 1 / 3))
    elif screening == False:
        if z == 1:
            return 0.5 # If this is -ve, get a negative number of Higgs...
        else:
            return np.log((2*E0*(1-z)) / (m_e_MeV * z * np.sqrt(1+f(mass, z)))) - 0.5

def diff_CS(mass, screening = True):
    return (((2 * FSC**2 * a_H * Z**2) / (m_e_MeV**2)) * ((z * (1 + ((2/3)*f(mass, z)))) / ((1 + f(mass, z))**2))) * F(mass, screening)
    #return (((2 * FSC**2 * a_H * Z**2) / (m_e_MeV**2)) * (z / ((1 + f(mass, z))**2))) * F(mass, screening)


N0_Screening = []

Raw_CS = []

for Hm in H_mass:
    DCS = lambda z: (((2 * FSC**2 * a_H * Z**2) / (m_e_MeV**2)) * z * ((1 + (2/3)*(Hm**2 / m_e_MeV**2)*((1-z)/z**2)) / (1 + ((Hm**2 / m_e_MeV**2)*((1-z)/z**2)))**2)) * np.log(184*np.power(Z, -1/3))

    CS = integrate.quad(DCS, 0, 1)

    Raw_CS.append(CS[0])

    #print(CS)

    N0_Screening.append(N_e * CS[0])

N_Screen = []

for i in range(len(surviv_frac)):
    N_Screen.append(N0_Screening[i] * surviv_frac[i])


# N0_No_Screening = [N_e * diff_CS(m, screening=False) for m in H_mass]
#
# N0_Screening = [N_e * diff_CS(m, screening=True) for m in H_mass]
#
# N_Screen = []
#
# N_No_Screen = []
#
# for i in range(len(surviv_frac)):
#     N_Screen.append(N0_Screening[i] * surviv_frac[i])
#     N_No_Screen.append(N0_No_Screening[i] * surviv_frac[i])
#

plt.figure(figsize=(13.3, 10.0))
plt.title('Cross-Section vs Higgs Mass', fontsize=20)
plt.xlabel(r'm$_{H}$' + ' [MeV]', fontsize=20, loc='right')
plt.ylabel(f'$\sigma$' + ' [A.U., for now]', fontsize=20, loc='top')
plt.plot(H_mass, Raw_CS, '-', linewidth=2, color='darkgreen', label='Complete Screening')
plt.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.9, 0.9)
plt.show()


plt.figure(figsize=(13.3, 10.0))
plt.title('Expected Number of Higgs vs Higgs Mass', fontsize=20)
plt.xlabel(r'm$_{H}$' + ' [MeV]', fontsize=20, loc='right')
plt.ylabel('Number of Events', fontsize=20, loc='top')
# plt.plot(H_mass, N_No_Screen, '-', linewidth=2, color='blue', label='No Screening')
plt.plot(H_mass, N_Screen, '-', linewidth=2, color='darkgreen', label='Complete Screening')
plt.grid()
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.9, 0.9)
plt.savefig(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part C.png')
plt.show()


################            Part d) Energy Distribution in Cherenkov Calo           ################
################            ---------------------------------------------           ################
np.random.seed(2)

####    Exploring Multiple Masses (Also GeV Corrected)    ####

Test_H_Masses = [0.01, 0.02, 0.03, 0.04, 0.05]

def Hee_GeV(mass):
    theta = np.arccos(np.random.uniform(-1, 1))

    # theta = np.random.uniform(-np.pi/2, np.pi/2)  # Change e) from and to or

    phi = np.random.uniform(-np.pi, np.pi)

    p_H = np.sqrt(1.6**2 - mass**2)

    H_4V = ROOT.TLorentzVector(0, 0, p_H, 1.6)

    H_Boost = H_4V.BoostVector()

    p_e = np.sqrt((0.5*mass)**2 - m_e_GeV**2)

    # Rest frame electrons
    e1 = ROOT.TLorentzVector(p_e*np.sin(theta)*np.cos(phi), p_e*np.sin(theta)*np.sin(phi), p_e*np.cos(theta), 0.5*mass)

    e2 = ROOT.TLorentzVector(-1*p_e*np.sin(theta)*np.cos(phi), -1*p_e*np.sin(theta)*np.sin(phi), -1*p_e*np.cos(theta), 0.5*mass)

    e1.Boost(H_Boost)

    e2.Boost(H_Boost)

    e1_Energy = e1.E()

    e2_Energy = e2.E()

    return [e1_Energy, e2_Energy, e1_Energy * np.random.normal(1, 0.107*np.sqrt(e1_Energy)), e2_Energy * np.random.normal(1, 0.107*np.sqrt(e2_Energy))]

    # Try:

# Old Smearing: np.random.normal(e1_Energy, 0.107*np.sqrt(e1_Energy))

for i in Test_H_Masses:
    # energies_1_NS = []
    # energies_2_NS = []
    # energies_1_S = []
    # energies_2_S = []
    hist_NS_V2_stack = ROOT.THStack('Stacked No Smear', 'Expected Calorimeter Energy Deposits for a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000))
    hist_high_NS = ROOT.TH1F('Electron 1 No Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)
    hist_low_NS = ROOT.TH1F('Electron 2 No Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)

    hist_S_V2_stack = ROOT.THStack('Stacked No Smear', 'Expected Calorimeter Energy Deposits for a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000))
    hist_high_S = ROOT.TH1F('Electron 1 Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)
    hist_low_S = ROOT.TH1F('Electron 2 Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)

    hist_NS_V2_tot = ROOT.TH1F('Energy Deposit', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)

    hist_S_V2_tot = ROOT.TH1F('Energy Deposit', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)

    hist_no_smear_stack = ROOT.THStack('Stacked No Smear', 'Expected Calorimeter Energy Deposits for a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000))
    hist_electron_1_NS = ROOT.TH1F('Electron 1 No Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)
    hist_electron_2_NS = ROOT.TH1F('Electron 2 No Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (No Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)

    hist_smear_stack = ROOT.THStack('Stacked Smear', 'Expected Calorimeter Energy Deposits for a {} MeV Higgs Boson (Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000))
    hist_electron_1_S = ROOT.TH1F('Electron 1 Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)
    hist_electron_2_S = ROOT.TH1F('Electron 2 Smear', 'Expected Electron Calorimeter Energy Deposits from a {} MeV Higgs Boson (Smearing);Energy [GeV];Entries / 0.01 GeV'.format(i*1000), 250, 0.0, 2.5)

    for j in range(5000):
        e1_NS, e2_NS, e1_S, e2_S = Hee_GeV(i)

        # energies_1_NS.append(e1_NS)
        # energies_2_NS.append(e2_NS)
        # energies_1_S.append(e1_S)
        # energies_2_S.append(e2_S)
        hist_electron_1_NS.Fill(e1_NS)
        hist_electron_2_NS.Fill(e2_NS)
        hist_electron_1_S.Fill(e1_S)
        hist_electron_2_S.Fill(e2_S)

        hist_NS_V2_tot.Fill(e1_NS + e2_NS)
        hist_S_V2_tot.Fill(e1_S + e2_S)


        if e1_NS > e2_NS:
            hist_high_NS.Fill(e1_NS)
            hist_low_NS.Fill(e2_NS)
        else:
            hist_high_NS.Fill(e2_NS)
            hist_low_NS.Fill(e1_NS)

        if e1_S > e2_S:
            hist_high_S.Fill(e1_S)
            hist_low_S.Fill(e2_S)
        else:
            hist_high_S.Fill(e2_S)
            hist_low_S.Fill(e1_S)


    c1 = ROOT.TCanvas('c1', 'c1', 1600, 1000)
    hist_electron_1_NS.Draw('HIST')
    hist_electron_1_NS.GetXaxis().SetRangeUser(-0.05, 1.8)
    hist_electron_1_NS.SetFillColor(4)
    hist_electron_1_NS.SetLineColor(4)
    c1.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/E1 {} MeV Higgs (No Smear).png'.format(i*1000))

    c2 = ROOT.TCanvas('c2', 'c2', 1600, 1000)
    hist_electron_2_NS.Draw('HIST')
    hist_electron_2_NS.GetXaxis().SetRangeUser(-0.05, 1.8)
    hist_electron_2_NS.SetFillColor(2)
    hist_electron_2_NS.SetLineColor(2)
    c2.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/E2 {} MeV Higgs (No Smear).png'.format(i*1000))

    c3 = ROOT.TCanvas('c3', 'c3', 1600, 1000)
    leg_NS = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    hist_electron_1_NS.SetFillStyle(3001)
    hist_electron_2_NS.SetFillStyle(3001)
    leg_NS.AddEntry(hist_electron_1_NS, 'Electron', 'f')
    leg_NS.AddEntry(hist_electron_2_NS, 'Electron', 'f')
    hist_no_smear_stack.Add(hist_electron_1_NS)
    hist_no_smear_stack.Add(hist_electron_2_NS)
    hist_no_smear_stack.Draw('HIST, nostack')
    hist_no_smear_stack.GetXaxis().SetRangeUser(-0.05, 1.8)
    leg_NS.Draw()
    c3.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/Combined {} MeV Higgs (No Smear).png'.format(i*1000))

    c4 = ROOT.TCanvas('c4', 'c4', 1600, 1000)
    hist_electron_1_S.Draw('HIST')
    # hist_electron_1_S.GetXaxis().SetRangeUser(-0.05, 2.0)
    hist_electron_1_S.SetFillColor(4)
    hist_electron_1_S.SetLineColor(4)
    c4.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/E1 {} MeV Higgs (Smear).png'.format(i*1000))

    c5 = ROOT.TCanvas('c5', 'c5', 1600, 1000)
    hist_electron_2_S.Draw('HIST')
    # hist_electron_2_S.GetXaxis().SetRangeUser(-0.05, 2.0)
    hist_electron_2_S.SetFillColor(2)
    hist_electron_2_S.SetLineColor(2)
    c5.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/E2 {} MeV Higgs (Smear).png'.format(i*1000))

    c6 = ROOT.TCanvas('c6', 'c6', 1600, 1000)
    leg_S = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    hist_electron_1_S.SetFillStyle(3001)
    hist_electron_2_S.SetFillStyle(3001)
    leg_S.AddEntry(hist_electron_1_S, 'Electron', 'f')
    leg_S.AddEntry(hist_electron_2_S, 'Electron', 'f')
    hist_smear_stack.Add(hist_electron_1_S)
    hist_smear_stack.Add(hist_electron_2_S)
    hist_smear_stack.Draw('HIST, nostack')
    # hist_smear_stack.GetXaxis().SetRangeUser(-0.05, 2.0)
    leg_S.Draw()
    c6.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/Combined {} MeV Higgs (Smear).png'.format(i*1000))

    c7 = ROOT.TCanvas('c7', 'c7', 1600, 1000)
    leg_hl_NS = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    hist_high_NS.SetFillColor(4)
    hist_high_NS.SetFillStyle(3001)
    hist_low_NS.SetFillColor(2)
    hist_low_NS.SetFillStyle(3001)
    leg_hl_NS.AddEntry(hist_high_NS, 'Electron', 'f')
    leg_hl_NS.AddEntry(hist_low_NS, 'Electron', 'f')
    hist_NS_V2_stack.Add(hist_high_NS)
    hist_NS_V2_stack.Add(hist_low_NS)
    hist_NS_V2_stack.Draw('HIST, nostack')
    leg_hl_NS.Draw()
    c7.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/High Low Combined {} MeV Higgs (No Smear).png'.format(i*1000))

    c8 = ROOT.TCanvas('c8', 'c8', 1600, 1000)
    leg_hl_S = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
    hist_high_S.SetFillColor(4)
    hist_high_S.SetFillStyle(3001)
    hist_low_S.SetFillColor(2)
    hist_low_S.SetFillStyle(3001)
    leg_hl_S.AddEntry(hist_high_S, 'Electron', 'f')
    leg_hl_S.AddEntry(hist_low_S, 'Electron', 'f')
    hist_S_V2_stack.Add(hist_high_S)
    hist_S_V2_stack.Add(hist_low_S)
    hist_S_V2_stack.Draw('HIST, nostack')
    leg_hl_S.Draw()
    c8.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/High Low Combined {} MeV Higgs (Smear).png'.format(i*1000))

    c9 = ROOT.TCanvas('c9', 'c9', 1600, 1000)
    hist_NS_V2_tot.Draw('HIST')
    hist_NS_V2_tot.SetFillColor(4)
    hist_NS_V2_tot.SetLineColor(4)
    c9.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/Total {} MeV Higgs (No Smear).png'.format(i*1000))

    c10 = ROOT.TCanvas('c10', 'c10', 1600, 1000)
    hist_S_V2_tot.Draw('HIST')
    hist_S_V2_tot.SetFillColor(4)
    hist_S_V2_tot.SetLineColor(4)
    c10.SaveAs(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part D/Total {} MeV Higgs (Smear).png'.format(i*1000))

################            Part e) Expected Number of H, f(mH), require 750MeV in Calo            ###############
################            -----------------------------------------------------------            ###############
# For the calo requirement, require that both electrons have an energy above the cut.
Events_e = []

for m in range(2, 50):
    count = 0

    DCS = lambda z: ((2 * FSC**2 * a_H * Z**2) / (m_e_MeV**2)) * z * ((1 + (2/3)*(m**2/m_e_MeV**2)*((1-z)/z**2)) / (1 + ((m**2 / m_e_MeV**2)*((1-z)/z**2)))**2) * np.log(184*np.power(Z, -1/3))

    CS = integrate.quad(DCS, 0, 1)

    n0 = N_e * CS[0]

    surviv_no = n0 * Survive_Frac(m)

    for i in range(int(surviv_no)):
        e1_NS, e2_NS, e1_S, e2_S = Hee_GeV(m/1e3)

        if e1_S > 0.75 and e2_S > 0.75:
            count += 1
    Events_e.append(count)

H_masses_e = np.linspace(2, 50, len(Events_e))

plt.figure(figsize=(13.3, 10.0))
plt.title('Expected Number of Higgs with a Calorimeter Cut', fontsize=20)
plt.xlabel(r'm$_{H}$' + ' [MeV]', fontsize=20, loc='right')
plt.ylabel('Entries', fontsize=20, loc='top')
plt.plot(H_masses_e, Events_e, '-', color='blue', linewidth=2, label='Complete Screening')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.subplots_adjust(0.15, 0.15, 0.9, 0.9)
plt.savefig(r'/home/ehl857/Documents/MPAG HIGGS/Higgs Problem 3 Plots/Final Plots/Part E.png')
plt.show()
plt.clf()
plt.close()

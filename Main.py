# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 13:56:18 2021

@author: Keyton Feller
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
os.chdir(os.path.dirname(__file__)) # set current working directory to this files save location
from Torch_Ray_Tracer import MCRT
from ZnO_Storage import ZnO_Data
#%%
T0 = time.perf_counter() # starts timer
# Number of rays to be launched in one batch
# Limit is based on hardware (VRAM) (500k with GTX 2060)
rays = 500000
# Dimensions of the 2D vat
vatx, vatz = 1000, 1000
# Size of each voxel/pixel in the simulation
mesh_size = 1.1
# Peak intensity of the intensity distribution mW/cm^2
Max_Int = 6.82
# CSV file that contains the properties of the resin and particles, see example, , for format of CSV.
# Do not change names of colums as that is the feature the code looks for
Spec_Sheet = 'ZnO Sheet.csv'
# filler for saving data
mat = 'ZnO'
# Spatial Intensity distribution, see example, , for format
SpaceI = np.array(pd.read_csv('7pixline.csv',encoding='latin-1', engine='python', header=None))
# list of volume loadings to be tested
vols = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
# debye length of the particles,
# If particle don't have charge, debye goes to infinity, 1000 is big enough as substitute
debyes = [1000]
# loop through debye lengths and volume loadings to be tested
for debye in debyes:
    for v in vols:
        # number of trials to run: trials * rays = number of rays launched
        trials = 1
        # list of simulation runtimes
        simtimes = []
        # generates a zero array for the simulation vat in shape of vatz and vaty
        Vat_total = np.zeros((vatz,vatx))
        # Phase_Calc value determines if the simulation look up table is to be generated
        # The look-up table stores all the Mie Theory calculations so the simulation doesn't have to calculate it for each event
        # Must be set to True if this is the first time running the material through the simulation
        Phase_Calc = True
        for i in range(trials):
            # MCRT.Spec_Sheet pulls the resin parameters of the Spec_sheet csv file
            Specs = MCRT.Sheet_Spec(Spec_Sheet,vol=v)
            vol, Dps, par_n, med_n, par_k, radi, radi_prob, wave, wave_prob = Specs
            # Trace_Rays runs the MCRT simulation
            Vat, phase_time, sim_time = MCRT.Trace_Rays(rays, vol, Specs, SpaceI, vatx, vatz, mesh_size, Max_Int, debye, mat, Phase_Calc=Phase_Calc)
            # sets Phase_Calc  to false so the look up table isn't re-generated for each trial
            Phase_Calc = False
            # Add the simulated vat to the vat total
            Vat_total += Vat
            # record the simulation time
            simtimes.append(sim_time)
        print(f'Total runtime: {np.sum(np.array(simtimes))} s')
        # Normalizes the vat by the trials
        Vat = Vat_total/trials
        # removes edges of simulation to avoid simulation edge effects/artifact
        Vat = Vat[:-2, 2:-2]
        # defines the working directory as the location of the energy distribution folders (where the simulated vat is saved)
        # change to where user wants simulation data saved
        os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO\Energy Dists')
        # saves the simulated vat
        np.savetxt(f'{vol*100}_{debye}_run.csv', Vat)
        # working directory set to Sets back to where this script is located
        os.chdir(os.path.dirname(__file__))
        print(f'{vol*100}_{debye}_run.csv')
# Generates simulation run time
T1 = time.perf_counter()
print(f'Run Time: {T1 - T0}')


#%%
os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO\Energy Dists')
def read(vol, debye, mesh_size):
    cwd = os.getcwd()
    for file in os.listdir(cwd):
        filename = os.fsdecode(file)
        v, deb, run = filename.split('_')
        if v == str(vol*100) and deb == str(debye):
            data = np.array(pd.read_csv(filename, header=None, delimiter=' '))
            # data = ((data/mesh_size)*(10**4))**2
            data = ((data/mesh_size**2)*(10**8))
            cen = int(np.where(data[0] == np.amax(data[0]))[0])
            data = data[:500, cen-200:cen+200]
    return data
            
def find_cure(Time, data, mesh_size, Ec):
    Cds = []
    Wds = []
    for i, t in enumerate(Time):
        dist = data * t
        dist[dist < Ec] = 0
        dist[dist > Ec] = 1
        ind = np.nonzero(dist[0])
        cen = int(np.floor(np.average(ind)))
        Cd = len(np.nonzero(dist[:,cen])[0]) * 1.1
        Wd = len(np.nonzero(dist[4])[0]) * 1.1 
        Cds.append(np.round(Cd, 1))
        Wds.append(np.round(Wd, 1))
    return Cds, Wds      


vols = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

debye = 1000
mesh_size = 1.1
Ec = 12.93
E0 = 6.82

Cds = []
# fig, ax = plt.subplots(2,2, figsize=(12, 8))
fig, ax = plt.subplots(2,3, figsize=(15,10))

ax[1][2].set_visible(False)

ax[1][0].set_position([0.24,0.125,0.228,0.343])
ax[1][1].set_position([0.55,0.125,0.228,0.343])
row = [0,0,0,1,1,1]
col = [0,1,2,0,1,2]

line_labels = ['Experimental C$_d$',
               'Experimental W$_{ex}$',
               'Simulated C$_d$',
               'Simulated W$_{ex}$']

Cd_errors = np.array([])
Cd_errors_std = np.array([])
Wd_errors = np.array([])
Wd_errors_std = np.array([])
Dps = np.array([])
Dp_error = np.array([])

for i, v in enumerate(vols):
    # Ec_i = Ec * (1-v)
    r = row[i]
    c = col[i]
    Time, Eng, Cd, Cd_std, Wd, Wd_std = ZnO_Data(v, E0)
    Wd = (Wd - 248.5)/2
    x = np.log(Eng)
    
    ax[r,c].set_ylim(5,210)
    ax[r,c].errorbar(x, Cd, yerr=Cd_std, color='red', marker='o', linestyle='')
    m,b = np.polyfit(x, Cd, 1)
    E_Dp = m
    # print(E_Dp)
    line = (m * x) + b
    l1 = ax[r,c].plot(x, line, linestyle='--', color='red')
    E_Ec = np.exp(-b/m)
    # print(E_Ec)
    # print(v)
    # print(m)
    
    ax[r,c].errorbar(np.log(Eng), Wd, yerr=Wd_std, color='orange', marker='o', linestyle='')
    m, b = np.polyfit(x, Wd, 1)
    line = (m * x) + b
    l2 = ax[r,c].plot(x, line, linestyle='--', color='orange')
    
    data = read(v, debye, mesh_size)
    
    # find Cd and Wd values
    
    Cd_m, Wd_m = find_cure(Time, data, mesh_size, Ec)
    # print((Wd_m[0]-248.5)/2)
    m,b = np.polyfit(x, Cd_m, 1)
    S_Dp = m
    # print(S_Dp)
    S_Ec = np.exp(-b/m)
    # print(S_Ec)
    print((np.abs(E_Dp - S_Dp)/S_Dp)*100)
    print((np.abs(E_Ec - S_Ec)/S_Ec)*100)
    print()
    Wd_m = (np.array(Wd_m)-248.5)/2
    
    
    # print(m)
    # print()
    
    
    # plot working cures with error form Ec
    l3 = ax[r,c].plot(x, Cd_m, color='blue')
    l4 = ax[r,c].plot(x, Wd_m, color='green')
    
    
    # save differences 
    Cd_error = np.average((Cd - Cd_m)/Cd_m) * 100
    Cd_errors = np.append(Cd_errors, Cd_error)
    Cd_err_std = np.std((Cd - Cd_m)/Cd_m) * 100
    Cd_errors_std = np.append(Cd_errors_std, Cd_err_std)
    
    Wd_error = np.average((Wd - Wd_m)/Wd_m) * 100
    Wd_errors = np.append(Wd_errors, Wd_error)
    Wd_err_std = np.std((Wd - Wd_m)/Wd_m) * 100
    Wd_errors_std = np.append(Wd_errors_std, Wd_err_std)
    
    Dp_error = np.append(Dp_error, (E_Dp - S_Dp)/S_Dp)
    
# fig.supxlabel('ln(Energy Dose)')
# fig.supylabel('Cure Distance ($\mu$m)')
fig.text(0.5, 0.07, 'ln(Exposure)', ha='center', fontsize=15)
fig.text(0.07, 0.5, 'Cure Distance ($\mu$m)', va='center', rotation='vertical', fontsize=15)
fig.legend([l1, l2, l3, l4],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper center",   # Position of legend
           borderaxespad=0.7,    # Small spacing around legend box
           ncol = 2,
           fontsize = 15
           )


#%%
plt.errorbar(vols*100, np.abs(Wd_errors), yerr=Wd_errors_std, label='Cure Width')
plt.errorbar(vols*100, np.abs(Cd_errors), yerr=Cd_errors_std, label='Cure Depth')
plt.xlabel('SBR Vol%')
plt.ylabel(r'$\frac{Simulation-Experimental}{Experimental}$ X 100%')
plt.legend(loc=0)


#%%
os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO')
from Vector_Storage import Scan_Data
os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO\Energy Dists')

def read(filename):
    data = np.array(pd.read_csv(filename, header=None, delimiter=' '))
    data = ((data/1.1**2)*(10**8))
    # cen = int(np.where(data[0] == np.amax(data[0]))[0])
    cen = int(np.average(np.nonzero(data[0])))
    print(cen)
    cen = 465
    # data = data[:180, cen-170:cen+170]
    data = data[:180, cen-190:cen+190]
    return data
    
v = 0.01
debye = 1000
mesh_size = 1.1
Ec = 12.93
E0 = 6.82
h = [1.0, 2.0, 3.0, 4.0, 5.0]
vols = [0.05]
fig,ax = plt.subplots(1,1, figsize=(10,4), sharex=False, sharey=True, constrained_layout=True)
for i, v in enumerate(vols):
    os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO\Energy Dists')
    data = read(f'{h[i]}_1000_run.csv')
    os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO')
    Time, Eng, Cd, Cd_std, Wd, Wd_std = ZnO_Data(v,E0)
    
    eng = data * Time[4]
    data[0,0] = 20
    # pos = ax.imshow(eng, extent=(-187,187,198,0))
# print(Time[0])
# print(np.amax(eng))
    eng[eng < Ec] = np.nan
    ax.imshow(eng,extent=(-187,187,198,0))
    
# cbar = fig.colorbar(pos, ax=ax, location='top', shrink=0.5, orientation='horizontal')
# cbar.set_label('Exposure (mJ/cm$^2$)')


fig.text(0.52, -0.03, 'XY Direction ($\mu$m)', ha='center')
fig.text(0.1, 0.47, 'Z Direction ($\mu$m)', va='center', rotation='vertical')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
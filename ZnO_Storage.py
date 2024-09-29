import numpy as np
from scipy import stats
import pandas as pd
import os
import matplotlib.pyplot as plt
os.environ['PATH']


def ZnO_Data(vol, E0):
    if vol == 0.0:
        Time = np.array([3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75])
        Cd = np.array([126.9, 168.3, 230.5, 256.3, 279.2, 353.8, 371.9, 409.9, 442.7, 472.8])
        Cd_std = np.array([5.3, 5.7, 12.3, 5.2, 7.3, 19.0, 5.0, 6.1, 8.6, 6.6])
        width = np.array([0])
        width_std = np.array([0])
    if vol == 0.01:
        Time = np.array([5,6,7,8,9,10,11])
        Cd = np.array([136.8, 141.4, 150.6, 176.4, 180.3, 191.7, 201.7])
        Cd_std = np.array([2.4, 2.2, 3.4, 5.7, 3.3, 2.1, 4.8])
        width = np.array([276.4, 290.2, 308.8, 335.9, 338.8, 354.6, 360.5])
        width_std = np.array([3.6, 4.1, 3.4, 2.9, 3.7, 3.7, 4.9])
    if vol == 0.02:
        Time = np.array([5,6,7,8,9,10,11])
        Cd = np.array([115.7, 127.1, 139.6, 150.7, 155.0, 161.0, 162.4])
        Cd_std = np.array([2.2, 4.8, 6.0, 6.2, 1.5, 5.0, 5.2])
        width = np.array([285.5, 298.5, 311.1, 322. , 338.4, 352.7, 358.5])
        width_std = np.array([6.0, 3.4, 3.0, 8.7, 3.1, 6.1, 3.4])
    if vol == 0.03:
        Time = np.array([5,6,7,8,9,10,11])
        Cd = np.array([103.3, 114.2, 121.3, 135.0, 143.3, 153.3, 160.9])
        Cd_std = np.array([3.7, 2.5, 4.4, 4.5, 5.9, 4.8, 3.4])
        width = np.array([287.4, 317.4, 322.5, 351.2, 357.2, 364.9, 370.9])
        width_std = np.array([3.4, 3.3, 2.2, 4.3, 4.4, 2.8, 4.0])
    if vol == 0.04:
        Time = np.array([5,6,7,8,9,10,11])
        Cd = np.array([ 88.8, 100.2,  107.4, 119.1, 124.3, 133.5, 140.0])
        Cd_std = np.array([4.6, 2.7, 4.0, 3.4, 6.8, 3.4, 3.3])
        width = np.array([288.9, 309.6, 318.8, 339.2, 346.1, 352.4, 361.5])
        width_std = np.array([3.9, 3.4, 4.6, 5.6, 6.7, 2.0, 4.9])
    if vol == 0.05:
        Time = np.array([5,6,7,8,9,10,11])
        Cd = np.array([74.2, 86.2, 93.4, 103.2, 105.2, 113.6, 119.2])
        Cd_std = np.array([5.4, 2.8, 3.5, 2.3, 7.6, 1.9, 3.1])
        width = np.array([300.3, 311.7, 325.1, 337.2, 344.9, 349.9, 362.0])
        width_std = np.array([4.3, 3.5, 7.0, 6.8, 8.9, 1.2, 5.8])
    Eng = E0 * Time
    m, b = np.polyfit(np.log(Eng),Cd,deg=1)
    Dp = m
    Ec = np.exp(-b/Dp)
    return Time, Eng, Cd, Cd_std, width, width_std

vols = [0.01, 0.02, 0.03, 0.04, 0.05]
Dps = []
Ecs = []
for vol in vols:
    Time, Eng, Cd, Cd_std, width, width_std = ZnO_Data(vol, 6.2)
    lnE = np.log(Eng)
    # plt.errorbar(lnE, Cd, yerr=Cd_std, linestyle='', marker='o')
    m,b = np.polyfit(lnE,Cd,deg=1)
    line = m*lnE + b
    Dp = m
    Ec = np.exp(b/-Dp)
    Dps.append(Dp)
    Ecs.append(Ec)
    
plt.plot(vols, Dps)
#%%
Time, Eng, Cd, Cd_std, width, width_std = ZnO_Data(0.02, 6.2)
lnE = np.log(Eng)
plt.errorbar(lnE, Cd, yerr=Cd_std, linestyle='', marker='o')
m,b = np.polyfit(lnE,Cd,deg=1)
line = m*lnE + b
plt.plot(lnE, line, linestyle='--')
plt.xlabel('ln(Exposure)')
plt.ylabel('Cure Depth ($\mu$m)')


    


# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:40:25 2021

@author: Keyton Feller
"""

import numpy as np
import pandas as pd
from numba import njit
import time
from scipy import integrate
import os 
import torch
os.chdir(os.path.dirname(__file__))
from Mie_Calculator import Mie_Calc
from lineDraw_V2 import Lines

class MCRT:
    @njit()
    def Scat_Var(wave, radi, ang, Phase):
        mu_s_all, mu_a_all = np.zeros((len(wave),len(radi))), np.zeros((len(wave),len(radi)))
        phase_all = np.zeros((len(wave),len(radi),len(ang)))
        row = 0
        for lam in range(len(wave)):
            for rad in range(len(radi)):
                phase_all[lam,rad] = Phase[row, 4:]
                mu_s_all[lam,rad] =  Phase[row, 2]
                mu_a_all[lam,rad] =  Phase[row, 3]
                row += 1 
        return phase_all, mu_s_all, mu_a_all
    
    def Filter_Rays(x0, x1, z0, z1, E0, E1, Ecut, Dp, mesh_size, vatx, vatz):

        mx0 = torch.floor(x0/mesh_size)
        mx1 = torch.floor(x1/mesh_size)
        mz0 = torch.floor(z0/mesh_size)
        mz1 = torch.floor(z1/mesh_size)
        
        mx1[mx1>=vatx-1] = vatx-1
        mz1[mz1>=vatz-1] = vatz-1
        mx1[mx1<0] = 0
        mz1[mz1<0] = 0
        
        Elow = E1 >= Ecut
        xlow = mx1 > 0
        xhig = mx1 <= vatx-1
        zlow = mz1 > 0
        zhig = mz1 <= vatz-1
        
        inds = Elow & xlow & xhig & zlow & zhig        
        inds = torch.where(inds == True)[0]
        
        mx0 = mx0.reshape(-1,1)                                    
        mx1 = mx1.reshape(-1,1)   
        mz0 = mz0.reshape(-1,1)   
        mz1 = mz1.reshape(-1,1)   
        mE0 = E0.reshape(-1,1)   
        mDp = (Dp/mesh_size).reshape(-1,1)
        dist1 = (((mx1-mx0)**2) + ((mz1-mz0)**2))**0.5
        dist0 = dist1 * 0

        vec1 = torch.hstack((mx0, mz0, dist0, mE0, mDp))
        vec2 = torch.hstack((mx1, mz1, dist1, mE0, mDp))
        ray_data = torch.hstack((vec1, vec2))
        ray_data.reshape(-1, 2, 5)

        return inds, ray_data
    
    def Sheet_Spec(Specs,vol):
        grid = pd.read_csv(Specs)
        # Source spectral intensity distribution
        wave = np.array(grid['Wavelength'][~np.isnan(grid['Wavelength'])]).astype(int)
        wave_prob = np.array(grid['Wave_I'][~np.isnan(grid['Wave_I'])])
        wave_prob = wave_prob/np.sum(wave_prob)
        # Medium penatration depths for all wavelengths used 
        Dp = np.array(grid['Dp'][~np.isnan(grid['Dp'])])
        # DLS particle radii distrinution and mean radius
        radi = np.array(grid['Radi'][~np.isnan(grid['Radi'])])
        radi_I = np.array(grid['Radi_I'][~np.isnan(grid['Radi_I'])])
        radi_I = radi_I/np.sum(radi_I)
        # Refractive index values of particle (complex) and medium for all wavelengths used 
        RI_par = np.array(grid['np'][~np.isnan(grid['np'])])
        kp = np.array(grid['kp'][~np.isnan(grid['kp'])])
        RI_med = np.array(grid['nm'][~np.isnan(grid['nm'])])
        return vol, Dp, RI_par, RI_med, kp, radi, radi_I, wave, wave_prob
        
    
    def Trace_Rays(rays, vol, Specs, spaceI, vatx, vatz, mesh_size, Max_Int, debye, mat, Phase_Calc=True):
        T0 = time.perf_counter() # starts timer
        # checks if systems has cuda enabled GPU, if not code runs on CPU
        if torch.cuda.is_available():
            pro = torch.device("cuda:0")
        else:
            pro = torch.device("cpu")
        
        vol, Dps, par_n, med_n, par_k, radi, radi_prob, wave, wave_prob = Specs 

        if Phase_Calc == True:
            print('Creating Phasers')
            ang = Mie_Calc.Mie_Phase(radi,radi_prob, wave, par_n, par_k, med_n, vol, mat, debye, Dps)
        
        Dps = torch.from_numpy(Dps).to(pro)
        os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO\Phasers')
        Phase = pd.read_csv(f'{vol}_{debye}_Phasers.csv')
        os.chdir(os.path.dirname(__file__))
        
        ang = np.array(Phase.columns[4:]).astype('float64') 
        Phase = np.array(Phase)
        phase_all, mu_s_all, mu_a_all = MCRT.Scat_Var(wave, radi, ang, Phase)

        space = 4.4

        div = int(np.round(space/mesh_size))
        mesh_size = space/div
        print(f'{rays} rays')
        print(f'{vol*100} vol%')

        Energy = (Max_Int) * (10**-8) * (mesh_size**2)
        spaceI = spaceI.flatten() / np.amax(spaceI.flatten())
        Energy = spaceI * Energy
        spaceI = np.repeat(Energy, div)

        space = (np.arange(len(spaceI)) * mesh_size) + (np.floor(vatx / 3))

        E_tot = integrate.simps(spaceI, space)
        spaceI = spaceI/np.sum(spaceI)

        x0 = np.random.choice(space, size=rays, p=spaceI)

        x0 = torch.from_numpy(x0).to(pro)
        
        z0 = torch.zeros((rays,)).to(pro)
        E_tot = torch.as_tensor(E_tot).to(pro)
        E0 = torch.zeros((rays,)).to(pro) + (E_tot/rays)
        Ecut = E0[0] * 0.01
        
        del spaceI, space, Phase
        
        lam_ind = torch.as_tensor(np.random.choice(np.arange(0,wave.size), size=rays, p=wave_prob),dtype=torch.int64).to(pro)
        par_k = torch.from_numpy(par_k).to(pro)
        lam = torch.from_numpy(wave)[lam_ind].to(pro)
        Dp = torch.as_tensor(Dps)[lam_ind].to(pro)
        phase_all = torch.tensor(phase_all).to(pro)
        radi_prob = torch.tensor(radi_prob).to(pro)
        mu_s_all = torch.tensor(mu_s_all).to(pro)
        mu_a_all = torch.tensor(mu_a_all).to(pro)
        
        T1 = time.perf_counter()
        phase_time = T1-T0
        print(f'Phaser Time: {T1-T0}')
        
        T0 = time.perf_counter()
        
        ang = torch.from_numpy(ang).to(pro)
        radi = torch.from_numpy(radi).to(pro)
        rad_index = torch.arange(len(radi)).to(pro)
        del radi
        rays = torch.tensor(rays).to(pro)
        mesh_size = torch.tensor(mesh_size).to(pro)
        wave = torch.from_numpy(wave).to(pro)
        
        j = 0
        while True:
              
            rad_ind = rad_index[torch.searchsorted(torch.cumsum(radi_prob,dim=0), torch.cuda.FloatTensor(lam.shape[0]).uniform_(), right=True)]
            
            mu_s = mu_s_all[lam_ind,rad_ind]
            mu_a = mu_a_all[lam_ind,rad_ind]
            
            if j == 0:
                MFP_avg = ((1/(torch.mean(mu_s)+torch.mean(mu_a))) / 1000)
            
            ABS = mu_a/(mu_a + mu_s)
            SCA = mu_s/(mu_a + mu_s)
            
            ABS_SCA = torch.cumsum(torch.hstack((torch.unsqueeze(ABS,1), torch.unsqueeze(SCA,1))), dim=1)
            
            rand = torch.unsqueeze(torch.cuda.FloatTensor(ABS_SCA.shape[0]).uniform_(),1).view(-1,1)
            ABS_SCA = torch.searchsorted(ABS_SCA, rand).view(1,-1)[0]

            RN = torch.cuda.FloatTensor(lam.shape[0]).uniform_()
            MFP = MFP_avg * (-torch.log(RN)) 
            MFP = MFP.double()
            if j == 0:
                print(MFP_avg)
                
                d_angles = np.linspace(-1.5,1.5, 180)
                std = 0.5 + vol**0.5
                p = (np.pi*std) * np.exp(-0.5*((d_angles-0)/std)**2)
                p = p/np.sum(p)
                angle = np.random.choice(d_angles, p=p, size=MFP.shape)
                angle = torch.from_numpy(angle).to(pro)
                x1 = (torch.sin(angle) * MFP) + x0
                z1 = (torch.cos(angle) * MFP) + z0
                loss = torch.exp(-MFP/Dp)
                E1 = ABS_SCA * E0 * loss
                vat = Lines((vatx,vatz))

            else:
                angle = torch.cumsum(phase_all[lam_ind, rad_ind], dim=1).type(torch.float64)
                rand = torch.unsqueeze(torch.cuda.FloatTensor(angle.shape[0]).uniform_(),1)
                angle = torch.flatten(ang[torch.searchsorted(angle, rand)])
                z1 = ((torch.cos(angle) * MFP) + z0).type(torch.float64)
                z1[z1 < 0] = 0
                MFP[z1 < 0] = torch.abs(z0[z1 < 0] / torch.cos(angle[z1 < 0]))
                x1 = MFP * torch.sin(angle) + x0
                loss = torch.exp(-MFP/Dp)
                E1 = ABS_SCA * E0 * loss
                
            inds, ray_data = MCRT.Filter_Rays(x0, x1, z0, z1, E0, E1, Ecut, Dp, mesh_size, vatx, vatz)

            ray_data = np.array(ray_data.detach().cpu().numpy())

            if len(ray_data) == 0:
                array = np.array(vat.vat_out())
                T1 = time.perf_counter()
                sim_time = T1-T0
                print(f'Simulation Time = {T1-T0}')
                return array, phase_time, sim_time

            vat.plotLines(ray_data)
            
            del ray_data
            
            lam_ind = lam_ind[inds]
            
            lam = wave[lam_ind]
            Dp = Dps[lam_ind]
            x0 = x1[inds]
            z0 = z1[inds]
            E0 = E1[inds]
            j += 1
            


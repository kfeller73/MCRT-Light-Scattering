# This function computes all of the possible Mie form factors, assuming the real componet of the refractive index is constant for the wavelengths used
# This fuction creates a pandas dataframe of the data and writes it to a CSV file where:
# col_1 is the particle radius
# col_2 is the wavelength
# col_3 is the scattering coefficient
# col_4 is the absorbtion coefficient 
# col_5 to end is the normalized intensity of the form factor, corrected by the PY structure factor, at a given scattering angle, starting at -pi radians to pi radians

import numpy as np
import PyMieScatt as ps
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir(os.path.dirname(__file__))

class Mie_Calc():

    # Percus-Yevick hard sphere approximation structure factor
    # https://onlinelibrary-wiley-com.ezproxy.lib.vt.edu/doi/pdfdirect/10.1107/S1600576720014041
    def PY_SF(rad,vol,ang,lam, radi_prob):
        rad = np.average(rad, weights=radi_prob)
        D = 2*rad
        Q = 4*np.pi*np.sin(ang)/lam
        L1 = ((1+(2*vol))**2) / (1-vol)**4
        L2 = -((1+(vol/2))**2) / (1-vol)**4
        QD = Q*D
        sin, cos = np.sin(QD), np.cos(QD)
        t1 = L1*((sin-(QD*cos))/(QD**3))
        t2 = 6*vol*L2*((((QD**2)*cos) - (2*QD*sin) - (2*cos) + 2) / (QD**4))
        t3 = ((QD**4)*cos) - (4*(QD**3)*sin) - (12*(QD**2)*cos) + (24*QD*sin) + (24*cos) - 24
        t3 = vol*L1*0.5*(t3/(QD**6))
        NCQ = -24*vol*(t1-t2-t3)
        PYSF = 1/(1-NCQ)
        return PYSF

    def PY_Poly(radi, vol, ang, lam, radi_prob, debye):
        # citation link for the paper used to generate charged structure factors
        # https://onlinelibrary-wiley-com.ezproxy.lib.vt.edu/doi/pdfdirect/10.1107/S1600576720014041
        q = 4*np.pi*np.sin(ang/2)/lam
        radi_prob = radi_prob/np.sum(radi_prob)
        i = complex(0,1)
        u1, u2, u3, v3 = 0, 0, 0, 0
        voly_num, voly_den = 0, 0
        for j in range(len(radi)):
            a = radi[j]
            
            qa = q * a
            rp = radi_prob[j]
            voly_num += ((debye*a)**3) * rp
            voly_den += (((debye*a)+1)**3) * rp
            
            sin, cos = np.sin(qa), np.cos(qa)
        
            u1 += (1-(i*qa))* (sin-(qa*cos))*(np.exp(i*qa)) *rp
            u2 += (qa**2)*sin*np.exp(i*qa) * rp
            u3 += qa * (sin - (qa*cos)) * np.exp(i*qa) * rp
            v3 += (qa**3) * rp
        
        voly = voly_num/voly_den
        psi = (3 * vol) / (voly - vol)
        u3 = u3 * -i
        f11, f22, f12 = 1 + (psi*(u1/v3)), 1 + (psi*(u2/v3)), psi*u3/v3
        
        num = np.imag(np.conj(f22)*((f11*f22)+(f12**2)))
        den = np.imag(f11) * (np.abs((f11*f22)+(f12**2))**2)
        Sq = num/den
        return Sq
        
    def Mie_Phase(radi, radi_prob, wavelengths, par_n, par_k, med_n, vol, mat, debye, Dps):
        i = 0
        pi = np.pi
        for lam in wavelengths:
            ind = np.where(wavelengths == lam)
            RI_par = np.complex(par_n[ind], par_k[ind])
            RI_med = med_n[ind] 
            RI_med = (vol*complex(par_n, par_k)) + ((1-vol)*med_n)
            lam = lam
            for rad in radi:
                x = 2*pi*rad/lam
                Phase = ps.ScatteringFunction(RI_par,lam,2*rad,RI_med,minAngle=-180*0.999,maxAngle=180*0.999, normalization=None,angularResolution=1)
                ang = Phase[0]
                Phase = Phase[3]
                Qs = integrate.simps(Phase[:180] , np.cos(ang[:180]))/(x**2)

                Sq = Mie_Calc.PY_Poly(radi,vol,ang,lam, radi_prob, debye)
                Phase = Phase / (2*pi*Qs*(x**2))
                if rad == radi[2]:
                    print(f'rad = {rad}')
                    fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
                    p_theta = Phase/np.sum(Phase)
                    ax.plot(ang,p_theta, label=r'P($ \theta $)')
                    print('here')
                    pass
                Phase = Phase * Sq
               
                if rad == radi[2]:
                    s_theta = Phase/np.sum(Phase)
                    ax.plot(ang,s_theta, label=r'P($\theta$)$S_F$($\theta$)')
                    #ax.set_rticks([0.05, 0.1, 0.15, 0.2])
                    # plt.polar(ang,Sq)
                    plt.legend(bbox_to_anchor=(1.1, 1.15))
                mu_s = float((3*pi*vol*Qs/(2*rad)) * integrate.simps(Phase[:180] , np.cos(ang[:180])))
                mu_a = (4 * pi * np.imag(RI_par) / (lam)) #+ (1/(Dps[0]*1000))
                
                para = [lam, rad]
                Scat = Phase/np.sum(Phase)
                
                if i == 0:
                    headers = ['Wavelength (nm)', 'Radius (nm)', 'mu_s (1/nm)', 'mu_a (1/nm)']
                    headers.extend(ang.tolist())
                    scat_set = pd.DataFrame(columns=(headers))
                para.append(mu_s)
                para.append(mu_a)
                
                para.extend(Scat.tolist())
                scat_set.loc[i] = para
                i += 1
        os.chdir(r'C:\Users\Keyton Feller\Desktop\MCRT_ZnO\Phasers')
        scat_set.to_csv(f'{vol}_{debye}_Phasers.csv', index=False)
        return ang
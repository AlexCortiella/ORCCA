######################
#### FNPEG models ####
######################

## Imports
import time
import torch
import torch.nn as nn
import torch.optim as optim

import torchdyn
import numpy as np
import os

# FUNCTION DECLARATIONS

#FNPEG longitudinal equations of motion

def EOM_FNPEG_long_func(t, xSph, sigma0, e0, sigmaF, eF, lRef, tRef, planet, vehicle, guid, pred):

    """
    EOM_FNPEG_LONG Right-hand side of equations of motion for longitudinal
    atmospheric entry dynamics for FNPEG.

    This function computes the value of the right-hand side of the equations
    of motion for the atmospheric entry of a vehicle, expressed as a function
    of time. *Only longitudinal (along-track) dynamics* are computed. In
    particular, we only integrate equations for r, V, gamma, s
    (Eqs. (1,4,5,17) in Ref. [1]).
    The planetary rotation rate and the offset between the heading and
    azimuth of the target site are neglected (Omega = 0, \Delta{\psi} = 0).
    The independent variable is time.

    All quantities used in the computation of the right-hand side are
    dimensionless.

    AUTHOR:
    Alexandre Cortiella , CU Boulder, alexandre.cortiella@colorado.edu  (based on Davide Amato's matlab version)
    Unpack, non-dimensionalize, create auxiliary variables
    """

    #Unpack state vector
    r = xSph[0]#radial component
    theta = xSph[1]#longitude
    phi = xSph[2]#latitude
    v = xSph[3]#velocity magnitude
    gamma = xSph[4]#flight path angle
    psi = xSph[5]#heading angle

    #Auxialiry variables
    sgam = torch.sin(gamma)
    cgam = torch.cos(gamma);

    #Reference acceleration (m/s^2)
    g0_ms2 = lRef / tRef**2

    #Kinematic equations
    rdot = v * sgam
    sdot = -v * cgam / r

    ##########################
    #Atmospheric density
    ##########################
    rho = 1.0

    ##########################
    #Aerodynamic accelerations
    ##########################

    #Dimensionalize velocity
    v_ms = v * lRef / tRef

    #Dimensional drag and lift acceleration magnitudes (m/s^2)
    D_ms2 = 0.5 * rho * v_ms**2 / vehicle['B0']
    L_ms2 = vehicle.LD * D_ms2

    #Non-dimensionalize accelerations
    L = L_ms2 / g0_ms2
    D = D_ms2 / g0_ms2

    if guid['filter']['status']: #% check if fading-memory filter is turned on
        #scale lift and drag appropriately
        L = L * guid['filter']['rho_L']
        D = D * guid['filter']['rho_D']

    #Bank angle magnitude
    if guid['FNPEG']['bankProfile'] is "constant":
        sigma = sigma0

    elif guid['FNPEG']['bankProfile'] is "linear":
        e = 1 / r - 0.5 * v ** 2
        sigma = sigma0 + (sigmaF - sigma0) * (e - e0) / (eF - e0)

    csig = torch.cos(sigma)

    ################################
    #Dynamic equations (V, gamma)
    ################################

    #V/dt
    vdot = - D - sgam / r ** 2

    #d(gamma)/dt
    gammadot = ( L * csig + ( v ** 2 - 1 / r ) * ( cgam / r ) ) / v

    return torch.cat([rdot, 0, 0, vdot, gammadot, 0, sdot], -1)


#FNPEG longitudinal equations of motion

def EOM_FNPEG_func(t, xSph, sigma0, e0, sigmaF, eF, lRef, tRef, planet, vehicle, guid, pred):

    """
    EOM_FNPEG Right-hand side of equations of motion for longitudinal
    atmospheric entry dynamics for FNPEG.

    This function computes the value of the right-hand side of the equations
    of motion for the atmospheric entry of a vehicle, expressed as a function
    of time. *Only longitudinal (along-track) dynamics* are computed. In
    particular, we only integrate equations for r, V, gamma, s
    (Eqs. (1,4,5,17) in Ref. [1]).
    The planetary rotation rate and the offset between the heading and
    azimuth of the target site are neglected (Omega = 0, \Delta{\psi} = 0).
    The independent variable is time.

    All quantities used in the computation of the right-hand side are
    dimensionless.

    AUTHOR:
    Alexandre Cortiella , CU Boulder, alexandre.cortiella@colorado.edu  (based on Davide Amato's matlab version)
    Unpack, non-dimensionalize, create auxiliary variables
    """

    #Unpack state vector
    r = xSph[0]#radial component
    theta = xSph[1]#longitude
    phi = xSph[2]#latitude
    v = xSph[3]#velocity magnitude
    gamma = xSph[4]#flight path angle
    psi = xSph[5]#heading angle

    #Auxialiry variables
    sgam = torch.sin(gamma)
    cgam = torch.cos(gamma)
    tgam = torch.tan(gamma)

    sphi = torch.sin(phi)
    cphi = torch.cos(phi)
    tphi = torch.tan(phi)

    spsi = torch.sin(psi)
    cpsi = torch.cos(psi)

    #Reference acceleration (m/s^2)
    g0_ms2 = lRef / tRef**2

    #Kinematic equations
    rdot = v * sgam
    thetadot = (v * cgam * spsi) / (r * cphi)
    phidot = (v * cgam * cpsi) / r
    sdot = -v * cgam / r

    ##########################
    #Atmospheric density
    ##########################
    rho = 1.0

    ##########################
    #Aerodynamic accelerations
    ##########################

    #Dimensionalize velocity
    v_ms = v * lRef / tRef

    #Dimensional drag and lift acceleration magnitudes (m/s^2)
    D_ms2 = 0.5 * rho * v_ms**2 / vehicle['B0']
    L_ms2 = vehicle.LD * D_ms2

    #Non-dimensionalize accelerations
    L = L_ms2 / g0_ms2
    D = D_ms2 / g0_ms2

    if guid['filter']['status']: #% check if fading-memory filter is turned on
        #scale lift and drag appropriately
        L = L * guid['filter']['rho_L']
        D = D * guid['filter']['rho_D']

    #Bank angle magnitude
    if guid['FNPEG']['bankProfile'] is "constant":
        sigma = sigma0

    elif guid['FNPEG']['bankProfile'] is "linear":
        e = 1 / r - 0.5 * v ** 2
        sigma = sigma0 + (sigmaF - sigma0) * (e - e0) / (eF - e0)

    ssig = torch.sin(sigma)
    csig = torch.cos(sigma)

    ################################
    #Dynamic equations (V, gamma, psi)
    ################################

    #V/dt
    vdot = - D - sgam / r ** 2 + Omega ** 2 * r * cphi * (sgam * cphi - cgam * sphi * cpsi)

    #d(gamma)/dt
    gammadot = ( L * csig + ( v ** 2 - 1 / r ) * ( cgam / r ) + 2 * Omega * v * cphi * spsi + Omega ** 2 * r * cphi * (cgam * cphi + sgam * cpsi * sphi)) / v

    #d(psi)/dt
    psidot = (L * ssig / cgam + v ** 2 / r * cgam * spsi * tphi - 2 * Omega * v * (tgam * cpsi * cphi - sphi) + Omega ** 2 * r / cgam * spsi * sphi  cphi) / v

    return torch.cat([rdot, thetadot, phidot, vdot, gammadot, psidot, sdot], -1)

    class EOM_FNPEG_long(nn.Module):
    
    """
    EOM_FNPEG_LONG Right-hand side of equations of motion for longitudinal
    atmospheric entry dynamics for FNPEG.

    This function computes the value of the right-hand side of the equations
    of motion for the atmospheric entry of a vehicle, expressed as a function
    of time. *Only longitudinal (along-track) dynamics* are computed. In 
    particular, we only integrate equations for r, V, gamma, s 
    (Eqs. (1,4,5,17) in Ref. [1]).
    The planetary rotation rate and the offset between the heading and
    azimuth of the target site are neglected (Omega = 0, \Delta{\psi} = 0).
    The independent variable is time.

    All quantities used in the computation of the right-hand side are
    dimensionless.

    AUTHOR:
    Alexandre Cortiella , CU Boulder, alexandre.cortiella@colorado.edu  (based on Davide Amato's matlab version)
    Unpack, non-dimensionalize, create auxiliary variables
    """
    
    def __init__(self, sigma0, e0, sigmaF, eF, lRef, tRef, planet, vehicle, guid):
        super().__init__()
        
        #Reference acceleration (m/s^2)
        self.g0_ms2 = lRef / tRef**2    
        #Dimensionalize velocity
        self.v_ms = v * lRef / tRef
        #Vehicle parameters
        self.vehicle = vehicle
        #Planet parameters
        self.planet = planet
        #Guidance parameters
        self.guid = guid
        self.e0 = e0
        self.sigmaF = sigmaF
        self.eF = eF
        
        #Optimization variables
        self.sigma0 = nn.Parameter(torch.tensor(sigma0))
        
        
    def forward(self, t, state):
    
        #Unpack state vector
        r, theta, phi, v, gamma, psi, s = state
    #     r = xSph[0]#radial component
    #     theta = xSph[1]#longitude
    #     phi = xSph[2]#latitude
    #     v = xSph[3]#velocity magnitude
    #     gamma = xSph[4]#flight path angle
    #     psi = xSph[5]#heading angle

        #Auxialiry variables
        sgam = torch.sin(gamma)
        cgam = torch.cos(gamma);

        #Kinematic equations
        rdot = v * sgam
        thetadot = torch.tensor(0.0)
        phidot = torch.tensor(0.0)
        sdot = -v * cgam / r

        ##########################
        #Atmospheric density 
        ##########################
        rho = 1.0

        ##########################
        #Aerodynamic accelerations
        ##########################

        #Dimensional drag and lift acceleration magnitudes (m/s^2)
        D_ms2 = 0.5 * rho * self.v_ms**2 / self.vehicle['B0']
        L_ms2 = self.vehicle['LD'] * D_ms2

        #Non-dimensionalize accelerations
        L = L_ms2 / self.g0_ms2
        D = D_ms2 / self.g0_ms2

        if guid.filter.status: #% check if fading-memory filter is turned on
            #scale lift and drag appropriately 
            L = L * self.guid['filter']['rho_L']
            D = D * self.guid['filter']['rho_D']

        #Bank angle magnitude
        if self.guid['FNPEG']['bankProfile'] is "constant":
            sigma = self.sigma0

        elif guid['FNPEG']['bankProfile'] is "linear":
            e = 1 / r - 0.5 * v ** 2
            sigma = self.sigma0 + (self.sigmaF - self.sigma0) * (e - self.e0) / (self.eF - self.e0)

        csig = torch.cos(sigma)

        ################################
        #Dynamic equations (V, gamma)
        ################################

        #V/dt
        vdot = - D - sgam / r ** 2

        #d(gamma)/dt
        gammadot = ( L * csig + ( v ** 2 - 1 / r ) * ( cgam / r ) ) / v
    
        #d(psi)/dt
        psidot = torch.tensor(0.0)
        
        return torch.cat([rdot, thetadot, phidot, vdot, gammadot, psidot, sdot], -1)


class EOM_FNPEG(nn.Module):
    
    """
    EOM_FNPEG Right-hand side of equations of motion for longitudinal
    atmospheric entry dynamics for FNPEG.

    This function computes the value of the right-hand side of the equations
    of motion for the atmospheric entry of a vehicle, expressed as a function
    of time. *Only longitudinal (along-track) dynamics* are computed. In 
    particular, we only integrate equations for r, V, gamma, s 
    (Eqs. (1,4,5,17) in Ref. [1]).
    The planetary rotation rate and the offset between the heading and
    azimuth of the target site are neglected (Omega = 0, \Delta{\psi} = 0).
    The independent variable is time.

    All quantities used in the computation of the right-hand side are
    dimensionless.

    AUTHOR:
    Alexandre Cortiella , CU Boulder, alexandre.cortiella@colorado.edu  (based on Davide Amato's matlab version)
    Unpack, non-dimensionalize, create auxiliary variables
    """
    def __init__(self, sigma0, e0, sigmaF, eF, lRef, tRef, planet, vehicle, guid):
        super().__init__()

        #Reference acceleration (m/s^2)
        self.g0_ms2 = lRef / tRef**2    
        #Dimensionalize velocity
        self.v_ms = v * lRef / tRef
        #Vehicle parameters
        self.vehicle = vehicle
        #Planet parameters
        self.planet = planet
        #Guidance parameters
        self.guid = guid
        self.e0 = e0
        self.sigmaF = sigmaF
        self.eF = eF

        #Optimization variables
        self.sigma0 = nn.Parameter(torch.tensor(sigma0))
        
        
    def forward(self, t, state):   
        
        #Unpack state vector
        r, theta, phi, v, gamma, psi, s = state

        #Auxialiry variables
        sgam = torch.sin(gamma)
        cgam = torch.cos(gamma)
        tgam = torch.tan(gamma)

        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        tphi = torch.tan(phi)

        spsi = torch.sin(psi)
        cpsi = torch.cos(psi)

        #Kinematic equations
        rdot = v * sgam
        thetadot = (v * cgam * spsi) / (r * cphi)
        phidot = (v * cgam * cpsi) / r
        sdot = -v * cgam / r

        ##########################
        #Atmospheric density 
        ##########################
        rho = 1.0

        ##########################
        #Aerodynamic accelerations
        ##########################

        #Dimensional drag and lift acceleration magnitudes (m/s^2)
        D_ms2 = 0.5 * rho * self.v_ms**2 / self.vehicle['B0']
        L_ms2 = self.vehicle['LD'] * D_ms2

        #Non-dimensionalize accelerations
        L = L_ms2 / self.g0_ms2
        D = D_ms2 / self.g0_ms2

        if guid.filter.status: #% check if fading-memory filter is turned on
            #scale lift and drag appropriately 
            L = L * self.guid['filter']['rho_L']
            D = D * self.guid['filter']['rho_D']

        #Bank angle magnitude
        if self.guid['FNPEG']['bankProfile'] is "constant":
            sigma = self.sigma0

        elif guid['FNPEG']['bankProfile'] is "linear":
            e = 1 / r - 0.5 * v ** 2
            sigma = self.sigma0 + (self.sigmaF - self.sigma0) * (e - self.e0) / (self.eF - self.e0)

        ssig = torch.sin(sigma)
        csig = torch.cos(sigma)

        ################################
        #Dynamic equations (V, gamma, psi)
        ################################

        Omega = self.planet['Omega']
        
        #V/dt
        vdot = - D - sgam / r ** 2 + Omega ** 2 * r * cphi * (sgam * cphi - cgam * sphi * cpsi)

        #d(gamma)/dt
        gammadot = ( L * csig + ( v ** 2 - 1 / r ) * ( cgam / r ) + 2 * Omega * v * cphi * spsi + Omega ** 2 * r * cphi * (cgam * cphi + sgam * cpsi * sphi)) / v

        #d(psi)/dt
        psidot = (L * ssig / cgam + v ** 2 / r * cgam * spsi * tphi - 2 * Omega * v * (tgam * cpsi * cphi - sphi) + Omega ** 2 * r / cgam * spsi * sphi  cphi) / v

        return torch.cat([rdot, thetadot, phidot, vdot, gammadot, psidot, sdot], -1)

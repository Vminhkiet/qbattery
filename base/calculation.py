import numpy as np
import base.circuit as cr
import base.hamiltonian as hm
import base.constant as ct
import base.Gradient as gr
from scipy.linalg import expm
import tqix as tq

def ps0():
    """
       Return ps0 = dw tensor dw tensor ... dw   (have N dw)

    """
    ps_0 = ct.dw
    for i in range(ct.N):
        ps_0 = np.kron(ps_0 , ct.dw)
    return ps_0

def psi(t,thetas):
    """
        Return psit = e^-i*t*H1 * ps0()
    """
    psi_t = expm(-1j * t * hm.h1(thetas))
    return psi_t @ ps0()

def E(t,thetas):
    if(t == 0):
        return np.real(( tq.daggx(ps0()) @ hm.h0() @ ps0() )[0,0])
    
    return np.real(( tq.daggx(psi(t,thetas)) @ hm.h0() @ psi(t,thetas) )[0,0])

def W(t,thetas):
    return E(t,thetas) - E(0,thetas)

def P(t,thetas):
    if(t == 0):
        return 0
    
    return W(t,thetas) / t

def caculator_manytime(t1,thetas,shots):
    time=np.linspace(0.0001,t1,shots)
    arrayW=[]
    arrayP=[]

    for i in time:
        w=W(i,thetas)
        p=P(i,thetas)
        arrayW.append(w)
        arrayP.append(p)

    return time,arrayW,arrayP
        

import numpy as np
from numpy import pi
from scipy.special import gammainc
from scipy import signal

def calc_spec(phi, d1, d2):
    # wavenumber one (equals to dk1 and dk2)

    ndim = phi.ndim
    n2, n1 = phi.shape
    L1 = d1*n1
    L2 = d2*n2
    dk1 = 2*np.pi/L1
    dk2 = 2*np.pi/L2

    k2 = dk2*np.append(np.arange(0.,n2/2), np.arange(-n2/2,0.))
    k1 = dk1*np.arange(0.,n1/2+1)
    kk1, kk2 = np.meshgrid(k1, k2)

    kk1 = np.fft.fftshift(kk1,axes=0)
    kk2 = np.fft.fftshift(kk2,axes=0)
    kappa2 = kk1**2 + kk2**2
    kappa = np.sqrt(kappa2)

    phih = np.fft.rfft2(phi,axes=(0,1))
    spec = 2.*(phih*phih.conj()).real/ (dk1*dk2)\
                / (n1*n2)**2
    #spec =  np.fft.fftshift(spec,axes=0)

    var_dens = np.fft.fftshift(spec.copy(),axes=0)
    var_dens[:,0], var_dens[:,-1] = var_dens[:,0]/2., var_dens[:,-1]/2.
    #var = var_dens.sum()*dk1*dk2

    return spec, var_dens, k1, k2, dk1, dk2


def calc_ispec(k, l, E):
    """ Calculates the azimuthally-averaged spectrum

        Parameters
        ===========
        - E is the two-dimensional spectrum
        - k is the wavenumber is the x-direction
        - l is the wavenumber in the y-direction

        Output
        ==========
        - kr: the radial wavenumber
        - Er: the azimuthally-averaged spectrum """

    dk = np.abs(k[2]-k[1])
    dl = np.abs(l[2]-l[1])

    k, l = np.meshgrid(k,l)

    wv = np.sqrt(k**2+l**2)

    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()

    dkr = np.sqrt(dk**2 + dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr/2.,dkr)
    Er = np.zeros((kr.size))

    for i in range(kr.size):
        fkr =  (wv>=kr[i]-dkr/2) & (wv<=kr[i]+dkr/2)
        dth = np.pi / (fkr.sum()-1)
        Er[i] = (E[fkr]*(wv[fkr])*dth).sum(axis=(0))

    return kr, Er.squeeze()

def spectral_slope(k,E,kmin,kmax,stdE):
    ''' compute spectral slope in log space in
        a wavenumber subrange [kmin,kmax],
        m: spectral slope; mm: uncertainty'''

    fr = np.where((k>=kmin)&(k<=kmax))

    ki = np.matrix((np.log10(k[fr]))).T
    Ei = np.matrix(np.log10(np.real(E[fr]))).T
    dd = np.matrix(np.eye(ki.size)*((np.abs(np.log10(stdE)))**2))

    G = np.matrix(np.append(np.ones((ki.size,1)),ki,axis=1))
    Gg = ((G.T*G).I)*G.T
    m = Gg*Ei
    mm = np.sqrt(np.array(Gg*dd*Gg.T)[1,1])
    yfit = np.array(G*m)
    m = np.array(m)[1]

    return m, mm

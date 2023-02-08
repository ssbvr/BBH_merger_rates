"""
These functions comes from POSYDON v1.0.0, see
https://github.com/POSYDON-code/POSYDON
"""

import numpy as np

def orbital_period_from_separation(separation, m1, m2):
    """Apply the Third Kepler law.

    Parameters
    ----------
    separation : float
        Orbital separation (semi-major axis) in Rsun.
    m1 : float
        Mass of one of the stars in solar units.
    m2 : type
        Mass of the other star in solar units.

    Returns
    -------
    float
        The orbital period in days.

    """
    dayyer = 365.25 # days per year
    aursun = 214.95 # Rsuns per AU 
    return dayyer * ((separation / aursun)**3.0 / (m1 + m2)) ** 0.5


def roche_lobe_radius(q, a_orb=1):
    """Approximate the Roche lobe radius from Eggleton (1983).

    Parameters
    ----------
    q : float
        Dimensionless mass ratio = MRL/Mcomp, where
        MRL is the mass of the star we calculate the RL and
        Mcomp is the mass of its companion star.
    a_orb : float
        Orbital separation. The return value will have the same unit.

    Returns
    -------
    float
        Roche lobe radius in similar units as a_orb
    References
    ----------
    .. [1] Eggleton, P. P. 1983, ApJ, 268, 368

    """
    RL = a_orb * (0.49 * q**(2. / 3.)) / (
        0.6 * q**(2. / 3.) + np.log(1 + q**(1. / 3))
    )
    return RL


def rzams(m, z=0.02, Zsun=0.02):
    """Evaluate the zero age main sequence radius.

    Parameters
    ----------
    m : array_like
        The masses of the stars in Msun.
    z : float
        The metallicity of the star.

    Returns
    -------
    ndarray
        Array of same size as `m` containing the ZAMS radii of the stars (Rsun)

    References
    ----------
    .. [1] Tout C. A., Pols O. R., Eggleton P. P., Han Z., 1996,
           MNRAS, 281, 257


    """
    m = np.asanyarray(m)
    xz = [
        0.0, 3.970417e-01, -3.2913574e-01, 3.4776688e-01, 3.7470851e-01,
        9.011915e-02, 8.527626e+00, -2.441225973e+01, 5.643597107e+01,
        3.706152575e+01, 5.4562406e+00, 2.5546e-04, -1.23461e-03, -2.3246e-04,
        4.5519e-04, 1.6176e-04, 5.432889e+00, -8.62157806e+00, 1.344202049e+01,
        1.451584135e+01, 3.39793084e+00, 5.563579e+00, -1.032345224e+01,
        1.944322980e+01, 1.897361347e+01, 4.16903097e+00, 7.8866060e-01,
        -2.90870942e+00, 6.54713531e+00, 4.05606657e+00, 5.3287322e-01,
        5.86685e-03, -1.704237e-02, 3.872348e-02, 2.570041e-02, 3.83376e-03,
        1.715359e+00, 6.2246212e-01, -9.2557761e-01, -1.16996966e+00,
        -3.0631491e-01, 6.597788e+00, -4.2450044e-01, -1.213339427e+01,
        -1.073509484e+01, -2.51487077e+00, 1.008855000e+01, -7.11727086e+00,
        -3.167119479e+01, -2.424848322e+01, -5.33608972e+00, 1.012495e+00,
        3.2699690e-01, -9.23418e-03, -3.876858e-02, -4.12750e-03, 7.490166e-02,
        2.410413e-02, 7.233664e-02, 3.040467e-02, 1.97741e-03, 1.077422e-02,
        3.082234e+00, 9.447205e-01, -2.15200882e+00, -2.49219496e+00,
        -6.3848738e-01, 1.784778e+01, -7.4534569e+00, -4.896066856e+01,
        -4.005386135e+01, -9.09331816e+00, 2.2582e-04, -1.86899e-03,
        3.88783e-03, 1.42402e-03, -7.671e-05
    ]
    lzs = np.log10(z / Zsun)

    msp = np.zeros(17)
    msp[0] = 0.0
    msp[1] = xz[1] + lzs * (xz[2] + lzs * (xz[3] + lzs
                                           * (xz[4] + lzs * xz[5])))
    msp[2] = xz[6] + lzs * (xz[7] + lzs * (xz[8] + lzs
                                           * (xz[9] + lzs * xz[10])))
    msp[3] = xz[11] + lzs * (xz[12] + lzs * (xz[13] + lzs
                                             * (xz[14] + lzs * xz[15])))
    msp[4] = xz[16] + lzs * (xz[17] + lzs * (xz[18] + lzs
                                             * (xz[19] + lzs * xz[20])))
    msp[5] = xz[21] + lzs * (xz[22] + lzs * (xz[23] + lzs
                                             * (xz[24] + lzs * xz[25])))
    msp[6] = xz[26] + lzs * (xz[27] + lzs * (xz[28] + lzs
                                             * (xz[29] + lzs * xz[30])))
    msp[7] = xz[31] + lzs * (xz[32] + lzs * (xz[33] + lzs
                                             * (xz[34] + lzs * xz[35])))
    msp[8] = xz[36] + lzs * (xz[37] + lzs * (xz[38] + lzs
                                             * (xz[39] + lzs * xz[40])))
    msp[9] = xz[41] + lzs * (xz[42] + lzs * (xz[43] + lzs
                                             * (xz[44] + lzs * xz[45])))
    msp[10] = xz[46] + lzs * (xz[47] + lzs * (xz[48] + lzs
                                              * (xz[49] + lzs * xz[50])))
    msp[11] = xz[51] + lzs * (xz[52] + lzs * (xz[53] + lzs
                                              * (xz[54] + lzs * xz[55])))
    msp[12] = xz[56] + lzs * (xz[57] + lzs * (xz[58] + lzs
                                              * (xz[59] + lzs * xz[60])))
    msp[13] = xz[61]
    msp[14] = xz[62] + lzs * (xz[63] + lzs * (xz[64] + lzs
                                              * (xz[65] + lzs * xz[66])))
    msp[15] = xz[67] + lzs * (xz[68] + lzs * (xz[69] + lzs
                                              * (xz[70] + lzs * xz[71])))
    msp[16] = xz[72] + lzs * (xz[73] + lzs * (xz[74] + lzs
                                              * (xz[75] + lzs * xz[76])))
    mx = np.sqrt(m)
    r = ((msp[8] * m**2 + msp[9] * m**6) * mx + msp[10] * m**11
         + (msp[11] + msp[12] * mx) * m**19) / (
             msp[13] + msp[14] * m**2
             + (msp[15] * m**8 + m**18 + msp[16] * m**19) * mx)

    return r
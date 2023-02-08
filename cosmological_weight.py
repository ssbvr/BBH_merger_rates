"""
This code allows reweighting a synthetic population of merging
binary black holes (BBHs) resulting from a population synthesis model.
Given some model assumption, the code can compute the observable 
merging BBH population by a gravitational-wave detector as well as 
the intrinsic BBH population (equivalent to a gravitational-wave
detector with infinite sensitivity). Additionally, this code computes
the BBH detection rate and the BBH merger rate density. 
If long-duration gamma-ray bursts (LGRBs) are modelled, the code 
computes distributions and rates of LGRBs (they have zero delay times
compared to the BBH merger event). The code can also compute rates
provided a synthetic BBH population from a rectilinear MESA gird
(instead of a Monte Carlo sampling of the initial parameter space).
Finally, the code can also compute the stochastic gravitational-wave
background of a given intrinsic merging BBH population.
"""

__author__ = ['Simone Bavera <Simone.Bavera@unige.ch>']
__credits__ = ['Tassos Fragos <Anastasios.Fragkos@unige.ch>',
               'Emmanouil Zapartas <Emmanouil.Zapartas@unige.ch>']

import os
import csv
import numpy as np
import scipy as sp
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.interpolate import interp1d
from astropy import units as u
from astropy import constants as const
from astropy.cosmology import Planck15, z_at_value
from utils import orbital_period_from_separation, roche_lobe_radius, rzams
import selection_effects

class CosmologicalWeight(object):

    def __init__(self, 
                 path, 
                 MESA_grid=False, 
                 reject_ZAMS_RLOF=False,
                 SFR='Madau+Fragos17',
                 sigma_SFR='Bavera+20',
                 Z_max=0.03,
                 select_one_met=None,
                 dlogZ = 0.085,
                 Zsun='Grevesse+Sauval98', 
                 IMF='Krupa+2001',
                 f_corr=True, 
                 fbin='Sana+12', 
                 beaming_method='Goldstein+15', 
                 eta=0.01,
                 LGRB_properties=False,
                 columns_header=None, 
                 columns=None, 
                 class_variables=None,
                 verbose=False,
                 ):
        """Assign a cosmological weghit to a binary assuming Planck15 LCDM model parameters.

        Parameters
        ----------
        path : string
            Path to the txt file or Pandas dataframe containing the synthetic binary population.
        MESA_grid : bool
            `True` if the synthetic population originates from a rectilinear MESA grid.evolved for each metallicty bin.
        reject_ZAMS_RLOF : bool
            `True` to exclude systems roche-lobe-overflowing at ZAMS.
        SFR : string
            Star-formation-rate history you want to use:
            'Madau+Fragos17', 'Neijssel+19', 'Mason+15', 'Illustris'
        sigma_SFR : string
            Standard deviation of the trucated lognormal metallicity distribution
            assumed for the star-formation rate.
        Z_max : float
            The absolute metallicity cut off of the star formation rate. This should be
            used when assuming a truncated log-normal distribution of metallcity or
            to truncate the high Z Illustris star formation rate.
        select_one_met : bool
            `True` if your synthetic binary population only have one metallicity.
        dlogZ : float or range 
            If float value than assume this is the dlogZ range around the provided metallcity.
            if range [Z_min, Z_max], consider the star formation rate inside this absolute
            metallicity range.
        Zsun : string
            Solar metallcity referece used to generate the synthetic binary population:
            'Grevesse+Sauval98': Z_sun=0.017
            'Asplund+09': 0.0142
        IMF : string
            Inital mass function used to generate the synthetic binary population:
            'Krupa+2001', 'Krupa+202'
        f_corr : bool
            True if you need to renormalize the totalMassEvolvedForZ to account
            for the undelying stellar population not simulated.
        fbin : string
            Binary mass fraction used to generate the synthetic binary population:
            'Sana+12' is 0.7
        beaming_method : string
            Opening angle distribution which determines the beaming factor of LGRBs:
            'Goldstein+15', 'Lloyd-Ronning+19', 'median-f_b=0.01', 'median-f_b=0.02',
            'median-f_b=0.04', 'median-f_b=0.05'.
        eta : float
            LGRB energy efficinty.
        columns_header : string
            Default column header to match dataset columns name with class varaibles.
        columns : list strings
            List of columns names in the header of the file read by the class to be mapped
            to class variables.
        class_variables : array strings
            These are the availabele class variables which must be matched to the one of the file read by the class:
                'channel': integer corresponding to a specific evolutionari channnel
                'optimistic': if True the star udergoes common envelope in the HR gap
                'totalMassEvolvedForZ': total mass of systems simulated for a specific metallicity. If this quantity needs
                                        to be renormalized to account for the non-simulated underlying stellar polulation 
                                        set f_corr == True.
                'metallicity': absolute metallicity of the binary
                'q_i': initial binary mass ratio
                'm_star1_i': primary mass in Msun corresponding to BH1 at ZAMS
                'm_star2_i': secodary mass in Msun corresponding to BH2 at ZAMS
                'm_star1_postCE': primary mass in Msun post common envelope event
                'm_star2_postCE': secondary mass in Msun post common envelope event
                'm_star1_preSN':  primary mass in Msun before the supernova event
                'm_star2_preSN':  secondary mass in Msun before the supernova event
                'm_BH1': primary BH mass in Msun
                'm_BH2': secondary BH mass in Msun
                'spin_BH1': primary BH dimensionless spin parameter
                'tilt_BH1': angle between the orbital angular momentum and the spin axis of BH2
                'spin_BH2': secondary BH dimensionless spin parameter 
                'tilt_BH2': angle between the orbital angular momentum and the spin axis of BH2
                'p_i': orbital period of the binary system at ZAMS in days
                'e_i': eccentricity of the binary system at ZAMS
                'p_preCE': orbital period of the binary system post common envelope event in days
                'e_postCE': eccentricity of the binary system pre common envelope event
                'p_postCE': orbital period of the binary system post common envelope event in days
                'e_postCE': eccentricity of the binary system post common envelope event
                'p_preSN': orbital period of the binary system before the supernova event in days
                'e_preSN': eccentricity of the binary system pre supernova
                'p_f': orbital period of the BBH system after the supernova event in days
                'e_f': eccentricity of the BBH binary system post supernova
                'Dt_binary': binary life time in Myr
                'Dt_inspiral': gravitaional-wave inspiral timescale in Myr
                # TODO: extend the code to account LGRB emission from BH1 formation
                'flag_GRB': if True the secondary will emit a LGRB during the formation of BHs
                'm_disk_rad': radiated disk mass in Msun during the collapse of the secondary
                'L_GRB': luminosity of the LGRB in ergs emiteted by the secondary
                'E_GRB': total energy released in the collapse of the secondary in ergs
                'E_GRB_iso': total isotropic equivalent energy (E_GRB/f_beaming) released in the collapse of the secondary in ergs
                'f_beaming': beaming fraction, steradiant area of GRB cones
                'eta' : energy conversion rate from radiated gravitaional mass to LGRB energy
                'max_he_mass_ejected' : maximum He-mass ejected during the collapse of the secondary
        verbose : bool
            `True` if you want the print statements.
        """

        # store the class options
        self.MESA_grid = MESA_grid
        self.SFR = SFR
        self.sigma_SFR = sigma_SFR
        self.Zsun = Zsun
        self.IMF = IMF
        self.fbin = fbin
        self.beaming_method = beaming_method
        self.eta = eta
        self.LGRB_properties = LGRB_properties
        self.f_corr = f_corr
        self.Z_max = Z_max
        self.reject_ZAMS_RLOF = reject_ZAMS_RLOF
        self.select_one_met = select_one_met
        self.dlogZ = dlogZ
        self.verbose = verbose

        if self.SFR == 'Illustris':
            if self.verbose:
                print('Loading Illustris data...')
            self.illustris_data = np.load('./SFR/illustris_SFR.npz')

        # function to read the txt file passed to the class
        def ReadColumns(filename, col, nhead=1):
            header_entries = open(filename, "r").readlines()[0]
            header_entries = header_entries.split()
            #print("header_entries ", header_entries)

            # Read the history.data file
            data = np.loadtxt(filename, skiprows=nhead)

            # Create the list which will be filled with data
            data_cols = list()

            for c in range(0,len(col)):
                if col[c] in header_entries:
                    ind = header_entries.index(col[c])
                    data_cols.append(data[:,ind])
                else:
                    raise ValueError('Option asked col = %s not found in header_entries of final_data.txt'%col[c])
            return data_cols

        # general list of class variables
        class_var_options = ['channel','optimistic','totalMassEvolvedForZ','metallicity','q_i','m_star1_i',
                             'm_star2_i','m_star1_postCE','m_star2_postCE','m_star1_preSN','m_star2_preSN','m_BH1',
                             'm_BH2','spin_BH1','tilt_BH1','spin_BH2','tilt_BH2','p_i','e_i','p_preCE','e_postCE',
                             'p_postCE','e_postCE','p_preSN','e_preSN','p_f','e_f','Dt_binary','Dt_inspiral',
                             'flag_GRB','m_disk_rad','L_GRB','E_GRB','E_GRB_iso','f_beaming','eta','max_he_mass_ejected',
                             'Mchirp','q','chi_eff']

        # for conveniece store columns header of the file to be matched with the class variables, e.g.
        if columns_header == "arXiv_2106.15841_CE":
            # list of column names in the data file
            col = ['EvolvedMassForZ', 'channel', 'optimistic', 'met', 'm_BH1',
                   'm_star2_postCE', 'm_BH2', 'spin_BH1', 'spin_BH2', 'm_disk_rad',
                   'max_he_mass_ejected', 'p_postCE', 'p_f', 'e_f', 'tilt_postSN',
                   't_f', 't_inspiral']
            # class_var is a list of class varriables correspandaning to the dataset columns
            class_var = ['totalMassEvolvedForZ', 'channel', 'optimistic', 'metallicity', 'm_BH1', 'm_star2_postCE', 'm_BH2',
                 'spin_BH1', 'spin_BH2', 'm_disk_rad', 'max_he_mass_ejected', 'p_postCE', 'p_f', 'e_f', 
                   'tilt_BH2', 'Dt_binary', 'Dt_inspiral']
        elif columns_header is not None:
            raise ValueError(f'columns_header = {columns_header} not lnown!')
            
        if path is None:
            print('No detaset profived!')
            return
        
        # check if col an class_var were passed to the class
        if columns_header is None and columns is not None and class_variables is not None:
            col = columns
            class_var = class_variables
        elif columns_header is None and (columns is None or class_variables is None):
            raise ValueError('If columns_header is None then you need to pass columns and class_variables!')

        # check that class_var are in class_var_options and the size of col matches class_var size
        for c in range(0,len(class_var)):
            if class_var[c] not in class_var_options:
                raise ValueError('Option asked class_var = %s is not in class_var_options'%class_var[c])
        if len(col) != len(class_var):
            raise ValueError('Number of column to be read from final_data.txt is not equal to the number of class variables that are asked to correspond to.')

        # read the final_data.txt
        if isinstance(path, str):
            datacolumns = ReadColumns(path, col)
        elif isinstance(path, pd.DataFrame):
            datacolumns = path.to_numpy().T
        else:
            raise ValueError('path is not a string nor a data frame!')

        #store everithing in a dictionary
        self.mydict = {}
        for c in range(0,len(col)):
            self.mydict[class_var[c]] =  datacolumns[col.index(col[c])]

        # do other thinghts to the data
        if self.reject_ZAMS_RLOF:
            m_1 = self.mydict['m_star1_i']
            q = self.mydict['q_i']
            m_2 = m_1*q
            p_i = self.mydict['p_i']
            Z = self.mydict['metallicity']
            ecc = np.zeros(len(m_1))
            ZAMS_RLOF = np.zeros(len(m_1),dtype=bool)

            for i in range(len(m_1)):
                sep_min = rzams(m_1[i], Z[i], Zsun=self.getZsun()) / (roche_lobe_radius(q[i]) * (1. - ecc[i]))
                p_min = orbital_period_from_separation(sep_min, m_1[i], m_2[i])
                if p_min > p_i[i]:
                    ZAMS_RLOF[i] = True

            print('Binaries ZAMS RLOF in the dataset', sum(ZAMS_RLOF.astype(int)),'/',len(ZAMS_RLOF))
            # keep only sytems not RLOF at ZAMS
            for key in self.mydict.keys():
                self.mydict[key] = self.mydict[key][np.invert(ZAMS_RLOF)]
            
        if self.LGRB_properties:
            self.mydict["E_GRB"] = np.zeros(len(self.mydict["m_disk_rad"]))
            # for interpolation reasons the zero m_disk_rad = 1e-99
            disk = self.mydict["m_disk_rad"] > 1e-90
            self.mydict["E_GRB"][disk] = self.eta*self.mydict["m_disk_rad"][disk]*1.9892e33*2.99792458e10**2 # ergs
            self.mydict["f_beaming"] = self.BeamingFactor(len(self.mydict["E_GRB"]),self.beaming_method)
            self.mydict["E_GRB_iso"] = self.mydict["E_GRB"]/self.mydict["f_beaming"]
            self.mydict["flag_GRB"] = self.mydict["E_GRB_iso"] > 1e51
            self.mydict["L_GRB_iso"] = self.mydict["E_GRB_iso"]/2.5 # assume average LGRB t_B=25s

    ######################################################
    ###   BBH detection rate and merger rate density   ###
    ###    see arXiv:1906.12257, arXiv:2010.16333,     ###
    ###        arXiv:2106.15841, arXiv:221210924,      ###
    ###        arXiv:2204.02619, arXiv:2011.10057      ###
    ###              arXiv:2012.02274                  ###
    #######################################################

    def getData(self, argument, index=None):
        """Return class elements.

        Parameters
        ----------
        argument : string
            Keys of self.mydict plus a few extra quantities.
        index : array integers
            Index of the binaries.

        Returns
        -------
        array
            Array containing the self.mydict[argument][index].

        """
        if argument not in self.mydict:
            if  argument == 'LifeTime':
                if index is not None:
                    return (self.mydict["Dt_binary"][index] + self.mydict["Dt_inspiral"][index])*1e-3 #Gyr
                else:
                    return (self.mydict["Dt_binary"] + self.mydict["Dt_inspiral"])*1e-3 #Gyr
            elif argument == 'Mtot':
                if index is not None:
                    return self.mydict["m_BH1"][index] + self.mydict["m_BH2"][index]
                else:
                    return self.mydict["m_BH1"] + self.mydict["m_BH2"]
            elif argument == 'Mchirp':
                if index is not None:
                    return (self.mydict["m_BH1"][index]*self.mydict["m_BH2"][index]
                           )**(3./5.)/(self.mydict["m_BH1"][index]+self.mydict["m_BH2"][index])**(1./5.)
                else:
                    return (self.mydict["m_BH1"]*self.mydict["m_BH2"])**(3./5.)/(self.mydict["m_BH1"]+self.mydict["m_BH2"])**(1./5.)
            elif argument == 'chi_eff':
                # check if spins and tilts are provided
                if "spin_BH1" in self.mydict.keys():
                    spin_BH1 = self.mydict["spin_BH1"]
                    if "tilt_BH1" in self.mydict.keys():
                        tilt_BH1 = self.mydict["tilt_BH1"]
                    else:
                        tilt_BH1 = np.zeros(len(self.mydict["spin_BH1"])) # approximation
                else:
                    spin_BH1 = np.zeros(len(self.mydict["m_BH1"]))
                    tilt_BH1 = np.zeros(len(self.mydict["m_BH1"]))
                if "spin_BH2" in self.mydict.keys():
                    spin_BH2 = self.mydict["spin_BH2"]
                    if "tilt_BH2" in self.mydict.keys():
                        tilt_BH2 = self.mydict["tilt_BH2"]
                    else:
                        tilt_BH2 = np.zeros(len(self.mydict["spin_BH2"])) # approximation
                else:
                    spin_BH2 = np.zeros(len(self.mydict["m_BH2"]))
                    tilt_BH2 = np.zeros(len(self.mydict["m_BH2"]))
                if index is not None:
                    return (self.mydict["m_BH1"][index]
                            * spin_BH1[index]
                            * np.cos(tilt_BH1[index])
                            + self.mydict["m_BH2"][index]
                            * spin_BH2[index]
                            * np.cos(tilt_BH2[index])) \
                            / (self.mydict["m_BH1"][index]+self.mydict["m_BH2"][index])
                else:
                    return (self.mydict["m_BH1"]
                            * spin_BH1
                            * np.cos(tilt_BH1)
                            + self.mydict["m_BH2"]
                            * spin_BH2
                            * np.cos(tilt_BH2)) \
                            / (self.mydict["m_BH1"]+self.mydict["m_BH2"])
            elif argument == 'pessimistic':
                if index is not None:
                    return np.invert(self.mydict["optimistic"][index])
                else:
                    return np.invert(self.mydict["optimistic"])
            else:
                raise ValueError('Option asked %s is not in class_var_options'%argument)
        if index is not None:
            return self.mydict[argument][index]
        else:
            return self.mydict[argument]

    def chi_eff(self, m_1, m_2, a_1, a_2, tilt_1, tilt_2):
        return (m_1*a_1*np.cos(tilt_1)+m_2*a_2*np.cos(tilt_2))/(m_1+m_2)

    def m_chirp(self, m_1, m_2):
        return (m_1*m_2)**(3./5)/(m_1+m_2)**(1./5)

    def StarFormationRate(self, z):
        """Star formation rate in M_sun yr^-1 Mpc^-3.

        Parameters
        ----------
        z : double
            Cosmological redshift.

        Returns
        -------
        double
            The total mass of stars in M_sun formed per comoving volume Mpc^-3
            per year.
        """
        if self.SFR=="Madau+Fragos17":
            return 0.01 * (1. + z) ** 2.6 / (1. + ((1. + z) / 3.2) ** 6.2)  # M_sun yr^-1 Mpc^-3
        elif self.SFR=="Madau+Dickinson15":
            return 0.015 * (1. + z) ** 2.7 / (1. + ((1. + z) / 2.9) ** 5.6)  # M_sun yr^-1 Mpc^-3
        elif self.SFR=="Neijssel+19":
            return 0.01 * (1. + z) ** 2.77 / (1. + ((1. + z) / 2.9) ** 4.7) # M_sun yr^-1 Mpc^-3
        elif self.SFR=="Mason+15":
            #poorly extrapolated data from Mason et al. (2015)
            SFR_data = np.array([[0., -1.6],[2, -0.95],[4, -1],[6, -1.3],[10, -2.2],[16,-3.7]])
            f_SFR = interp1d(SFR_data[:,0], SFR_data[:,1], kind='linear')
            return 10**f_SFR(z) # M_sun yr^-1 Mpc^-3
        elif self.SFR=='Illustris':
            SFR = self.illustris_data['SFR'] # M_sun yr^-1 Mpc^-3
            redshifts = self.illustris_data['redshifts']
            SFR_interp = interp1d(redshifts, SFR)
            return SFR_interp(z)
        else:
            raise ValueError('Invalid SFR!')

    def getZsun(self):
        if self.Zsun == 'Grevesse+Sauval98':
            return 0.017
        elif self.Zsun == 'Asplund+09':
            return 0.0142
        else:
            raise ValueError('Invalid Zsun!')

    def MeanMetallicity(self, z):
        """Mean metallicity function.

        Parameters
        ----------
        z : double
            Cosmological redshift.
        Z_sun : double
            Sun metallicity.

        Returns
        -------
        double
            Mean metallicty of the universe at the given redhist.

        """
        Z_sun = self.getZsun()

        if self.SFR=="Madau+Fragos17" or self.SFR=="Mason+15" or self.SFR=="Madau+Dickinson15":
            return 10 ** (0.153 - 0.074 * z ** 1.34) * Z_sun
        elif self.SFR=="Neijssel+19":
            return 0.035*10**(0.035*z)
        else:
            raise ValueError('Invalid SFR!')

    def StdMetallicity(self):
        """Standard deviation of the metallicity distribution.

        Log-metallicities are normal distributed around the log-mean-
        metallicity with a starndard deviation retourned by this function.

        Returns
        -------
        double
            Standard deviation of the adopted distribution.

        """
        if isinstance(self.sigma_SFR, str):
            if self.sigma_SFR == 'Bavera+20':
                return 0.5
            elif self.sigma_SFR == "Neijssel+19":
                return 0.39
            else:
                raise ValueError('sigma_SFR!')
        elif isinstance(self.sigma_SFR, float):
            return self.sigma_SFR
        else:
            raise ValueError('sigma_SFR!')

    def ISFR(self, Z):
        """Compute the integrated SFR as in Eq. (B.8) of Bavera et al. (2019).

        Parameters
        ----------
        Z : double
            Metallicity.

        Returns
        -------
        double
            The total mass of stars formed per comoving volume at a given
            metallicity Z.

        """
        # integrand
        def E(z,Omega_m=0.307):
            Omega_L = 1.-Omega_m
            return (Omega_m*(1.+z)**3+Omega_L)**(1./2.)
        def f(z,Z):
            if self.SFR=="Madau+Fragos17" or self.SFR=="Mason+15":
                sigma = self.StdMetallicity()
                mu = np.log10(self.MeanMetallicity(z))-sigma**2*np.log(10)/2.
                H_0 = Planck15.H0.to('1/yr').value # yr
                #put a cutoff on metallicity at Z_max
                norm = stats.norm.cdf(np.log10(self.Z_max), mu, sigma)
                return self.StarFormationRate(z)*stats.norm.pdf(np.log10(Z), mu, sigma)/norm*(H_0*(1.+z)*E(z))**(-1)
            elif self.SFR=="Neijssel+19":
                sigma = self.StdMetallicity()
                mu = np.log10(self.MeanMetallicity(z))-sigma**2/2.
                H_0 = Planck15.H0.to('1/yr').value # yr
                return self.StarFormationRate(z)*stats.norm.pdf(np.log(Z), mu, sigma)*(H_0*(1.+z)*E(z))**(-1)
            else:
                raise ValueError('Invalid SFR!')

        return sp.integrate.quad(f, 1e-10, np.inf, args=(Z,))[0] # M_sun yr^-1 Mpc^-3

    def getCosmologicalTime(self, z):
        """Compute the age of the universe.

        Parameters
        ----------
        z : double
            Cosmological redshift.

        Returns
        -------
        double
            Return age of the universe in Gyr at redshift z.

        """
        return Planck15.age(z).value  # Gyr

    def getComovingDistance(self, z):
        """Compute the comoving distance.

        Parameters
        ----------
        z : double
            Cosmological redshift.

        Returns
        -------
        double
            Comoving distance in Mpc corresponding to the cosmological
            redhisft z.

        """
        # check the integral manually, it's ok
        # h=0.673, c=299792.458
        # def f(x, Omega_M=0.315, Omega_k=0, Omega_L=0.685):
        #    return 1./(np.sqrt(Omega_M*(1.+x)**3+Omega_k*(1.+x)**2+Omega_L))
        # return c/(100*h)*sp.integrate.quad(f, 0., z)[0] ,
        return Planck15.comoving_distance(z).value  # Mpc

    def getTimeMerger(self, z_birth):
        """Get age of the universe at the time of the BBH merger.

        Parameters
        ----------
        z_birth : double
            Cosmological redshift of formation of the BBH system
            (must be the same for every binary).

        Returns
        -------
        double
            Age of the universe in Gyr at merger time for of all binaries born at
            z_birth

        """
        n = len(self.mydict["m_BH1"])
        t_birth = self.getCosmologicalTime(z_birth) * np.ones(n)  # Gyr
        return t_birth + (self.mydict["Dt_binary"] + self.mydict["Dt_inspiral"]) * 10 ** (-3)  #Gyr

    def getRedshiftMerger(self, z_birth):
        """Get redshift of merger of BBHs.

        Parameters
        ----------
        z_birth : double
            Redshift of formation of the BBH system (must be the same for every
            binary).

        Returns
        -------
        double
            Redshift of merger of BBHs born at z_birth.

        """
        n = len(self.mydict["m_BH1"])
        t_merger = self.getTimeMerger(z_birth)
        z_merger = np.ones(n) * np.nan
        bool_merger = t_merger < self.getCosmologicalTime(0.) * np.ones(n)  # check if the binary merges
        z_merger[bool_merger] = z_at_value(Planck15.age, t_merger[bool_merger] * u.Gyr)
        return z_merger

    def get_f_binary(self):
        """Binary fraction.

        Returns
        -------
        double
            Binary fraction of the IMF.
        """
        if self.fbin == 'Sana+12':
            f_bin=0.7
        else:
            raise ValueError('Invalid fbin!')
        return f_bin


    def getConstantsRenormalization(self):
        """IMF parameters.

        Returns
        -------
        double
            Parameters for the renormalization factor Appendix A in Bavera et at. (2019).

        """
        f_bin = self.get_f_binary()

        if self.IMF == 'Krupa+2001':
            m_min=0.01 # minimum star mass to support hydrogen fusion
            m_max=150. # maximum star mass
            m_A=5.     # minimum mass of a primary star that can lead to the formation of a BH
            m_B=150.   # maximum star mass
            m_1 = 0.08 #IMF Krupa et al. 2001
            m_2 = 0.5  #IMF Krupa et al. 2001
            a_1=0.3    #IMF Krupa et al. 2001
            a_2=1.3    #IMF Krupa et al. 2001
            a_3=2.3    #IMF Krupa et al. 2001
        elif self.IMF == 'Krupa+2002':
            m_min=0.01 # minimum star mass to support hydrogen fusion
            m_max=150. # maximum star mass
            m_A=5.     # minimum mass of a primary star that can lead to the formation of a BH
            m_B=150.   # maximum star mass
            m_1=0.5   #IMF Krupa et al. 2002
            m_2=1.    #IMF Krupa et al. 2002
            a_1=1.3   #IMF Krupa et al. 2002
            a_2=2.2   #IMF Krupa et al. 2002
            a_3=2.7   #IMF Krupa et al. 2002
        elif self.IMF == 'Salpter':
            m_min=0.01 # minimum star mass to support hydrogen fusion
            m_max=150. # maximum star mass
            m_A=5.     # minimum mass of a primary star that can lead to the formation of a BH
            m_B=150.   # maximum star mass
            m_1=0.08    #IMF Salpter et al. 1988
            m_2=0.5     #IMF Salpter et al. 1988
            a_1=2.35    #IMF Salpter et al. 1988
            a_2=2.35    #IMF Salpter et al. 1988
            a_3=2.35    #IMF Salpter et al. 1988
        else:
            raise ValueError('Invalid IMF!')

        return f_bin, m_min, m_max, m_A, m_B, m_1, m_2, a_1, a_2, a_3

    def f_0(self):
        """IMF renormalization constant.

        Returns
        -------
        double
            Renormalization constant f_0 as in eq. A.1 of Bavera et at. (2019).

        """
        f_bin, m_min, m_max, m_A, m_B, m_1, m_2, a_1, a_2, a_3 = self.getConstantsRenormalization()
        return (1. / (1. - a_1) * ((m_1 ** (1. - a_1) - m_min ** (1. - a_1)) / m_min ** (-a_1))
                + 1. / (1. - a_2) * (m_1 / m_min) ** (-a_1)
                * ((m_2 ** (1. - a_2) - m_1 ** (1. - a_2)) / m_1 ** (-a_2))
                + 1. / (1. - a_3) * (m_1 / m_min) ** (-a_1) * (m_2 / m_1) ** (-a_2)
                * ((m_max ** (1. - a_3) - m_2 ** (1. - a_3)) / m_2 ** (-a_3))) ** (-1)

    def Mean_MassOfSystem(self):
        """Mean stellar systems mass.

        Returns
        -------
        double
            Mean system mass in M_sun, i.e. a star or a binary,
            in the population as in eq. A.2 of Bavera et at. (2019).

        """
        f_bin, m_min, m_max, m_A, m_B, m_1, m_2, a_1, a_2, a_3 = self.getConstantsRenormalization()
        return ((1. + f_bin / 2.) * self.f_0()
                * (1. / (2. - a_1) * ((m_1 ** (2. - a_1) - m_min ** (2. - a_1)) / m_min ** (-a_1))
                   + 1. / (2. - a_2) * (m_1 / m_min) ** (-a_1)
                   * ((m_2 ** (2. - a_2) - m_1 ** (2. - a_2)) / m_1 ** (-a_2))
                   + 1. / (2. - a_3) * (m_1 / m_min) ** (-a_1) * (m_2 / m_1) ** (-a_2)
                   * ((m_max ** (2. - a_3) - m_2 ** (2. - a_3)) / m_2 ** (-a_3))))

    def Mean_MassOfBinaries(self):
        """Mean binary mass.

        Returns
        -------
        double
            Mean binary system mass in M_sun in the population as in eq. A.4 of
            Bavera et at. (2019).

        """
        f_bin, m_min, m_max, m_A, m_B, m_1, m_2, a_1, a_2, a_3 = self.getConstantsRenormalization()
        return (3. / 2.  * (1 - a_3) / (2. - a_3)
                * ((m_B ** (2. - a_3) - m_A ** (2. - a_3)) / (m_B ** (1 - a_3) - m_A ** (1 - a_3))))

    def f_Model(self):
        """Fraction of systems modeled.

        Returns
        -------
        double
            Fraction of systems with masses between m_A and m_B modeled from
            the underlying population as in eq. A.3 of Bavera et at. (2019).

        """
        f_bin, m_min, m_max, m_A, m_B, m_1, m_2, a_1, a_2, a_3 = self.getConstantsRenormalization()
        return (f_bin * self.f_0() * 1. / (1. - a_3) * (m_1 / m_min) ** (-a_1) * (m_2 / m_1) ** (-a_2)
                * ((m_B ** (1. - a_3) - m_A ** (1. - a_3)) / m_2 ** (-a_3)))

    def fCorrection(self):
        """Short summary.

        Returns
        -------
        double
            Renormalization constant of the total modeled mass as in eq. A.4 of
            Bavera et at. (2019). The underlying stellar population mass is
            M_* = M_*,model / fCorrection.

        """
        f_bin, m_min, m_max, m_A, m_B, m_1, m_2, a_1, a_2, a_3 = self.getConstantsRenormalization()
        return self.f_Model() * self.Mean_MassOfBinaries()/self.Mean_MassOfSystem()

    def bin_met_center(self):
        """Return metallicity bin centers.

        Returns
        -------
        array double
            Returns all the metallicities of the populattion. This correponds
            to the center of each metallicity bin.

        """
        met_val = []
        metallicity = self.mydict["metallicity"]
        for i in range(len(metallicity)):
            if metallicity[i] not in met_val:
                met_val.append(metallicity[i])
        met_val = np.array(np.sort(met_val))
        return met_val

    def bin_met_boundary(self):
        """Short summary.

        Returns
        -------
        array double
            Returns the boundaries of all log-metallicity bins.

        """
        met_val = np.log10(self.bin_met_center())
        bin_met = np.zeros(len(met_val)+1)
        # more than one metallicty is passed
        if len(met_val) > 1 :
            bin_met[0] = met_val[0] - (met_val[1] - met_val[0]) / 2.
            bin_met[-1] = met_val[-1] + (met_val[-1] - met_val[-2]) / 2.
            bin_met[1:-1] = met_val[:-1] + (met_val[1:] - met_val[:-1]) / 2.
        # just one metallicty is passed
        elif len(met_val) == 1 :
            if isinstance(self.dlogZ, float):
                #dlogZ = 0.085 # estimated from the 30 bins of Bavera et al. (2020)
                bin_met[0] = met_val[0] - self.dlogZ / 2.
                bin_met[-1] = met_val[0] + self.dlogZ / 2.
            elif isinstance(self.dlogZ, list) or isinstance(self.dlogZ, np.array):
                bin_met[0] = self.dlogZ[0]
                bin_met[-1] = self.dlogZ[1]

        return 10**bin_met

    def MergerRateDensity(self, z_birth, i):
        """Compute the merger rate density.

        Parameters
        ----------
        z_birth : array doubles
            Cosmological redshift of formation of the binary systems. This MUST
            be the same for all binaries.
        i : array integers
            Index array correponding to the binaries you want from the population.

        Returns
        -------
        array doubles
            Return the merger rate density for each binary. This
            quantity correponds to the first two terms in eq. B.7 of Bavera et
            al. (2019), i.e. fSFR(z_birth)*fCorrection/M_sim_metallicty_bin

        """

        # load quantities
        SFR_at_z_birth = self.StarFormationRate(z_birth)

        # get metallicity bin boundary
        bins = self.bin_met_boundary()

        # distribute the metallicity to the corresponding bin
        binplace = np.digitize(self.getData("metallicity",i), bins)

        # compute bin size
        DeltaZ = bins[binplace]-bins[binplace-1]

        # compute fSFR assuming log-normal distributed metallicities
        if self.SFR == "Madau+Fragos17" or self.SFR == "Mason+15" or self.SFR=="Madau+Dickinson15":
            sigma = self.StdMetallicity()
            mu = np.log10(self.MeanMetallicity(z_birth))-sigma**2*np.log(10)/2.
            # put a cutoff on metallicity at Z_max
            norm = stats.norm.cdf(np.log10(self.Z_max), mu[0], sigma)
            fSFR = SFR_at_z_birth*(stats.norm.cdf(np.log10(bins[binplace]), mu, sigma)/norm
                                   - stats.norm.cdf(np.log10(bins[binplace-1]), mu, sigma)/norm)
            if self.select_one_met is None:
                #left edge
                fSFR[binplace == 1] = SFR_at_z_birth[0]*stats.norm.cdf(np.log10(bins[1]), mu[0], sigma)/norm
            #right edge
            #fSFR[binplace == 29] = SFR_at_z_birth[0]*(stats.norm.cdf(np.inf, mu[0], sigma)
            #                       - stats.norm.cdf(np.log10(bins[29]), mu[0], sigma))
        elif self.SFR == "Neijssel+19":
            sigma = self.StdMetallicity()
            mu = np.log(self.MeanMetallicity(z_birth))-sigma**2/2.
            # put a cutoff on metallicity at Z_max
            norm = stats.norm.cdf(np.log10(self.Z_max), mu[0], sigma)
            fSFR = SFR_at_z_birth*(stats.norm.cdf(np.log(bins[binplace]), mu, sigma)/norm
                                   - stats.norm.cdf(np.log(bins[binplace-1]), mu, sigma)/norm)
            # left edge
            fSFR[binplace == 1] = SFR_at_z_birth[0]*stats.norm.cdf(np.log(bins[1]), mu[0], sigma)/norm
            # right edge
            # fSFR[binplace == 29] = SFR_at_z_birth[0]*(stats.norm.cdf(np.inf, mu[0], sigma)
            #                       - stats.norm.cdf(np.log(bins[29]), mu[0], sigma))
        elif self.SFR == 'Illustris':

            redshifts = self.illustris_data['redshifts']
            Z = self.illustris_data['mets']
            M = self.illustris_data['M'] # Msun
            # only use data within the metallicity bounds (no lower bound)
            valid_met_idxs = np.where(Z <= self.Z_max)[0]
            # get the index of the correct redshift in the data
            redz_idx = np.where(redshifts <= z_birth[0])[0][0]
            # take values of the data at this redshift between our metallicity bounds
            Z_dist = M[redz_idx, valid_met_idxs]
            if Z_dist.sum() == 0.:
                fSFR = np.zeros(len(z_birth))
            else:
                Z_dist_cdf = np.cumsum(Z_dist)/Z_dist.sum()
                Z_dist_cdf_interp = interp1d(np.log10(Z[valid_met_idxs]), Z_dist_cdf, fill_value="extrapolate")
                fSFR = SFR_at_z_birth*(Z_dist_cdf_interp(np.log10(bins[binplace]))-Z_dist_cdf_interp(np.log10(bins[binplace-1])))
                if self.select_one_met is None:
                    #left edge
                    fSFR[binplace == 1] = SFR_at_z_birth[0]*Z_dist_cdf_interp(np.log10(bins[1]))
        elif self.SFR == 'FIRE_MW':
            redshifts = self.FIRE_MW_data['redshifts']
            Z = self.FIRE_MW_data['mets']
            M = self.FIRE_MW_data['M'] # Msun
            # only use data within the metallicity bounds (no lower bound)
            valid_met_idxs = np.where(Z <= self.Z_max)[0]
            # get the index of the correct redshift in the data
            redz_idx = np.where(redshifts <= z_birth[0])[0][0]
            # take values of the data at this redshift between our metallicity bounds
            Z_dist = M[valid_met_idxs, redz_idx] # OPPOSITE ORDER COMPARED TO ILLUSTRIS DATA
            if Z_dist.sum() == 0.:
                fSFR = np.zeros(len(z_birth))
            else:
                Z_dist_cdf = np.cumsum(Z_dist)/Z_dist.sum()
                Z_dist_cdf_interp = interp1d(np.log10(Z[valid_met_idxs]), Z_dist_cdf, fill_value="extrapolate")
                fSFR = SFR_at_z_birth*(Z_dist_cdf_interp(np.log10(bins[binplace]))-Z_dist_cdf_interp(np.log10(bins[binplace-1])))
                if self.select_one_met is None:
                    #left edge
                    fSFR[binplace == 1] = SFR_at_z_birth[0]*Z_dist_cdf_interp(np.log10(bins[1]))
        else:
            raise ValueError('Invalid SFR!')

        if self.MESA_grid:
            M1 = self.getData('m_star1_i', i)
            q = self.getData('q_i', i)
            p = self.getData('p_i', i)
            met = self.getData('metallicity', i)

            f_IMF = np.zeros(len(M1))
            for i in range(len(M1)):
                f_IMF[i] = self.prob_IBF(M1[i], q[i], p[i], Dm_log = 0.025, Dq = 0.2 , Dp = 0.025, metallicity=met[i], permass = True)

            return fSFR*f_IMF #Mpc^-3 yr^-1

        else:
            # load quantities
            M_model = self.getData("totalMassEvolvedForZ",i)
            f_correction = self.fCorrection()
            # fist two coefficients of eq. B.7
            if self.f_corr:
                return fSFR*f_correction/M_model #Mpc^-3 yr^-1
            else:
                return fSFR/M_model #Mpc^-3 yr^-1

    def MergerRate(self, z_birth, z_merger, p_det, i):
        """Compute the cosmological weight.

        Parameters
        ----------
        z_birth : array doubles
            Cosmological redshift of formation of the binary systems. This MUST
            be the same for all binaries.
        z_merger : array doubles
            Cosmological redshift of merger of the binary systems.
        p_det : array doubles
            Detection probability of the binary.
        i : array integers
            Index array correponding to the binaries you want from the population.

        Returns
        -------
        array doubles
            Return the cosmological weioght (detection rate contibution) of
            the binary k in the metallicity bin j born at redshift birth i
            as in eq. 12 of Bavera et al. (2019).

        """
        # get quantities
        c = const.c.to('Mpc/yr').value  # Mpc/yr
        deltaT = 100 * 10 ** 6  # yr
        D_c = self.getComovingDistance(z_merger)

        # first two coefficients of eq. B.7
        r_i = self.MergerRateDensity(z_birth, i)

        # eq. 12
        return 4. * np.pi * c * D_c ** 2 * p_det * r_i * deltaT  # yr^-1


    def getRedshiftBinCenter(self):
        """Redshift of birth centers for 100 Myr bins.

        Returns
        -------
        array doubles
            Devide the cosmic time into 100 Myr bins and compute the
            cosmological redshift correponding to the center of each bin.

        """
        # generate t_birth at the middle of each 100 Myr bin
        t_birth_bin = [Planck15.age(0.).value]
        t_birth = []
        for i in range(138):
            t_birth.append(t_birth_bin[i] - 0.05)
            t_birth_bin.append(t_birth_bin[i] - 0.1)
        t_birth = np.array(t_birth)
        # compute the redshift
        z_birth = []
        for i in range(len(t_birth)):
            z_birth.append(z_at_value(Planck15.age, t_birth[i] * u.Gyr))
        z_birth = np.array(z_birth)
        return z_birth

    def getRedshiftBinEdges(self):
        """Redshift of birth edges for 100 Myr bins.

        Returns
        -------
        array doubles
            Devide the cosmic time into 100 Myr bins and compute the
            cosmological redshift correponding to the edge of each bin.

        """
        # generate t_birth at the middle of each 100 Myr bin
        t_birth_bin = [Planck15.age(0.).value]
        t_birth = []
        for i in range(138):
            t_birth.append(t_birth_bin[i] - 0.05)
            t_birth_bin.append(t_birth_bin[i] - 0.1)
        t_birth = np.array(t_birth)
        # compute the redshift
        z_birth = []
        z_birth_bin = []
        for i in range(len(t_birth)):
            z_birth.append(z_at_value(Planck15.age, t_birth[i] * u.Gyr))
            if i < 137:
                z_birth_bin.append(z_at_value(Planck15.age, t_birth_bin[i+1] * u.Gyr))
        z_birth = np.array(z_birth)
        #z_birth_bin = np.array(z_birth_bin)
        
        #add the first snf lsdy bin edge at z=0. and z=inf.
        z_birth_bin = np.array([0.]+z_birth_bin+[100.])
        
        return z_birth_bin

    def getRedshift_function(self):
        """Funtion to compute the cosmological redshift given the cosmic time.

        Returns
        -------
        array doubles
            Given an array of cosmic time compute the corresponding cosmoloogical
            redshift. This function fits the astropy one, which speeds up the
            computation by a few order of magnitude.

        """
        # interpolation z_merger
        t = np.linspace(1e-2, Planck15.age(1e-08).value*0.9999999, 1000)
        z = np.zeros(1000)
        for i in range(1000):
            z[i] = z_at_value(Planck15.age, t[i] * u.Gyr)
        f_z_m = interp1d(t, z, kind='cubic')
        return f_z_m

    def getRedshift(self, t_cosm):
        """Compute the cosmological redshift given the cosmic time..

        Parameters
        ----------
        t_cosm : array doubles
            Cosmic time to which you want to know the redhisft.

        Returns
        -------
        array doubles
            Given an array of cosmic time compute the corresponding cosmoloogical
            redshift. This function fits the astropy one, which speeds up the
            computation by a few order of magnitude.

        """
        # interpolation z_merger
        t = np.linspace(1e-2, 13.797616, 1000)
        z = np.zeros(1000)
        for i in range(1000):
            z[i] = z_at_value(Planck15.age, t[i] * u.Gyr)
        f_z_m = interp1d(t, z, kind='cubic')
        return f_z_m(t_cosm)

    def RunBBHsSimulation(self, sensitivity, flag_pdet=True, path_to_dir=None, extention='npz'):
        """Compute the cosmological weights of the BBH population.

        This function will create a directory three BBHs/ with 3 subdirectories
        containing the cosmological weights (/s_i), redshift of merger (z_merger)
        and binary indeces for each cosmic time bin.

        Parameters
        ----------
        sensitivity : string
            Choose which GW detector sensitivity you want to use, available:
            'O1': LIGO early high sensitivity, lower blue edge of fig. 1 of arXiv:1304.0670v3
            'O12': sensitivity present in the detector during GW150914
            'O3': LIGO late high sensitivity. lower orange edge of fig. 1 of arXiv:1304.0670v3
            'design': LIGO target design sensitivity, see fig. 1 of arXiv:1304.0670v3
            'infinite': whole BBH population, i.e. p_det = 1
        flag_pdet : bool
            In order to run infinite, p_det should be set to false.
        path_to_dir : string
            Path to the directory where you want to store the cosmological weights.

        """
        #define the path to the directory
        if path_to_dir:
            path = path_to_dir
        else:
            path = '.'

        # check if the folder three exists, otherwise create it
        if 'BBHs' not in os.listdir('%s/'%path):
            os.makedirs('%s/BBHs'%path)

        listDir = os.listdir('%s/BBHs'%path)
        if '%s_sensitivity'%sensitivity not in listDir:
            os.makedirs('%s/BBHs/%s_sensitivity'%(path,sensitivity))

        # number of BBHs
        n = len(self.getData('m_BH2'))

        # index of all BBHs
        index = np.arange(0, n)

        # center of each time bin
        z_birth = self.getRedshiftBinCenter()

        # store the funtion for the cosmological redshift to be used in the loop (is taking time to do the interpolation)
        get_redshift_from_time = self.getRedshift_function()

        # dictionary where we store the data
        data = {'index': {}, 'z_merger': {}, 'weights': {}}

        # run the code
        for i in tqdm(range(len(z_birth))):

            # compute the merger time of each BBHs
            t_merger = self.getTimeMerger(z_birth[i])

            if flag_pdet == True and sensitivity != 'infinite':
                # sort out z_merger <0. and z_merger > 2.05 since p_det=0
                bool_merger = t_merger < Planck15.age(1e-08).value*0.9999999 * np.ones(n)

                # check if the array is empty
                if len(index[bool_merger]) == 0:
                    data['index'][str(z_birth[i])] = np.array([])
                    data['z_merger'][str(z_birth[i])] = np.array([])
                    data['weights'][str(z_birth[i])] = np.array([])
                    continue

                #get quantities
                m_BH1 = self.getData('m_BH1',index[bool_merger])
                m_BH2 = self.getData('m_BH2',index[bool_merger])
                z_m = get_redshift_from_time(t_merger[bool_merger])

                #p_detection from Sebastian Gabel
                DL = [x.value for x in Planck15.luminosity_distance(z_m)]
                snr_threshold = 8.
                p_det = selection_effects.detection_probability(m1=m_BH1, m2=m_BH2, redshift=z_m, distance=DL, snr_threshold=snr_threshold, sensitivity=sensitivity)

                # sort out non-detectable sources
                bool_detect = p_det > 0.

                # check if the array is empty
                if len(index[bool_merger][bool_detect]) == 0:
                    data['index'][str(z_birth[i])] = np.array([])
                    data['z_merger'][str(z_birth[i])] = np.array([])
                    data['weights'][str(z_birth[i])] = np.array([])
                    continue

                # save quantities BBH index and redshift of merger
                data['index'][str(z_birth[i])] = index[bool_merger][bool_detect]
                data['z_merger'][str(z_birth[i])] = z_m[bool_detect]

                # compute and save marger rate weight as in eq. B.6 in Bavera et al. (2019)
                z_b = np.ones(len(index[bool_merger][bool_detect]))*z_birth[i]
                s_i = self.MergerRate(z_b, z_m[bool_detect], p_det[bool_detect], index[bool_merger][bool_detect])
                data['weights'][str(z_birth[i])] = s_i

                #if the next z_birth is outside the SFR interpolated rage stop
                if self.SFR=='Mason+15':
                    if z_birth[i+1]>16.:
                        break

            elif flag_pdet == False and sensitivity == 'infinite':
                # sort out z_merger <0.
                bool_merger = t_merger < Planck15.age(1e-08).value*0.9999999 * np.ones(n)

                # check if the array is empty
                if len(index[bool_merger]) == 0:
                    data['index'][str(z_birth[i])] = np.array([])
                    data['z_merger'][str(z_birth[i])] = np.array([])
                    data['weights'][str(z_birth[i])] = np.array([])
                    continue

                #get quantities
                m_BH1 = self.getData('m_BH1',index[bool_merger])
                m_BH2 = self.getData('m_BH2',index[bool_merger])
                z_m = get_redshift_from_time(t_merger[bool_merger])

                #p_detection == 1
                p_det = np.ones(len(index[bool_merger]))

                # save quantities BBH index and redshift of merger
                data['index'][str(z_birth[i])] = index[bool_merger]
                data['z_merger'][str(z_birth[i])] = z_m

                # compute and save marger rate weight as in eq. B.6 in Bavera et al. (2019)
                z_b = np.ones(len(index[bool_merger]))*z_birth[i]
                s_i = self.MergerRate(z_b, z_m, p_det, index[bool_merger])
                data['weights'][str(z_birth[i])] = s_i

                #if the next z_birth is outside the SFR interpolated rage stop
                if self.SFR=='Mason+15':
                    if z_birth[i+1]>16.:
                        break
            else:
                raise ValueError('Missmatch btw sensitivity and flag_pdet!')

        if extention == 'csv':
            # add empty lines to make all list the same size
            if self.verbose:
                print('Formatting the data ....')
            max_size = 0
            for dict in [data[key] for key in data.keys()]:
                # find the max size
                for key in dict.keys():
                    size = len(dict[key])
                    if size > max_size:
                        max_size = size
                # append empty string to reach max size
                for key in dict.keys():
                    size = len(dict[key])
                    n_to_append = max_size - size
                    if n_to_append > 0:
                        dict[key] = np.array(dict[key].tolist() + ['' for i in range(n_to_append)])

            # save data in csv files
            if self.verbose:
                print('Saving the data ....')
            for key in data.keys():
                with open("%s/BBHs/%s_sensitivity/%s.csv"%(path, sensitivity, key), "w") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(data[key].keys())
                    writer.writerows(zip(*data[key].values()))

        elif extention == 'npz':
            if self.verbose:
                print('Formatting the data ....')
            data_to_save = [[],[],[],[]]
            for i, dict in enumerate([data[key] for key in data.keys()]):
                for key in dict.keys():
                    data_to_save[i].extend(dict[key].tolist())
                    if i == 0:
                        data_to_save[3].extend(np.ones(len(dict[key]))*float(key))
            if self.verbose:
                print('Saving the data ....')
            for i, key in enumerate(data.keys()):
                if key == 'index':
                    fmt_str = '%i'
                else:
                    fmt_str = '%.8E'
                np.savez("%s/BBHs/%s_sensitivity/%s.npz"%(path, sensitivity, key), key=data_to_save[i], fmt=fmt_str)

            np.savez("%s/BBHs/%s_sensitivity/z_formation.npz"%(path, sensitivity), key=data_to_save[3], fmt='%.8E')
        else:
            raise ValueError('Extension not supported!')

    def loadBBHsSimulation(self, sensitivity, path_to_dir=None, extention='npz'):
        """Load the cosmological weights.

        Parameters
        ----------
        sensitivity : string
            Choose which GW detector sensitivity you want to use, available:
            'O1': LIGO early high sensitivity, lower blue edge of fig. 1 of arXiv:1304.0670v3
            'O12': sensitivity present in the detector during GW150914
            'O3': LIGO late high sensitivity. lower orange edge of fig. 1 of arXiv:1304.0670v3
            'design': LIGO target design sensitivity, see fig. 1 of arXiv:1304.0670v3
            'infinite': whole BBH population, i.e. p_det = 1
        path_to_dir : string
            Path to the directory where you the cosmological weights are stored.

        Returns
        -------
        array doubles
            For every cosmic time bin this function returns an arrays containing
            the redshifts of formatin and merger, binary indecies and the cosmological
            weights.

        """
        #define the path to the directory
        if path_to_dir:
            path = path_to_dir
        else:
            path = '.'

        if extention == 'csv':
            index = []
            z_formation = []
            z_merger = []
            weights = []

            file_index = pd.read_csv('%s/BBHs/%s_sensitivity/index.csv'%(path,sensitivity))
            file_z_merger = pd.read_csv('%s/BBHs/%s_sensitivity/z_merger.csv'%(path,sensitivity))
            file_weights = pd.read_csv('%s/BBHs/%s_sensitivity/weights.csv'%(path,sensitivity))

            columns_names = file_index.columns
            colums = file_index.columns.to_numpy().astype(float)

            for i, column in tqdm(enumerate(columns_names)):
                ind = file_index[column].to_numpy()
                index.extend(ind[np.invert(np.isnan(ind))].astype(int))
                z_f = np.ones(len(ind[np.invert(np.isnan(ind))]))*colums[i]
                z_formation.extend(z_f)
                z_m = file_z_merger[column].to_numpy()
                z_merger.extend(z_m[np.invert(np.isnan(z_m))])
                s_i = file_weights[column].to_numpy()
                weights.extend(s_i[np.invert(np.isnan(s_i))])

            index = np.array(index)
            z_formation = np.array(z_formation)
            z_merger = np.array(z_merger)
            weights = np.array(weights)

        elif extention == 'npz':
            if self.verbose:
                print('Loading the data ...')
            index = np.load('%s/BBHs/%s_sensitivity/index.npz'%(path,sensitivity), allow_pickle=True)['key']
            z_formation = np.load('%s/BBHs/%s_sensitivity/z_formation.npz'%(path,sensitivity), allow_pickle=True)['key']
            z_merger = np.load('%s/BBHs/%s_sensitivity/z_merger.npz'%(path,sensitivity), allow_pickle=True)['key']
            weights = np.load('%s/BBHs/%s_sensitivity/weights.npz'%(path,sensitivity), allow_pickle=True)['key']
        else:
            raise ValueError('Extension not supported!')
        return index, z_formation, z_merger, weights


    ############################
    ###  LGRB rate densities ###
    ### see arXiv_2106.15841 ###
    ############################


    def getTimeGRB(self, z_birth):
        """Get the time of the GRB.

        Parameters
        ----------
        z_brith : double
            Redshif of birth.

        Returns
        -------
        t_BRB : double
            Cosmic time of the GRB event in Gyr.

        """
        n = len(self.mydict["m_BH1"])
        t_birth = self.getCosmologicalTime(z_birth) * np.ones(n)  # Gyr
        t_GRB = t_birth + self.mydict["Dt_binary"] * 10 ** (-3) # Gyr
        return t_GRB

    def RunGRBsSimulation(self, sensitivity='infinite', path_to_dir=None, extention='npz'):
        """Compute the cosmological weights of the GRB population.

        This function will create a directory three GRBs/ with 3 subdirectories
        containing the cosmological weights (/s_i), redshift of merger (z_merger)
        and binary indeces for each cosmic time bin.

        Parameters
        ----------
        sensitivity : string
            'infinite': whole GRB population, i.e. p_det = 1
        path_to_dir : string
            Path to the directory where you want to store the cosmological weights.

        """

        #define the path to the directory
        if path_to_dir:
            path = path_to_dir
        else:
            path = '.'

        # check if the folder three exists, otherwise create it
        if 'GRBs' not in os.listdir('%s/'%path):
            os.makedirs('%s/GRBs'%path)

        listDir = os.listdir('%s/GRBs'%path)
        if '%s_sensitivity'%sensitivity not in listDir:
            os.makedirs('%s/GRBs/%s_sensitivity'%(path,sensitivity))

        # number of BBHs
        n = len(self.getData('m_BH1'))

        # index of all BBHs
        index = np.arange(0, n)

        # center of each time bin
        z_birth = self.getRedshiftBinCenter()

        # store the funtion for the cosmological redshift to be used in the loop (is taking time to do the interpolation)
        get_redshift_from_time = self.getRedshift_function()

        # dictionary where we store the data
        data = {'index': {}, 'z_GRB': {}, 'weights': {}}

        # run the code
        for i in tqdm(range(len(z_birth))):

            # compute the GRB time of each BBHs
            t_GRB = self.getTimeGRB(z_birth[i])

            # sort out z_GRB <0. and non-GRBs BBHs
            bool_GRB = np.logical_and(t_GRB < Planck15.age(1e-08).value*0.9999999 , self.mydict["flag_GRB"])

            # check if the array is empty
            if len(index[bool_GRB]) == 0:
                data['index'][str(z_birth[i])] = np.array([])
                data['z_GRB'][str(z_birth[i])] = np.array([])
                data['weights'][str(z_birth[i])] = np.array([])
                continue

            #get quantities
            m_BH1 = self.getData('m_BH1',index[bool_GRB])
            m_BH2 = self.getData('m_BH2',index[bool_GRB])
            z_GRB = get_redshift_from_time(t_GRB[bool_GRB])

            if sensitivity=='infinite':
                #p_detection == 1
                p_det = np.ones(len(index[bool_GRB]))
            else:
                raise ValueError('The sensitivity is not infinite!')

            # save quantities BBH index and redshift of GRB
            data['index'][str(z_birth[i])] = index[bool_GRB]
            data['z_GRB'][str(z_birth[i])] = z_GRB

            # compute and save marger rate weight as in eq. B.6 in Bavera et al. (2019)
            z_b = np.ones(len(index[bool_GRB]))*z_birth[i]
            s_i = self.MergerRate(z_b, z_GRB, p_det, index[bool_GRB])
            data['weights'][str(z_birth[i])] = s_i

            #if the next z_birth is outside the SFR interpolated rage stop
            if self.SFR=="Mason+15":
                if z_birth[i+1]>16.:
                    break

        if extention == 'csv':
            # add empty lines to make all list the same size
            if self.verbose:
                print('Formatting the data ....')
            max_size = 0
            for dict in [data[key] for key in data.keys()]:
                # find the max size
                for key in dict.keys():
                    size = len(dict[key])
                    if size > max_size:
                        max_size = size
                # append empty string to reach max size
                for key in dict.keys():
                    size = len(dict[key])
                    n_to_append = max_size - size
                    if n_to_append > 0:
                        dict[key] = np.array(dict[key].tolist() + ['' for i in range(n_to_append)])

            # save data in csv files
            if self.verbose:
                print('Saving the data ....')
            for key in data.keys():
                with open("%s/GRBs/%s_sensitivity/%s.csv"%(path, sensitivity, key), "w") as outfile:
                    writer = csv.writer(outfile)
                    writer.writerow(data[key].keys())
                    writer.writerows(zip(*data[key].values()))

        elif extention == 'npz':
            if self.verbose:
                print('Formatting the data ....')
            data_to_save = [[],[],[],[]]
            for i, dict in enumerate([data[key] for key in data.keys()]):
                for key in dict.keys():
                    data_to_save[i].extend(dict[key].tolist())
                    if i == 0:
                        data_to_save[3].extend(np.ones(len(dict[key]))*float(key))

            if self.verbose:
                print('Saving the data ....')
            for i, key in enumerate(data.keys()):
                if key == 'index':
                    fmt_str = '%i'
                else:
                    fmt_str = '%.8E'
                np.savez("%s/GRBs/%s_sensitivity/%s.npz"%(path, sensitivity, key), key=data_to_save[i], fmt=fmt_str)

            np.savez("%s/GRBs/%s_sensitivity/z_formation.npz"%(path, sensitivity), key=data_to_save[3], fmt='%.8E')
        else:
            raise ValueError('Extension not supported!')

    def loadGRBsSimulation(self, sensitivity='infinite', path_to_dir=None, extention='npz'):
        """Load the cosmological weights.

        Parameters
        ----------
        sensitivity : string
            'infinite': whole GRB population, i.e. p_det = 1
        path_to_dir : string
            Path to the directory where you the cosmological weights are stored.

        Returns
        -------
        array doubles
            For every cosmic time bin this function returns an arrays containing
            the redshifts of formatin and merger, binary indecies and the cosmological
            weights.

        """
        #define the path to the directory
        if path_to_dir:
            path = path_to_dir
        else:
            path = '.'

        if extention == 'csv':
            index = []
            z_formation = []
            z_GRB = []
            weights = []

            file_index = pd.read_csv('%s/GRBs/%s_sensitivity/index.csv'%(path,sensitivity))
            file_z_GRB = pd.read_csv('%s/GRBs/%s_sensitivity/z_GRB.csv'%(path,sensitivity))
            file_weights = pd.read_csv('%s/GRBs/%s_sensitivity/weights.csv'%(path,sensitivity))

            columns_names = file_index.columns
            colums = file_index.columns.to_numpy().astype(float)

            for i, column in tqdm(enumerate(columns_names)):
                ind = file_index[column].to_numpy()
                index.extend(ind[np.invert(np.isnan(ind))].astype(int))
                z_f = np.ones(len(ind[np.invert(np.isnan(ind))]))*colums[i]
                z_formation.extend(z_f)
                z_g = file_z_GRB[column].to_numpy()
                z_GRB.extend(z_g[np.invert(np.isnan(z_g))])
                s_i = file_weights[column].to_numpy()
                weights.extend(s_i[np.invert(np.isnan(s_i))])

            index = np.array(index)
            z_formation = np.array(z_formation)
            z_GRB = np.array(z_GRB)
            weights = np.array(weights)

        elif extention == 'npz':
            if self.verbose:
                print('Loading the data ...')
            index = np.load('%s/GRBs/%s_sensitivity/index.npz'%(path,sensitivity), allow_pickle=True)['key']
            z_formation = np.load('%s/GRBs/%s_sensitivity/z_formation.npz'%(path,sensitivity), allow_pickle=True)['key']
            z_GRB = np.load('%s/GRBs/%s_sensitivity/z_GRB.npz'%(path,sensitivity), allow_pickle=True)['key']
            weights = np.load('%s/GRBs/%s_sensitivity/weights.npz'%(path,sensitivity), allow_pickle=True)['key']
        else:
            raise ValueError('Extension not supported!')
        return index, z_formation, z_GRB, weights

    def ComovingVolume(self, z_hor_i, z_hor_f, sensitivity='infinite'):
        """Compute comoving volume.

        Parameters
        ----------
        z_hor_i : double
            Cosmological redshift. Lower bound of the integration.
        z_hor_f : double
            Cosmological redshift. Upper bound of the integration.
        sensitivity : string
            hoose which GW detector sensitivity you want to use. At the moment
            only 'infinite' is available, i.e. p_det = 1.

        Returns
        -------
        double
            This function returns the comoving volume withing a peanutshaoed
            antenna volume for the specific merger event. If sensitivity='infinite'
            this correpond to the comoving volume between the two shells z_hor_i
            and z_hor_f.

        """
        c = const.c.to('Gpc/yr').value  # Gpc/yr
        H_0 = Planck15.H(0).to('1/yr').value # km/Gpc*s
        def E(z):
            Omega_m = Planck15.Om0
            Omega_L = 1-Planck15.Om0
            return np.sqrt(Omega_m*(1.+z)**3+Omega_L)
        def f(z,sensitivity):
            if sensitivity=='infinite':
                return 1./(1.+z)*4*np.pi*c/H_0*(self.getComovingDistance(z)*10**(-3.))**2./E(z)
            else:
                # still need to code p_det for BBHs or others
                raise ValueError('Sensitivity is not equal to infinite!')
        return sp.integrate.quad(f, z_hor_i, z_hor_f, args=(sensitivity))[0] # Gpc^3

    def BeamingFactor(self, n, mechanism='Goldstein+15', z=None):
        """Computte the beaming factor probability of the GRB event.

        Parameters
        ----------
        n : integer
            Number of beaming factor to return
        mechanism : string
            Chose the time of distribution to draw the beaming factor from:
            'Goldstein+15': The opening angles are drawn from a log-normal distribution see arXiv:1512.04464.
            'median-f_b=0.01':
            'median-f_b=0.02':
            'median-f_b=0.04':
            'median-f_b=0.05':
            'Lloyd-Ronning+19': The opening angles are drawn from a Goldstein+15-like log-normal distribution
                                and the mean decreases as the redshift increases.
        z : array doubles
            If you choose 'Lloyd-Ronning+19' than you need to pass a redshift.

        Returns
        -------
        p_opening_angle : array doubles
            Probability of seeing a GRB given a distribution of opeing angle.
        """
        if mechanism == 'Goldstein+15':
            # generate a opening angle from a lognormal distribution as in Table 3 of https://arxiv.org/pdf/1512.04464.pdf
            theta = 10**(np.random.normal(0.77,0.37,n))/180.*np.pi # radian

            # compute the solid angle
            solid_angle = 4*np.pi*(1.-np.cos(theta)) # factor 2 to account for the two emispheres

            # compute the probability
            p_opening_angle = solid_angle/(4*np.pi)

        elif mechanism == 'Goldstein+15-truncated':
            # generate a opening angle from a lognormal distribution as in Table 3 of https://arxiv.org/pdf/1512.04464.pdf
            theta = 10**(np.random.normal(0.77,0.37,n))/180.*np.pi # radian

            redraw = np.logical_or(theta < 2.*np.pi/180., theta > 20.*np.pi/180.)
            while(redraw.any()):
                n_redraw = len(redraw[redraw])
                theta[redraw] = 10**(np.random.normal(0.77,0.37,n_redraw))/180.*np.pi # radian
                redraw = np.logical_or(theta < 2.*np.pi/180., theta > 20.*np.pi/180.)
            # compute the solid angle
            solid_angle = 4*np.pi*(1.-np.cos(theta)) # factor 2 to account for the two emispheres

            # compute the probability
            p_opening_angle = solid_angle/(4*np.pi)

        elif mechanism == 'median-f_b=0.01':

            mu = 0.915
            sigma = 0.2
            theta = 10**(np.random.normal(mu,sigma,n))/180.*np.pi
            limit = 22.5/180.*np.pi
            while len(theta[theta>limit]) > 0:
                theta[theta>limit] = 10**(np.random.normal(mu,sigma,len(theta[theta>limit])))/180.*np.pi
            p_opening_angle = 1. - np.cos(theta)

        elif mechanism == 'median-f_b=0.02':

            mu = 1.08
            sigma = 0.2
            theta = 10**(np.random.normal(mu,sigma,n))/180.*np.pi
            limit = 22.5/180.*np.pi
            while len(theta[theta>limit]) > 0:
                theta[theta>limit] = 10**(np.random.normal(mu,sigma,len(theta[theta>limit])))/180.*np.pi
            p_opening_angle = 1. - np.cos(theta)

        elif mechanism == 'median-f_b=0.04':

            mu = 1.33
            sigma = 0.2
            theta = 10**(np.random.normal(mu,sigma,n))/180.*np.pi
            limit = 22.5/180.*np.pi
            while len(theta[theta>limit]) > 0:
                theta[theta>limit] = 10**(np.random.normal(mu,sigma,len(theta[theta>limit])))/180.*np.pi
            p_opening_angle = 1. - np.cos(theta)

        elif mechanism == 'median-f_b=0.05':

            mu = 1.5
            sigma = 0.2
            theta = 10**(np.random.normal(mu,sigma,n))/180.*np.pi
            limit = 22.5/180.*np.pi
            while len(theta[theta>limit]) > 0:
                theta[theta>limit] = 10**(np.random.normal(mu,sigma,len(theta[theta>limit])))/180.*np.pi
            p_opening_angle = 1. - np.cos(theta)

        elif mechanism == 'Lloyd-Ronning+19':
            if z is None:
                raise ValueError('You need to pass the redshift!')
            # generate a opening angle from a lognormal distribution as in Table 3 of https://arxiv.org/pdf/1512.04464.pdf
            theta =10**(np.random.normal(0.77*(1+z)**(-0.75),0.37,n))/180.*np.pi # radian

            # compute the solid angle
            solid_angle = 4*np.pi*(1.-np.cos(theta)) # factor 2 to account for the two emispheres

            # compute the probability
            p_opening_angle = solid_angle/(4*np.pi)

        elif 'const=' in mechanism:
            f_b = float(mechanism.replace('const=',''))
            p_opening_angle = np.ones(n)*f_b

        elif mechanism == 'Pescalli+15':
            def f_B(E,eta):
                xi=0.5
                t=25.
                C = (7e46)**xi
                fb = np.ones(len(E))
                fb[E>0.] = (C*(eta*E[E>0.]/t)**(-xi))**(1./(1-xi))
                return fb
            p_opening_angle = f_B(self.mydict["E_GRB"],self.eta)

        return p_opening_angle

    def RateDensity(self, s_i, z_event, Type='GRBs', sensitivity='beamed', index=None):
        """Compute the BBHs merger / GRBs (coming soon) rate density.

        Parameters
        ----------
        s_i : array doubles
            Cosmological weights computed with Eq. B.7 of Bavera et at. (2019).
        z_event : array doubles
            Cosmolgocial redshift of the event you are tracking.
        Type : string
            Event you are tracking, available:
            'BBHs': merger event of a BBH system
            'GRBs': coming soon.
        sensitivity : string
            This takes into account the detector sensitivity, available:
            'infinite': p_det = 1
            'beamed': coming soon (for GRBs)

        Returns
        -------
        array doubles
            Return the BBHs merger rate density (Gpc^-3 yr^-1) as a function of
            cosmolgocial redshift..

        """

        z_hor = self.getRedshiftBinEdges()
        n = len(z_hor)

        if Type=='GRBs':
            z_GRB = z_event
            if sensitivity=='beamed':
                if index is not None:
                    f_beaming = self.mydict["f_beaming"][index]
                else:
                    raise ValueError('You should pass the index to generate the f_beaming array.')
                Rate_GRBs_beaming = np.zeros(n-1)
                for i in range(1,n):
                    #z_mid = self.getRedshiftBinCenter()
                    condition_GRB_beaming = np.logical_and(np.logical_and(z_GRB>z_hor[i-1], z_GRB<=z_hor[i]),self.mydict["flag_GRB"][index])
                    f_beam = f_beaming[condition_GRB_beaming]
                    Rate_GRBs_beaming[i-1] = sum(s_i[condition_GRB_beaming]*f_beam)/self.ComovingVolume(z_hor[i-1],z_hor[i],sensitivity='infinite')
                return Rate_GRBs_beaming

            elif sensitivity=='infinite':
                Rate_GRBs = np.zeros(n-1)
                for i in range(1,n):
                    if index is None:
                        raise ValueError('You should pass the index to generate the f_beaming array.')
                    condition_GRB = np.logical_and(np.logical_and(z_GRB>z_hor[i-1], z_GRB<=z_hor[i]),self.mydict["flag_GRB"][index])
                    Rate_GRBs[i-1] = sum(s_i[condition_GRB])/self.ComovingVolume(z_hor[i-1],z_hor[i],sensitivity='infinite')
                return Rate_GRBs
            else:
                raise ValueError('Unknown sensitivity!')

        elif Type=='BBHs':
            z_merger_BBH = z_event
            Rate_BBHs = np.zeros(n-1)
            if sensitivity=='infinite':
                for i in range(1,n):
                        condition_BBHs = np.logical_and(z_merger_BBH>z_hor[i-1], z_merger_BBH<=z_hor[i])
                        Rate_BBHs[i-1] = sum(s_i[condition_BBHs])/self.ComovingVolume(z_hor[i-1],z_hor[i],sensitivity)
                return Rate_BBHs
            else:
                raise ValueError('Unknown sensitivity!')

        else:
            raise ValueError('Unknown Type!')


    #############################################
    ###        weighting a MESA grid          ###
    ### phase space volume of a MESA track    ###
    ###   see appendix A1.2 arXiv:2011.10057  ###
    #############################################

    def prob_IBF(self, m, q, p, Dm = None, Dq = None , Dp = None, Dm_log=None, Dq_log=None, Dp_log=None , metallicity=None, permass = False):
        """Probability of a binary given an IMF

        Parameters
        ----------
        m : double
            Mass of the primary in Msun.
        q : double
            Binary mass fraction (0<q<1).
        p : double
            Orbital period in days.
        Dm : double
            Dimention of the mass bin in Msun.
        Dq : double
            Dimension of the mass fraction bin.
        Dp : double
            Dimension of the orbital period bin in days.
        Dm_log : double
            Dimention of the log-mass bin in Msun, if assigned Dm will be ignored.
        Dq_log : double
            Dimension of the log-mass fraction bin, if assigned Dm will be ignored.
        Dp_log : double
            Dimension of the log-orbital period bin in days, if assigned Dm will be ignored.
        f_bin : double
            Binary fraction in the IMF (0<f_bin<1).
        permass : bool
            If True then the outcome represents the probability of the formation per solar mass created.
            Else it is per stellar system created (one star or one binary system). To compute this we
            need the average mass per system in a star formation episode, which
            depends on the binary fraction f_bin. Here it is calculated for specific binary fractions,
            options [0,0.5,0.7,1] are available (TODO: this numbers should coputed from the other
            functions I have in the class. For the moment I just check that the number is consistent
            with what I am computing, but it is better if we compute it with the same IMF coefficints
            and mass ranges which here are slightly larger).

        Returns
        -------
        double
            This function calculate the probability of a specific point of the parameter space taking
            into account its volume depending on how sparce your grid is.

        """

        def IMF_prob(m, Dm, Dm_log=None, mmin=0.01, mmax=150):
            """Compute the probability of a system with mass m assuming a IMF.
            """

            if self.IMF == 'Krupa+2001':
                # Powers in the power laws
                a_alpha = -0.3
                a_beta = -1.3
                a_gamma = -2.3
                # Where the power laws break
                m_1 = 0.08
                m_2 = 0.5
                if m_1 < mmin:
                    m_1 = mmin
                if m_2 < mmin:
                    m_2 = mmin
            else:
                raise ValueError('Invalid IMF!')

            # Get the constants K_alpha, K_beta, K_gamma
            # 1) Boundary conditions (K_beta, K_gamma in terms of K_alpha)
            # 2) Normalising function
            j_alpha = (m_1**(a_alpha+1) - mmin**(a_alpha+1))/(a_alpha+1)
            j_beta = (m_2**(a_beta+1) - m_1**(a_beta+1))/(a_beta+1)
            j_gamma = (mmax**(a_gamma+1) - m_2**(a_gamma+1))/(a_gamma+1)
            c_beta = m_1**(a_alpha-a_beta)
            c_gamma = m_2**(a_beta-a_gamma)
            K_alpha = 1/(j_alpha + c_beta*j_beta + c_beta*c_gamma*j_gamma)
            K_beta = K_alpha * (m_1**(a_alpha-a_beta))
            K_gamma = K_beta * (m_2**(a_beta-a_gamma))

            if m < m_1:
                Gamma = 1+a_alpha
                K = K_alpha
            if m > m_1 and m < m_2:
                Gamma = 1+a_beta
                K = K_beta
            if m > m_2:
                Gamma = 1+a_gamma
                K = K_gamma
            if Dm_log is None:
                prob  = K/(Gamma) * ((m+Dm/2.)**(Gamma) - (m-Dm/2.)**(Gamma))
            else:
                prob  = K/(Gamma) * ((10**(np.log10(m)+Dm_log/2.))**(Gamma) - (10**(np.log10(m)-Dm_log/2.))**(Gamma))

            return prob


        def IQF_prob(q, Dq, Dq_log=None, qmin=0. ,qmax=1.0):
            """Compute the probability of a system for mass ratio q (<1), assuming a flat
            distribution (e.g., Sana et al 2012).
            """

            if Dq_log is None:
                norm = 1./(qmax - qmin)
                prob  = norm * Dq
            else:
                raise ValueError('This part of the code should be written!')

            return prob

        def IPF_prob(p, Dp, Dp_log=None, Pmin = 0.4 , Pmax=10**5.5, metallicity=None, distribution='Sana+12'):
            """Compute the probability of a system with orbital period p assuming a uniform log orbital separation.
            """

            if distribution == 'uniform':

                if Dp_log is None:
                    norm = 1./(np.log10(Pmax)-np.log10(Pmin))
                    left = np.log10(p-Dp/2)
                    right = np.log10(p+Dp/2)
                    prob = norm*np.abs(right-left)
                else:
                    norm = 1./(np.log10(Pmax)-np.log10(Pmin))
                    left = np.log10(p)-Dp_log/2
                    right = np.log10(p)+Dp_log/2
                    prob = norm*np.abs(right-left)

            elif distribution == 'Sana+12':

                # normalisation constants
                pi = 0.55
                beta = 1 - pi
                A = np.log10(10**0.15)**(-pi)*(np.log10(10**0.15) - np.log10(Pmin))
                B = 1./beta*(np.log10(Pmax)**beta - np.log10(10**0.15)**beta)
                C = 1./(A + B)

                if p < 10**0.15:
                    # the distribution is flat
                    if Dp_log is None:
                        norm = C
                        left = np.log10(p-Dp/2)
                        right = np.log10(p+Dp/2)
                        prob = norm*np.abs(right-left)*0.15**(-pi)
                    else:
                        norm = C
                        left = np.log10(p)-Dp_log/2
                        right = np.log10(p)+Dp_log/2
                        prob = norm*np.abs(right-left)*0.15**(-pi)
                else:

                    if Dp_log is None:
                        norm = C
                        prob  = norm * (1./beta) * ( (np.log10(p+Dp/2.))**beta - (np.log10(p-Dp/2.))**beta)
                    else:
                        norm = C
                        prob  = norm * (1./beta) * (((np.log10(p)+Dp_log/2.))**beta - ((np.log10(p)-Dp_log/2.))**beta)
                        raise ValueError('This is not implemented!')

                if (p < Pmin) or (p > Pmax):
                    prob = 0.0

            else:
                raise ValueError('Invalid distribution!')

            return prob

        f_bin = self.get_f_binary()

        prob = IMF_prob(m, Dm, Dm_log) * IPF_prob(p, Dp, Dp_log, metallicity=metallicity) * IQF_prob(q, Dq, Dq_log) * f_bin

        if permass == True:
            # for f_bin=0.7 and Kroupa+2001 the avg. mass of system is 0.525 Msun
            avg_mass_per_system = self.Mean_MassOfSystem() 
            prob = prob / avg_mass_per_system

        return prob

    ######################################
    ###  gravitational-wave backround  ###
    ###       see arXiv:2109.05836     ###
    ######################################

    def GW_energy_spectrum(self, frequency, index, waveforms='Ajith+08'):
        """Compute gravitational-wave energy spectrum.
        """

        frequency = np.ones(len(index))*frequency
        # no BBH spins
        if waveforms == 'Ajith+08':
            m_BH1 = self.getData('m_BH1',index)
            m_BH2 = self.getData('m_BH2',index)
            M = self.getData('Mtot',index)
            Mchirp = self.getData('Mchirp',index)
            eta = m_BH1*m_BH2/M**2

            def constants(eta, M):
                def model(eta, M, a, b, c):
                    return (a*eta**2.+b*eta+c)/(np.pi*M)
                G = 6.67259e-8 # cm^3 g^-1 s^-2
                c = 2.99792458e10 # cm/s
                Msun = 1.989e33 # g
                M = M*Msun*G/c**3 # s
                nu_1 = model(eta, M, 2.974e-1, 4.481e-2, 9.556e-2)
                nu_2 = model(eta, M, 5.9411e-1, 8.9794e-2, 1.9111e-1)
                sigma = model(eta, M, 5.0801e-1, 7.7515e-2, 2.2369e-2)
                nu_3 = model(eta, M, 8.4845e-1, 1.2848e-1, 2.7299e-1)
                return nu_1, nu_2, nu_3, sigma

            # compute merg, ring, cut consstants
            nu_1, nu_2, nu_3, sigma = constants(eta, M)

            # normalisation
            G = 6.67259e-8 # cm^3 g^-1 s^-2
            Msun = 1.989e33 # g
            norm = (G*np.pi)**(2./3)*(Mchirp*Msun)**(5./3)/3.
            omega_1 = nu_1**(-1.)
            omega_2 = nu_1**(-1.)*nu_2**(-4./3)
            kappa = (frequency/(1+((frequency-nu_2)/(sigma/2.))**2.))**2.
            dEdv = np.where(frequency < nu_1,
                        norm*frequency**(-1./3),
                        np.where(frequency < nu_2,
                                 norm*omega_1*frequency**(2./3),
                                 np.where(frequency < nu_3,
                                          norm*omega_2*kappa, 0.)))
            return dEdv

        # accounting for non-precessing BBH spins
        elif waveforms == 'Ajith+11':
            m_BH1 = self.getData('m_BH1',index)
            m_BH2 = self.getData('m_BH2',index)
            M = self.getData('Mtot',index)
            Mchirp = self.getData('Mchirp',index)
            chi = self.getData('chi_eff',index)
            eta = m_BH1*m_BH2/M**2

            def auxiliary_constant(frequency, M, eta, chi):
                G = 6.67259e-8 # cm^3 g^-1 s^-2
                c = 2.99792458e10 # cm/s
                Msun = 1.989e33 # g
                M = M*Msun*G/c**3 # s
                v = (np.pi*M*frequency)**(1./3)
                alpha2 = -323./224+451./168*eta
                alpha3 = (27./8-11./6*eta)*chi
                epsilon1 = 1.4547*chi-1.8897
                epsilon2 = -1.8153*chi+1.6557
                return v, alpha2, alpha3, epsilon1, epsilon2

            def constants(M, eta, chi):
                G = 6.67259e-8 # cm^3 g^-1 s^-2
                c = 2.99792458e10 # cm/s
                Msun = 1.989e33 # g
                M = M*Msun*G/c**3 # s
                y10, y11, y12, y20, y21, y30 = [0.6437,0.827,-0.2706,-0.05822,-3.935,-7.092]
                corr = y10*eta+y11*eta*chi+y12*eta*chi**2+y20*eta**2+y21*eta**2*chi+y30*eta**3
                nu_1 = 1.-4.455*(1.-chi)**(0.217)+3.521*(1.-chi)**(0.26)+corr
                y10, y11, y12, y20, y21, y30 = [0.1469,-0.1228,-0.02609,-0.0249,0.1701,2.325]
                corr = y10*eta+y11*eta*chi+y12*eta*chi**2+y20*eta**2+y21*eta**2*chi+y30*eta**3
                nu_2 = (1.-0.63*(1.-chi)**(0.3))/2.+corr
                y10, y11, y12, y20, y21, y30 = [-0.4098,-0.03523,0.1008,1.829,-0.02017,-2.87]
                corr = y10*eta+y11*eta*chi+y12*eta*chi**2+y20*eta**2+y21*eta**2*chi+y30*eta**3
                sigma = (1.-0.63*(1.-chi)**(0.3))*(1.-chi)**(0.45)/4.+corr
                y10, y11, y12, y20, y21, y30 = [-0.1331,-0.08172,0.1451,-0.2714,0.1279,4.922]
                corr = y10*eta+y11*eta*chi+y12*eta*chi**2+y20*eta**2+y21*eta**2*chi+y30*eta**3
                nu_3 = 0.3236+0.04894*chi+0.01346*chi**2+corr
                return nu_1/(np.pi*M), nu_2/(np.pi*M), nu_3/(np.pi*M), sigma/(np.pi*M)

            def renomarlisation(M, eta, chi):
                nu_1, nu_2, nu_3, sigma = constants(M, eta, chi)
                v, alpha2, alpha3, epsilon1, epsilon2 = auxiliary_constant(nu_1, M, eta, chi)
                omega_1 = nu_1**(-1.)*(1+alpha2*v**2+alpha3*v**3)**2/(1+epsilon1*v+epsilon2*v**2)**2
                v, alpha2, alpha3, epsilon1, epsilon2 = auxiliary_constant(nu_2, M, eta, chi)
                omega_2 = omega_1*nu_2**(-4./3)*(1+epsilon1*v+epsilon2*v**2)**2
                return omega_1, omega_2

            omega_1, omega_2 = renomarlisation(M, eta, chi)
            v, alpha2, alpha3, epsilon1, epsilon2 = auxiliary_constant(frequency, M, eta, chi)
            nu_1, nu_2, nu_3, sigma = constants(M, eta, chi)

            # normalisation
            G = 6.67259e-8 # cm^3 g^-1 s^-2
            Msun = 1.989e33 # g
            norm = (G*np.pi)**(2./3)*(Mchirp*Msun)**(5./3)/3.
            kappa = (frequency/(1+((frequency-nu_2)/(sigma/2.))**2.))**2.
            dEdv = np.where(frequency < nu_1,
                        norm*frequency**(-1./3)*(1+alpha2*v**2+alpha3*v**3)**2,
                        np.where(frequency < nu_2,
                                 norm*omega_1*frequency**(2./3)*(1+epsilon1*v+epsilon2*v**2)**2,
                                 np.where(frequency < nu_3,
                                          norm*omega_2*kappa, 0.)))
            return dEdv

        else:
            raise ValueError('waveforms not supported!')

    def energy_flux_per_frequency(self, frequency, z, index, waveforms='Ajith+08'):
        """Compute gravitational-wave energy flux.
        """
        # nu_obs = frequency
        nu = frequency*(1+z) # source frame
        dEdv = self.GW_energy_spectrum(nu, index, waveforms)
        d_L = (1+z)*self.getComovingDistance(z)*1e6*3.086e+18 # Mpc to cm
        return dEdv*(1.+z)/(4*np.pi*d_L**2)

    def spectral_energy_density(self, s_i, z_merger, index, frequency, waveforms='Ajith+08'):
        """Compute spectral energy density.
        """
        z_hor = self.getRedshiftBinEdges()
        n = len(z_hor)
        z_center = self.getRedshiftBinCenter()

        dR_f_nu = np.zeros(n-1)
        #dz = np.zeros(n-1)
        for i in range(0,n-1):
            condition_dz = np.logical_and(z_merger>z_hor[i], z_merger<=z_hor[i+1])
            # convert yr^-1 units of s_i to s^-1
            dR_f_nu[i] = sum(s_i[condition_dz]/(365.25*24*3600.)*self.energy_flux_per_frequency(frequency, z_center[i], index[condition_dz], waveforms))
            #dz[i] = z_hor[i+1]-z_hor[i]

        return sum(dR_f_nu)


    def duty_cicle(self, s_i, z_merger, index, nu_min=10.):
        """Compute duty cicly between two gravitational-wave signals in seconds.
        """
        Mchirp = self.getData('Mchirp',index)

        def tau(z, Mchirp, nu_min):
            #z = np.ones(len(Mchirp))*z
            G = 6.67259e-8 # cm^3 g^-1 s^-2
            c = 2.99792458e10 # cm/s
            Msun = 1.989e33 # g
            const = 5*c**5/(256*np.pi**(8./3)*G**(5./3))
            return const*((1.+z)*Mchirp*Msun)**(-5./3)*nu_min**(-8./3)

        z_hor = self.getRedshiftBinEdges()
        n = len(z_hor)
        z_center = self.getRedshiftBinCenter()

        taudR = np.zeros(n-1)
        #dz = np.zeros(n-1)
        for i in range(0,n-1):
            condition_dz = np.logical_and(z_merger>z_hor[i], z_merger<=z_hor[i+1])
            # convert yr^-1 units of s_i to s^-1
            taudR[i] = sum(s_i[condition_dz]/(365*24*3600.)*tau(z_merger[condition_dz], Mchirp[condition_dz], nu_min))
            #dz[i] = z_hor[i+1]-z_hor[i]

        DC = sum(taudR)

        return DC


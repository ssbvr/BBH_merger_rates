# BBH_merging_rates
Compute binary black hole merger rates given a synthetic population obtained through a population synthesis model.

Given some model assumptions, the code can compute the observable merging BBH population by a gravitational-wave detector as well as the intrinsic BBH population (equivalent to a gravitational-wave detector with infinite sensitivity). Additionally, this code computes the BBH detection rate and the BBH merger rate density. If long-duration gamma-ray bursts (LGRBs) are modelled, the code computes distributions and rates of LGRBs. The code can also compute rates provided a synthetic BBH population from a rectilinear MESA gird. Finally, the code also computes the stochastic gravitational-wave background of a given intrinsic merging BBH population.

This repository contains Jupyter notebooks which showcase how to use the different functionalities of the code and provides published population synthesis models of merging BBHs from the isolated binary evolution channel. These models were obtained using the POSYDON framework and are part of the work I conducted during my PhD, see [Unraveling the Origins of Stellar Mass Black Hole Mergers](https://archive-ouverte.unige.ch/unige:162269) by Simone Bavera.

This code was used in the following list of publications:
- [`The formation of 30Msun merging black holes at solar metallicity`, Bavera et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv221210924B/abstract)
- [`The χeff − z correlation of field binary black hole mergers and how 3G gravitational-wave detectors can constrain it`, Bavera et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...665A..59B/abstract)
- [`Stochastic gravitational-wave background as a tool for investigating multi-channel astrophysical and primordial black-hole mergers`, Bavera et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...660A..26B/abstract)
- [`Probing the progenitors of spinning binary black-hole mergers with long gamma-ray bursts`, Bavera et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...657L...8B/abstract)
- [`The impact of mass-transfer physics on the observable properties of field binary black hole populations`, Bavera et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021A%26A...647A.153B/abstract)
- [`The Role of Core-collapse Physics in the Observability of Black Hole Neutron Star Mergers as Multimessenger Sources`, Román-Garza, Bavera et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...912L..23R/abstract)
- [`One Channel to Rule Them All? Constraining the Origins of Binary Black Holes Using Multiple Formation Pathways`, Zevin, Bavera et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021ApJ...910..152Z/abstract)
- [`The origin of spin in binary black holes. Predicting the distributions of the main observables of Advanced LIGO`, Bavera et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...635A..97B/abstract)

If you plan to use this code or the datasets, please cite the corresponding article. 

UPDATE May 2023: part of this code is now integrated in the POSYDON v2 development branch, see [PR 63](https://github.com/POSYDON-code/POSYDON/pull/63).

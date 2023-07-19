# -*- coding: utf-8 -*-
import os
import matplotlib.path as mpath
import numpy as np
import glob
import astropy.io.fits as fits
from astropy.table import Table
from astropy import units as u
from astropy.io.fits import getdata
import matplotlib.pyplot as plt
import logging
from pylab import *
import emcee
import corner


def IMF_(author):
    """Defines dictionary for Kroupa and Salpeter initial mass functions.

    The code below simulates stars using a Kroupa or Salpeter IMF,
    and an exponential radius for the 2D distribution of stars.

    Parameters
    ----------
    author : str
        The name of the initial mass function (IMF)

    Returns
    -------
    dictionary
        a dict with the alpha values and mass breaks
    """
    if author == "Kroupa":
        return {"IMF_alpha_1": -1.3, "IMF_alpha_2": -2.3, "IMF_mass_break": 0.5}
    if author == "Salpeter":
        return {"IMF_alpha_1": -2.3, "IMF_alpha_2": -2.3, "IMF_mass_break": 0.5}
    print('Please, the IMF type was not added to the list of IMF authors.')


def apply_err(mag, mag_table, err_table, factor_error):
    """This function returns magnitude errors for the 'mag' variable
    based on mag_table and err_table.

    Parameters
    ----------
    mag : list
        The list of magnitudes to be calculated
    mag_table : list
        List of magnitudes
    err_table : List
        List of magnitude errors (1-sigma) respective to mag_table

    Returns
    -------
    list
        a list of magnitude errors following Normal distribution with
        1-sigma error as informed
    """
    err_interp = np.interp(mag, mag_table, err_table)
    err_interp *= factor_error
    return np.multiply(err_interp, np.random.randn(len(err_interp)))



def faker_bin(total_bin, IMF_author, file_in, dist):
    """Calculates the fraction of binaries in the simulated clusters.

    Parameters
    ----------
    total_bin : float
        The amount of binaries. Definition: N is the total amount of
        stars (take care to count a system of a binary as two stars),
        and B is the amount of stars in binary systems.
        so bin_frac = B / N
    IMF_author : str
        Name of the IMF (see function above)
    file_in : str
        The name of the file with star's masses and magnitudes
    dist : float
        Distance cluster-observer in parsecs

    Returns
    -------
    binaries[:,0]
        a list of magnitudes of the binaries in the first band
    binaries[:,1]
        a list of magnitudes of the binaries in the second band
    """

    mass, int_IMF, mag1, mag2 = np.loadtxt(file_in, usecols=(3, 4, 29, 30), unpack=True)

    # bin in mass (solar masses)
    binmass = 5.0e-4

    mag1 += 5 * np.log10(dist) - 5
    mag2 += 5 * np.log10(dist) - 5

    IMF = IMF_(IMF_author)

    # amostra is an array with the amount of stars in each bin of mass. ex.: [2,3,4,1,2]
    massmin = np.min(mass)
    massmax = np.max(mass)
    bins_mass = int((massmax - massmin) / binmass)
    amostra = np.zeros(bins_mass)

    for i in range(bins_mass):
        if (i * binmass) + massmin <= IMF["IMF_mass_break"]:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_1"]))
        else:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_2"]))
    # Soma is the total amount of stars (float), the sum of amostra
    soma = np.sum(amostra)
    # Now normalizing the array amostra
    # for idx, num in enumerate(amostra):
    amostra = np.multiply(amostra, total_bin / soma)

    massa_calculada = np.zeros(int(total_bin))

    count = 0

    for j in range(bins_mass):  # todos os intervalos primarios de massa
        for k in range(
            int(amostra[j])
        ):  # amostra() eh a amostra de estrelas dentro do intervalo de massa
            massa_calculada[count] = (
                massmin + (j * binmass) + (k * binmass / amostra[j])
            )
            # massa calculada eh a massa de cada estrela
            count += 1

    # mag1 mag1err unc1 mag2 mag2err unc2
    binaries = np.zeros((total_bin, 3))

    for i in range(total_bin):
        for k in range(len(mass) - 1):  # abre as linhas do arquivo em massa
            # se a massa estiver no intervalo das linhas
            if (mass[k] < massa_calculada[i]) & (mass[k + 1] > massa_calculada[i]):
                # vai abrir tantas vezes quantas forem as estrelas representadas
                intervalo = (massa_calculada[i] - mass[k]) / (
                    mass[k + 1] - mass[k]
                )  # intervalo entre zero e um
                binaries[i, 0] = mag1[k] - (mag1[k] - mag1[k + 1]) * intervalo
                binaries[i, 1] = mag2[k] - (mag2[k] - mag2[k + 1]) * intervalo
                binaries[i, 2] = massa_calculada[i]
    return binaries[:, 0], binaries[:, 1], binaries[:, 2]


def unc(mag, mag_table, err_table):
    """Interpolates the uncertainty in magnitude for a specific magnitude
    using magnitude and error from table.

    Parameters
    ----------
    mag : float or list
        The magnitude to be interpolated
    mag_table : list
        List of magnitudes in table
    err_table : list
        List of magnitude errors in table

    Returns
    -------
    err_interp : float or list
        Magnitudes interpolated
    """

    err_interp = np.interp(mag, mag_table, err_table)
    return err_interp


def faker(N_stars_cmd, frac_bin, IMF_author,
          dist, cmin, cmax, mmin, mmax, mag1_, err1_, err2_, age, FeH, c_steps, m_steps, factor_error_g, factor_error_r):
    """Creates an array with positions, magnitudes, magnitude errors and magnitude
    uncertainties for the simulated stars in two bands.

    The stars belong to a simple
    stellar population and they are spatially distributed following an exponential profile.
    The code firstly simulates the stars in the CMDs and finally simulates only the
    companions of the binaries (it does not matter how massive the companions are) to
    join to the number of points in the CMD.
    Bear in mind these quantities (definitions):
    N_stars_cmd = number of stars seen in the CMD. The binaries are seen as a single star.
    N_stars_single = amount of stars that are single stars.
    N_stars_bin = amount of stars that are binaries in the CMD. For each of these kind of
    stars, a companion should be calculated later.
    Completeness function is a step function modified where the params are a magnitude of
    reference and its completeness, and the completeness at maximum magnitude. The complete-
    ness function is equal to unity to magnitudes brighter than reference magnitude, and
    decreases linearly to the maximum magnitude.

    Parameters
    ----------
    N_stars_cmd : int
        Points in simulated cmd given the limiting magnitude. Some of this stars are single,
        some of the are binaries. This amount obeys the following relation:
        N_stars_cmd = N_stars_single + N_stars_bin, where N_stars_single are the single stars
        in the cmd and N_stars_bin are the points in CMD that are binaries. A single star in
        each system are accounted for. In this case, the total amount of stars simulated is
        N_stars_single + 2 * N_stars_bin
    frac_bin : float (0-1)
        Fraction of binaries. This is the total amount of stars in the CMD that belongs to a
        binary system (= 2 * N_stars_bin / total amount of stars).
    IMF_author : str
        Name of the IMF (see function above)
    x0 : float (degrees)
        RA position of the center of cluster
    y0 : float(degrees)
        DEC position of the center of cluster
    rexp : float (degrees)
        Exponential radii of the cluster following the exponential law of density:
        N = A * exp(-r/rexp)
    ell_ : float
        Ellipticity of the cluster (ell_=sqrt((a^2-b^2)/(a^2)))
    pa : float
        Positional angle (from North to East), in degrees
    dist : float
        Distance to the cluster in parsecs
    hpx : int
        Pixel where the cluster resides (nested)
    cmin : float
        Minimum color.
    cmax : float
        Maximum color.
    mmin : float
        Minimum magnitude.
    mmax : float
        Maximum magnitude.
    mag1_ : list
        List of magnitudes acoording error.
    err1_ : list
        List of errors in bluer magnitude.
    err2_ : list
        List of errors in redder magnitude.
    file_iso : str
        Name of file with data from isochrone.
    output_path : str
        Folder where the files will be written.
    mag_ref_comp : float
        Magnitude of reference where the completeness is equal to comp_mag_ref.
    comp_mag_ref : float
        Completeness (usually unity) at mag_ref_comp.
    comp_mag_max : float
        Completeness (value between 0. and 1.) at the maximum magnitude.

    """
    file_iso = 'bank/age_{:.2f}_Gyr_MH_{:.2f}.dat'.format(age, FeH)
    mass, int_IMF, mag1, mag2 = np.loadtxt(file_iso, usecols=(3, 4, 29, 30), unpack=True)

    # bin in mass (solar masses)
    binmass = 5.0e-4

    mag1 += 5 * np.log10(dist) - 5
    mag2 += 5 * np.log10(dist) - 5

    # Warning: cut in mass to avoid faint stars with high errors showing up in the
    # bright part of magnitude. The mass is not the total mass of the cluster,
    # only a lower limit for the total mass.
    cond = mag1 <= mmax + 0.2
    mass, mag1, mag2, int_IMF = mass[cond], mag1[cond], mag2[cond], int_IMF[cond]

    # amostra is an array with the amount of stars in each bin of mass. ex.: [2,3,4,1,2]
    massmin = np.min(mass)
    massmax = np.max(mass)
    bins_mass = int((massmax - massmin) / binmass)
    n_stars = int_IMF
    amostra = np.zeros(bins_mass)

    IMF = IMF_(IMF_author)

    for i in range(bins_mass):
        if (i * binmass) + massmin <= IMF["IMF_mass_break"]:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_1"]))
        else:
            amostra[i] = round((massmin + i * binmass) ** (IMF["IMF_alpha_2"]))
    # Soma is the total amount of stars (float), the sum of amostra
    soma = np.sum(amostra)

    # Now normalizing the array amostra
    for i in range(len(amostra)):
        amostra[i] = N_stars_cmd * amostra[i] / soma
    massa_calculada = np.zeros((N_stars_cmd))

    count = 0

    if np.sum([int(i) for i in amostra]) > 0:
        for j in range(bins_mass):  # todos os intervalos primarios de massa
            for k in range(int(amostra[j])
            ):  # amostra() eh a amostra de estrelas dentro do intervalo de massa
                massa_calculada[count] = (
                    massmin + (j * binmass) + (k * binmass / amostra[j])
                )
                # massa calculada eh a massa de cada estrela
                count += 1

        # 0-RA, 1-DEC, 2-mag1, 3-mag1err, 4-unc1, 5-mag2, 6-mag2err, 7-unc2, 8-mass
        star = np.zeros((N_stars_cmd, 9))

        for i in range(N_stars_cmd):
            for k in range(len(mass)-1):  # abre as linhas do arquivo em massa
                # se a massa estiver no intervalo das linhas
                if (mass[k] < massa_calculada[i]) & (mass[k + 1] > massa_calculada[i]):
                    # vai abrir tantas vezes quantas forem as estrelas representadas
                    intervalo = (massa_calculada[i] - mass[k]) / (
                        mass[k + 1] - mass[k]
                    )  # intervalo entre zero e um
                    star[i, 2] = mag1[k] - (mag1[k] - mag1[k + 1]) * intervalo
                    star[i, 5] = mag2[k] - (mag2[k] - mag2[k + 1]) * intervalo
                    star[i, 8] = massa_calculada[i]

        # apply binarity
        # definition of binarity: fb = N_stars_in_binaries / N_total
        N_stars_bin = int(N_stars_cmd / ((2.0 / frac_bin) - 1))
        mag1_bin, mag2_bin, mass_bin = faker_bin(
            N_stars_bin, "Kroupa", file_iso, dist)

        j = np.random.randint(N_stars_cmd, size=N_stars_bin)
        k = np.random.randint(N_stars_bin, size=N_stars_bin)

        for j, k in zip(j, k):
            star[j, 2] = -2.5 * np.log10(
                10.0 ** (-0.4 * star[j, 2]) + 10.0 ** (-0.4 * mag1_bin[k])
            )
            star[j, 5] = -2.5 * np.log10(
                10.0 ** (-0.4 * star[j, 5]) + 10.0 ** (-0.4 * mag2_bin[k])
            )

        # print(star[:,8])
        star[:, 3] = apply_err(star[:, 2], mag1_, err1_, factor_error_g)
        star[:, 6] = apply_err(star[:, 5], mag1_, err2_, factor_error_r)

        star[:, 4] = unc(star[:, 2], mag1_, err1_)
        star[:, 7] = unc(star[:, 5], mag1_, err2_)
    
        cor = star[:, 2] + star[:, 3] - (star[:, 5] + star[:, 6])
        mmag = star[:, 2] + star[:, 3]

        h1, xedges, yedges, im1 = plt.hist2d(cor, mmag, bins=(c_steps, m_steps), range=[[cmin, cmax], [mmin, mmax]])

        return h1
    else:
        return np.zeros((c_steps, m_steps))


def CSP(pars):

    distance, total_stars, binarity, peak_age, std_age, peak_FeH, std_FeH, factor_error_g, factor_error_r, comp_g, comp_r, comp_mag_g, comp_mag_r = pars

    global IMF_author, age_min, age_max, FeH_min, FeH_max, age_step, FeH_step, cmin, cmax, mmin, mmax, c_steps, m_steps, mag1_, err1_, err2_
    
    distance *= 1000
    
    total_stars *= 1e6
     
    mean = (peak_age, peak_FeH)
    cov = [[std_age, 0], [0, std_FeH]]
    x, y = np.random.multivariate_normal(mean, cov, int(total_stars)).T
    
    num_age = int((age_max-age_min)/age_step)+1
    num_FeH = int((FeH_max-FeH_min)/FeH_step)+1
    age_bins = np.linspace(age_min, age_max, num_age, endpoint=True)
    FeH_bins = np.linspace(FeH_min, FeH_max, num_FeH, endpoint=True)
    
    h1, xedges, yedges, im1 = plt.hist2d(x, y, bins=(age_bins, FeH_bins), range=[[cmin, cmax], [mmin, mmax]])
        
    main_CSP = np.zeros((c_steps, m_steps))

    for i, ii in enumerate(xedges[:-1]):
        for j, jj in enumerate(yedges[:-1]):
            if int(h1[i,j]) >= 1.:
                main_CSP =+ faker(int(h1[i,j]), binarity, IMF_author,
                       distance, cmin, cmax, mmin, mmax, mag1_, err1_, err2_, ii, jj, c_steps, m_steps, factor_error_g, factor_error_r)
    return main_CSP

def CSP_real_data(cmin, cmax, mmin, mmax, c_steps, m_steps, data_g, data_r):
    
    num_age = int((age_max-age_min)/age_step)+1
    num_FeH = int((FeH_max-FeH_min)/FeH_step)+1
    age_bins = np.linspace(age_min, age_max, num_age, endpoint=True)
    FeH_bins = np.linspace(FeH_min, FeH_max, num_FeH, endpoint=True)
    
    cor = data_g - data_r
    
    h1, xedges, yedges, im1 = plt.hist2d(cor, data_g, bins=(c_steps, m_steps), range=[[cmin, cmax], [mmin, mmax]])
        
    return h1


def ln_prior(theta):
    distance, total_stars, binarity, peak_age, std_age, peak_FeH, std_FeH, factor_error_g, factor_error_r, comp_g, comp_r, comp_mag_g, comp_mag_r = theta
    if 0.1 < distance < 300 and 0.01 < total_stars < 1000 and 0.0 < binarity < 1.0 and 0.0 < peak_age < 20.0 and 0.0 <  std_age < 3.0 and -10.0 <  peak_FeH < 1.0 and 0.0 <  std_FeH  < 1.0 and 0.0 < factor_error_g < 10.0 and 0.0 < factor_error_r < 10.0 and 0.0 < comp_g < 100.0 and 0.0 < comp_r < 100.0 and 0.0 < comp_mag_g < 100.0 and 0.0 < comp_mag_r < 100.0:
        return 0.0
    return -np.inf


def ln_like(theta):

    global CMD_real

    mod = CSP(theta)
    mod = np.ma.array(mod)
    obs = np.ma.array(CMD_real)
    valid = (mod > 0.) & (obs > 0.)
    l = (mod[valid] - obs[valid] + obs[valid] * np.log(obs[valid] / mod[valid])) * (2.0)
    if len(l) == 0:
        l = np.array([[np.inf, 0], [0, 0]])
        return l.sum()
    return l.sum()


def ln_prob(theta):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    val = ln_like(theta)
    print(val)
    return lp + val


cmin = -0.4
cmax = 1.5
mmin = 16.
mmax = 24.
c_steps = 50
m_steps = 50
IMF_author = 'Kroupa'
age_min = 10.
age_max = 13.
FeH_min = -2
FeH_max = -1
age_step = 0.2
FeH_step = 0.05

hdu = fits.open('/home/adriano/Dropbox/ga-wazpy/stack_cmd/Sculptor.fits', memmap=True)
g_obj = hdu[1].data.field('BDF_MAG_G_CORRECTED')
r_obj = hdu[1].data.field('BDF_MAG_R_CORRECTED')

CMD_real = CSP_real_data(cmin, cmax, mmin, mmax, c_steps, m_steps, g_obj, r_obj)

mag1_, err1_, err2_ = np.loadtxt('/home/adriano/ga_sim/surveys/des/errors.dat', usecols=(0,1,2), unpack=True)

ndim, nwalkers = 13, 30
# fitting pars: distance, total_stars, binarity, peak_age, std_age, peak_FeH, std_FeH, factor_error_g, factor_error_r, comp_g, comp_r, comp_mag_g, comp_mag_r
kick = [30, 0.5, 0.5, 12., 1., -1.5, 0.2, 1., 1., 0.9, 0.9, 23, 23]
pos = [kick + 1e-4*np.random.randn(ndim) for j in range(nwalkers)]

sampler = emcee.EnsembleSampler(
    nwalkers, ndim, ln_prob, args=())

sampler.run_mcmc(pos, 10)

fig, axes = plt.subplots(ndim, figsize=(30, 7), sharex=True)

samples = sampler.chain[:, 0:, :].reshape((-1, ndim))
labels = ["distance(kpc)", "total_M_stars", "binarity", "peak_age", "std_age", "peak_FeH", "std_FeH", "factor_error_g", "factor_error_r", "comp_g", "comp_r", "comp_mag_g", "comp_mag_r"]
'''
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
'''
distance, total_stars, binarity, peak_age, std_age, peak_FeH, std_FeH, factor_error_g, factor_error_r, comp_g, comp_r, comp_mag_g, comp_mag_r = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print(distance, total_stars, binarity, peak_age, std_age, peak_FeH, std_FeH, factor_error_g, factor_error_r, comp_g, comp_r, comp_mag_g, comp_mag_r)
# Plotting data
fig = corner.corner(samples, labels=labels, truths=[distance[0], total_stars[0], binarity[0], peak_age[0], std_age[0], peak_FeH[0], std_FeH[0], factor_error_g[0], factor_error_r[0], comp_g[0], comp_r[0], comp_mag_g[0], comp_mag_r[0]], quantiles=[0.16, 0.5, 0.84], show_titles=True, plot_contours=True)
plt.savefig('_plus.png')
plt.close()

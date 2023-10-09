import sys
sys.path.insert(0, '/home/rtodorov/ridge-calculation')
import os
from astropy.io import fits
import numpy as np
import scipy
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from image import plot as iplot
from plot_ridgeline import plot_sourse_map_w_ridge, shift_img
from build_ridgeline import get_average_ridgeline
from scipy.interpolate import UnivariateSpline, interp1d
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from utils import find_bbox, find_image_std, mas_to_rad, degree_to_rad


def find_extrema(slice_arr, std):
    # find extrema of array's interpolation
    slice_spl_pol = interp1d(np.arange(slice_arr.size), slice_arr)
    mins = []
    maxs = []
    
    flag_min = False
    flag_max = False
    max_value = np.max(slice_spl_pol(np.arange(slice_arr.size)))
    for x1 in np.arange(1, slice_arr.size - 2):
        if slice_spl_pol(x1 - 1) > slice_spl_pol(x1) < slice_spl_pol(x1 + 1):
            mins.append((x1, slice_spl_pol(x1)))
            flag_min = True
        if slice_spl_pol(x1 - 1) < slice_spl_pol(x1) > slice_spl_pol(x1 + 1):
            maxs.append((x1, slice_spl_pol(x1)))
            flag_max = True
        if flag_min == True and flag_max == True:
            if np.abs(maxs[-1][1] - mins[-1][1]) < max(20 * std, max_value / 10):
                maxs.pop()
                mins.pop()
                flag_min = False
                flag_max = False
    return mins, maxs


def generate_slices(image_data, ridge, spl, mapsize, pix_size, blc, trc, std, std_p):
    # maxlen_coord = ridge[0].flat[abs(ridge[1]).argmax()]
    maxlen_coord = np.max(ridge[0])
    minlen_coord = np.min(ridge[ridge > 0])
    ridge_x = ridge[0] * np.sin(ridge[1])
    ridge_y = ridge[0] * np.cos(ridge[1])

    rs = np.linspace(0, maxlen_coord, 1000)
    thetas = spl(rs)

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)

    linspace_size = 100
    ridge_space = np.linspace(minlen_coord, maxlen_coord, linspace_size)
    pix_to_mas_x = 2 / mapsize[0] * x_bound / mas_to_rad
    pix_to_mas_y = 2 / mapsize[1] * y_bound / mas_to_rad

    slice_arrs = []
    for r in ridge_space:
        # slope = np.tan(np.arctan(np.tan(spl(r)) - spl.derivative()(r)) + np.pi / 2)
        dr = maxlen_coord / 10000
        dy = ((r + dr) * np.cos(spl(r + dr)) - (r - dr) * np.cos(spl(r - dr)))
        dx = ((r + dr) * np.sin(spl(r + dr)) - (r - dr) * np.sin(spl(r - dr)))
        rigde_direction = np.arctan(dx / dy)
        slope = np.tan(rigde_direction + np.pi / 2)
        if np.abs(slope) < 1:
            delta = (blc[0] - trc[0]) * pix_to_mas_x / 2
            slice_x = np.linspace(r * np.cos(spl(r)) - delta, r * np.cos(spl(r)) + delta, linspace_size)
            slice_y = slope * (slice_x - r * np.cos(spl(r))) + r * np.sin(spl(r))
        else:
            delta = (blc[1] - trc[1]) * pix_to_mas_y / 2
            slice_y = np.linspace(r * np.sin(spl(r)) - delta, r * np.sin(spl(r)) + delta, linspace_size)
            slice_x = (slice_y - r * np.sin(spl(r))) / slope + r * np.cos(spl(r))

        slice_x_pix = np.array([round(-x / pix_to_mas_x + mapsize[0] / 2) for x in slice_x])
        slice_y_pix = np.array([round(y / pix_to_mas_y + mapsize[1] / 2) for y in slice_y])

        # fix_too_long slices
        x_mask = slice_x_pix < mapsize[0]
        x_mask[0 > slice_x_pix] = 0
        slice_y_pix = slice_y_pix[x_mask]
        slice_x = slice_x[x_mask]
        slice_y = slice_y[x_mask]
        slice_x_pix = slice_x_pix[x_mask]

        y_mask = slice_y_pix < mapsize[1]
        y_mask[0 > slice_y_pix] = 0
        slice_x_pix = slice_x_pix[y_mask]
        slice_x = slice_x[y_mask]
        slice_y = slice_y[y_mask]
        slice_y_pix = slice_y_pix[y_mask]
        if slice_x.size < 2:
            continue

        slice_values_i = np.array([image_data['i'][slice_y_pix[i]][slice_x_pix[i]]
                                     for i in np.arange(slice_y_pix.size)])
        slice_values_p = np.array([image_data['p'][slice_y_pix[i]][slice_x_pix[i]]
                                     for i in np.arange(slice_y_pix.size)])
        slice_values_chi = np.array([image_data['chi'][slice_y_pix[i]][slice_x_pix[i]]
                                     for i in np.arange(slice_y_pix.size)])
        slice_values_chi = slice_values_chi + rigde_direction + np.pi/2
        slice_values_chi[slice_values_chi > np.pi/2] -= np.pi
        slice_values_fpol = np.array([image_data['fpol'][slice_y_pix[i]][slice_x_pix[i]]
                                     for i in np.arange(slice_y_pix.size)])
        if image_data['std_EVPA'] is not None:
            slice_values_std_EVPA = np.array([image_data['std_EVPA'][slice_y_pix[i]][slice_x_pix[i]]
                                              for i in np.arange(slice_y_pix.size)])
        else:
            slice_values_std_EVPA = [None for _ in slice_x]

        slice_arr = np.array([slice_x, slice_y, slice_values_p, slice_values_chi, 
                              slice_values_std_EVPA, slice_values_fpol])
        slice_arr = slice_arr[:, slice_values_i > 20 * std]
        slice_arr = slice_arr[:, slice_arr[2] > 20 * std_p]
        # delete isolated points
        slice_arr_mask = np.full(slice_arr[0].size, True)
        for i in np.arange(slice_arr[2].size - 1):
            if np.hypot(slice_arr[0][i] - slice_arr[0][i + 1],
                        slice_arr[1][i] - slice_arr[1][i + 1]) > 4 * np.hypot(pix_to_mas_x, pix_to_mas_y):
                slice_arr_mask[i] = False
        slice_arr = slice_arr[:, slice_arr_mask]
        slice_arrs.append(slice_arr)
    return slice_arrs, ridge_space


def find_p_profie(slice_arr, std):
    mins, maxs = find_extrema(slice_arr, std)
    if len(maxs) >= 2 and len(mins) >= 1:
        return 'M-like'
    else:
        return 'A-like'


def resolve_fountain_chi_profile(slice_arr, slice_arr_p, std_p):
    mins_p, maxs_p = find_extrema(slice_arr_p, std_p)
    croses_0 = False
    if len(maxs_p) >= 2:
        mins, maxs = find_extrema(slice_arr[maxs_p[0][0]:maxs_p[-1][0]], 0)
        if slice_arr[maxs_p[0][0]:maxs_p[-1][0]].min() < 0 and \
                slice_arr[maxs_p[0][0]:maxs_p[-1][0]].max() > 0:
            croses_0  = True
    else:
        mins, maxs = find_extrema(slice_arr[slice_arr_p > 20 * std_p], 0)
        if slice_arr[slice_arr_p > 20 * std_p].min() < 0 and \
                slice_arr[slice_arr_p > 20 * std_p].max() > 0:
            croses_0  = True
    if len(mins) + len(maxs) > 0 or not croses_0:
        return False
    else:
        return True
        

def resolve_nonfountain_chi_profile(slice_arr):
    if abs(np.mean(abs(slice_arr))) < abs(np.pi/2 - np.mean(abs(slice_arr))):
        return 'parallel  '
    else:
        return 'orthogonal'


def find_chi_profie(slice_arr, slice_arr_p, std_p):
    if resolve_fountain_chi_profile(slice_arr, slice_arr_p, std_p):
        return 'foutain   '
    else:
        return resolve_nonfountain_chi_profile(slice_arr)
    # if slice_arr.max() - slice_arr.min() > np.pi/3:
    #     return     'foutain   '
    # else:
    #     if abs(np.mean(abs(slice_arr))) < abs(np.pi/2 - np.mean(abs(slice_arr))):
    #         return 'parallel  '
    #     else:
    #         return 'orthogonal'


def find_fpol_profie(slice_arr):
    mins, maxs = find_extrema(slice_arr, 0)
    if len(mins) == 2:
        return 'W-like'
    elif len(mins) == 1:
        return 'U-like'
    else:
        return 'flat  '


def find_std_profie(slice_arr, std, slice_arr_p, std_p):
    mins, maxs = find_extrema(slice_arr, std)
    mins_p, maxs_p = find_extrema(slice_arr_p, std_p)
    if len(maxs_p) >= 2 and len(mins_p) >= 1 and \
       len(maxs) == 1 and len(mins) == 0 and \
       not maxs[0][0] > maxs_p[-1][0] and \
       not maxs[0][0] < maxs_p[0][0]:
        return 'Yes'
    else:
        return 'No'


def return_profiles(slice_arr, std_p, std_std, pix_to_mas_x):
    p_profile = find_p_profie(slice_arr[2], std_p)
    fpol_profile = find_fpol_profie(slice_arr[5])
    std_profile = find_std_profie(slice_arr[4], std_std, slice_arr[2], std_p)
    chi_profile = find_chi_profie(slice_arr[3], slice_arr[2], std_p)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.plot(slice_arr[0], slice_arr[1], color='g')
    plot_sourse_map_w_ridge(source_name, data_folder, base_dir=base_dir, outfile="{}.jpg".format(source_name),  
                            contours_mode='I', colors_mode='fpol', vectors_mode='chi', fig=fig, ax=ax)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(8.5, 6))

    color = 'tab:red'
    ax1.set_ylabel('Polarized intensity (mJy/beam)', color=color, fontsize=20)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Transverse distance (mas)')
    ax1.plot(np.arange(slice_arr[2].size)*pix_to_mas_x, slice_arr[2] * 1000, color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('(Degrees)', color=color, fontsize=20)  # we already handled the x-label with ax1
    ax2.plot(np.arange(slice_arr[3].size)*pix_to_mas_x, slice_arr[3] / degree_to_rad,
             color=color, linestyle='dotted', label='EVPA')
    ax2.plot(np.arange(slice_arr[4].size)*pix_to_mas_x, slice_arr[4], color=color,
             linestyle='dashdot', label='STD EVPA')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.legend()         
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir+'/source_graphs', '{}_slice.jpg'.format(source_name)), bbox_inches='tight')
    plt.close()
    return fpol_profile, chi_profile, p_profile, std_profile


def detect_save_2pike_stdmax_profile(source_name, data_folder, base_dir):
    i_file = '{}/{}_stack_i_true.fits'.format(data_folder, source_name)
    p_file = '{}/{}_stack_p_true.fits'.format(data_folder, source_name)
    chi_file = '{}/{}_stack_p_ang.fits'.format(data_folder, source_name)
    image_data_i = fits.getdata(i_file, ext=0)[0][0]
    image_data_p = fits.getdata(p_file, ext=0)[0][0]
    image_data_chi = fits.getdata(chi_file, ext=0)[0][0] * degree_to_rad
    try:
        std_EVPA_file = '{}/std_evpa_fits/{}_std_EVPA_deg_unbiased_var_2.fits'.format(data_folder, source_name)
        image_data_std_EVPA = fits.getdata(std_EVPA_file, ext=0)
    except:
        print("STD EVPA is not defined!")
        image_data_std_EVPA = None

    # check file sanity
    if not (image_data_i.shape == image_data_p.shape == image_data_chi.shape):
        raise Exception('Files shape do not match')

    image_data_fpol = image_data_p / image_data_i

    hdul = fits.open(i_file)
    beam = [hdul[0].header['BMAJ'] * 3600000, hdul[0].header['BMIN'] * 3600000, hdul[0].header['BPA']]
    mapsize = [hdul[0].header['NAXIS1'], hdul[0].header['NAXIS2']]
    min_abs_level = 0.5 * np.abs(hdul[0].header['DATAMIN'])
    noise = hdul[0].header['NOISE']
    pix_size = [hdul[0].header['CDELT1'], hdul[0].header['CDELT2']]
    # print(beam)
    hdul.info()

    # fix nan in img
    image_data_i[np.isnan(image_data_i)] = 0
    image_data_p[np.isnan(image_data_p)] = 0
    image_data_fpol[np.isnan(image_data_fpol)] = 0
    image_data_chi[np.isnan(image_data_chi)] = 0

    image_data_fpol[image_data_fpol > 1] = 0
    image_data_fpol[image_data_fpol < 0] = 0

    image_data = {}
    image_data['i'] = image_data_i
    image_data['p'] = image_data_p
    image_data['fpol'] = image_data_fpol
    image_data['chi'] = image_data_chi
    image_data['std_EVPA'] = image_data_std_EVPA

    core = np.argmax(image_data_i)

    types = ['i', 'p', 'fpol', 'chi']
    for type_ in types:
        image_data[type_] = shift_img(image_data[type_], core, mapsize)

    npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
    std = find_image_std(image_data_i, beam_npixels=npixels_beam)
    std_p = find_image_std(image_data_p, beam_npixels=npixels_beam)
    std_std = find_image_std(image_data_std_EVPA, beam_npixels=npixels_beam)
    min_abs_level = 2 * std
    blc, trc = find_bbox(image_data_i, level=20 * std, min_maxintensity_mjyperbeam=40 * std,
                         min_area_pix=4 * npixels_beam, delta=10)
    if blc[0] == 0: blc = (blc[0] + 1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1] + 1)
    if trc[0] == image_data_i.shape: trc = (trc[0] - 1, trc[1])
    if trc[1] == image_data_i.shape: trc = (trc[0], trc[1] - 1)
    # blc = [400, 400]
    # trc = [624, 624]

    # plot_ridgeline(image_data, beam, mapsize, min_abs_level, std, pix_size)
    spl, ridge, raw_ridge = get_average_ridgeline(image_data_i, beam, mapsize, min_abs_level, std, pix_size)
    if spl is None:
        print('!!! Unable to build ridgeline !!!')
        return None

    label_size = 16
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size
    plt.rcParams['axes.titlesize'] = label_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['font.size'] = label_size
    plt.rcParams['legend.fontsize'] = label_size
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(round((blc[0] - trc[0]) / (blc[1] - trc[1])) * 6, 6 + 1))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r'Relative R.A. (mas)')
    ax.set_ylabel(r'Relative Decl. (mas)')
    # fig.set_size_inches(4.5, 3.5)

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)
    pix_to_mas_x = 2 / mapsize[0] * x_bound / mas_to_rad
    pix_to_mas_y = 2 / mapsize[1] * y_bound / mas_to_rad

    slice_arrs, ridge_space = generate_slices(image_data, ridge, spl, mapsize, pix_size, blc, trc, std, std_p)

    for slice_arr in slice_arrs:
        if slice_arr[2].size > 5:
            # smoothing slice_arr
            conv_core = np.array([1 / 10, 1 / 5, 2 / 5, 1 / 5, 1 / 10])
            slice_arr[2] = np.convolve(slice_arr[2], conv_core, 'same')
            slice_arr[3] = np.convolve(slice_arr[3], conv_core, 'same')
            if image_data_std_EVPA is not None:
                slice_arr[4] = np.convolve(slice_arr[4], conv_core, 'same')
            # weights = np.ones(slice_arr[2].size)
            # weights[0] = 10
            # weights[-1] = 10
            # slice_spl = UnivariateSpline(np.arange(slice_arr[2].size), slice_arr[2], w=weights)
            # slice_spl.set_smoothing_factor(0.1 * np.max(slice_arr[2]) ** 2)
            if image_data_std_EVPA is not None:
                slice_spl_stdEVPA = interp1d(np.arange(slice_arr[4].size), slice_arr[4])
            else:
                slice_spl_stdEVPA = None
            # plt.imshow(image_data)

            mins_p, maxs_p = find_extrema(slice_arr[2], std_p)
            if image_data_std_EVPA is not None:
                mins_stdEVPA, maxs_stdEVPA = find_extrema(slice_arr[4], std_std)
            else:
                mins_stdEVPA = None
                maxs_stdEVPA = None
                
            if len(mins_p) + len(maxs_p) > 2:
                print('Desired profile detected!')
                if len(maxs_stdEVPA) == 1 or image_data_std_EVPA is None:
                    print('STD EVPA has one max!')

                    fig, ax = plt.subplots(figsize=(8.5, 6))
                    ax.plot(slice_arr[0], slice_arr[1], color='g')
                    plot_sourse_map_w_ridge(source_name, data_folder, base_dir=base_dir, outfile="{}.jpg".format(source_name),  
                                            contours_mode='I', colors_mode='std', vectors_mode='n', fig=fig, ax=ax)
                    plt.close()

                    fig, ax1 = plt.subplots(figsize=(8.5, 6))

                    # ax1.plot(np.arange(slice_arr[2].size), slice_spl(np.arange(slice_arr[2].size)),
                    #          label='interpolated spline')
                    color = 'tab:red'
                    ax1.set_ylabel('Polarized intensity (mJy/beam)', color=color, fontsize=20)
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.set_xlabel('Transverse distance (mas)')
                    ax1.plot(np.arange(slice_arr[2].size)*pix_to_mas_x, slice_arr[2] * 1000, color=color)

                    ax2 = ax1.twinx()
                    color = 'tab:blue'
                    ax2.set_ylabel('(Degrees)', color=color, fontsize=20)  # we already handled the x-label with ax1
                    ax2.plot(np.arange(slice_arr[3].size)*pix_to_mas_x, slice_arr[3] / degree_to_rad,
                             color=color, linestyle='dotted', label='EVPA')
                    ax2.plot(np.arange(slice_arr[4].size)*pix_to_mas_x, slice_arr[4], color=color,
                             linestyle='dashdot', label='STD EVPA')
                    ax2.tick_params(axis='y', labelcolor=color)
                    plt.legend()
                    # plt.show()
                    plt.tight_layout()
                    plt.savefig(os.path.join(base_dir+'/source_graphs', '{}_slice.jpg'.format(source_name)), bbox_inches='tight')
                    plt.close()
                    return source_name
    plt.close()
    return None


def analize_profile(source_name, data_folder, base_dir):
    # determine transverse polarisation profile as in Puskarev et al. 2023
    i_file = '{}/{}_stack_i_true.fits'.format(data_folder, source_name)
    p_file = '{}/{}_stack_p_true.fits'.format(data_folder, source_name)
    chi_file = '{}/{}_stack_p_ang.fits'.format(data_folder, source_name)
    image_data_i = fits.getdata(i_file, ext=0)[0][0]
    image_data_p = fits.getdata(p_file, ext=0)[0][0]
    image_data_chi = fits.getdata(chi_file, ext=0)[0][0] * degree_to_rad
    try:
        std_EVPA_file = '{}/std_evpa_fits/{}_std_EVPA_deg_unbiased_var_2.fits'.format(data_folder, source_name)
        image_data_std_EVPA = fits.getdata(std_EVPA_file, ext=0)
    except:
        print("STD EVPA is not defined!")
        image_data_std_EVPA = None

    # check file sanity
    if not (image_data_i.shape == image_data_p.shape == image_data_chi.shape):
        raise Exception('Files shape do not match')

    image_data_fpol = image_data_p / image_data_i

    hdul = fits.open(i_file)
    beam = [hdul[0].header['BMAJ'] * 3600000, hdul[0].header['BMIN'] * 3600000, hdul[0].header['BPA']]
    mapsize = [hdul[0].header['NAXIS1'], hdul[0].header['NAXIS2']]
    min_abs_level = 0.5 * np.abs(hdul[0].header['DATAMIN'])
    noise = hdul[0].header['NOISE']
    pix_size = [hdul[0].header['CDELT1'], hdul[0].header['CDELT2']]
    # print(beam)
    hdul.info()

    # fix nan in img
    image_data_i[np.isnan(image_data_i)] = 0
    image_data_p[np.isnan(image_data_p)] = 0
    image_data_fpol[np.isnan(image_data_fpol)] = 0
    image_data_chi[np.isnan(image_data_chi)] = 0

    image_data_fpol[image_data_fpol > 1] = 0
    image_data_fpol[image_data_fpol < 0] = 0

    image_data = {}
    image_data['i'] = image_data_i
    image_data['p'] = image_data_p
    image_data['fpol'] = image_data_fpol
    image_data['chi'] = image_data_chi
    image_data['std_EVPA'] = image_data_std_EVPA

    core = np.argmax(image_data_i)

    types = ['i', 'p', 'fpol', 'chi']
    for type_ in types:
        image_data[type_] = shift_img(image_data[type_], core, mapsize)

    npixels_beam = np.pi * beam[0] * beam[1] / (4 * np.log(2) * mapsize[1] ** 2)
    std = find_image_std(image_data_i, beam_npixels=npixels_beam)
    std_p = find_image_std(image_data_p, beam_npixels=npixels_beam)
    if image_data_std_EVPA is not None:
        std_std = find_image_std(image_data_std_EVPA, beam_npixels=npixels_beam)
    else:
        std_std = None

    min_abs_level = 2 * std
    blc, trc = find_bbox(image_data_i, level=20 * std, min_maxintensity_mjyperbeam=40 * std,
                         min_area_pix=4 * npixels_beam, delta=10)
    if blc[0] == 0: blc = (blc[0] + 1, blc[1])
    if blc[1] == 0: blc = (blc[0], blc[1] + 1)
    if trc[0] == image_data_i.shape: trc = (trc[0] - 1, trc[1])
    if trc[1] == image_data_i.shape: trc = (trc[0], trc[1] - 1)
    # blc = [400, 400]
    # trc = [624, 624]

    # plot_ridgeline(image_data, beam, mapsize, min_abs_level, std, pix_size)
    spl, ridge, raw_ridge = get_average_ridgeline(image_data_i, beam, mapsize, min_abs_level, std, pix_size)
    if spl is None:
        print('!!! Unable to build ridgeline !!!')
        return None

    label_size = 16
    plt.rcParams['xtick.labelsize'] = label_size
    plt.rcParams['ytick.labelsize'] = label_size
    plt.rcParams['axes.titlesize'] = label_size
    plt.rcParams['axes.labelsize'] = label_size
    plt.rcParams['font.size'] = label_size
    plt.rcParams['legend.fontsize'] = label_size
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(round((blc[0] - trc[0]) / (blc[1] - trc[1])) * 6, 6 + 1))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.set_xlabel(r'Relative R.A. (mas)')
    ax.set_ylabel(r'Relative Decl. (mas)')
    # fig.set_size_inches(4.5, 3.5)

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)
    pix_to_mas_x = 2 / mapsize[0] * x_bound / mas_to_rad
    pix_to_mas_y = 2 / mapsize[1] * y_bound / mas_to_rad

    slice_arrs, ridge_space = generate_slices(image_data, ridge, spl, mapsize, pix_size, blc, trc, std, std_p)

    for slice_arr in slice_arrs:
        if slice_arr[2].size > 5:
            # smoothing slice_arr
            conv_core = np.array([1 / 10, 1 / 5, 2 / 5, 1 / 5, 1 / 10])
            slice_arr[2] = np.convolve(slice_arr[2], conv_core, 'same')
            slice_arr[3] = np.convolve(slice_arr[3], conv_core, 'same')
            if image_data_std_EVPA is not None:
                slice_arr[4] = np.convolve(slice_arr[4], conv_core, 'same')
            slice_arr[5] = np.convolve(slice_arr[5], conv_core, 'same')
            
            if find_p_profie(slice_arr[2], std_p) == 'M-like':
                    print(resolve_fountain_chi_profile(slice_arr[3], slice_arr[2], std_p))
                    return return_profiles(slice_arr, std_p, std_std, pix_to_mas_x)
            
    slice_arr = slice_arrs[int(len(slice_arrs)/3)]
    if slice_arr[2].size > 5:
        return return_profiles(slice_arr, std_p, std_std, pix_to_mas_x)
    else:
        fr = 4
        while slice_arr[2].size <= 5 and fr < 20:
            slice_arr = slice_arrs[int(len(slice_arrs)/fr)]
            fr += 1
        if fr == 20:
            return 'No data', 'No data', 'No data', 'No data'
        else:
            return return_profiles(slice_arr, std_p, std_std, pix_to_mas_x)


if __name__ == "__main__":
    data_folder = '/home/rtodorov/data_stack_fits'
    base_dir = '/home/rtodorov/transverse-profile-analysis'

    with open(os.path.join(data_folder, "source_list.txt")) as f:
        lines = f.readlines()

    f = open('./profiles-table.txt', 'w')
    f.write('Here is tranverse polarisation profiles table\n')
    f.write('sourse, m-profile, EVPA-profile, p-profile, if std EVPA distribution confirms precession model\n')
    for line in lines:
        source_name = line[0:8]
        print()
        print('########################################')
        print('########## Analyzing {} ##########'.format(source_name))
        print('########################################')
        fpol_profile, chi_profile, p_profile, std_profile = analize_profile(source_name, data_folder, base_dir)
        if p_profile == 'M-like':
            f.write('{}   {}   {}   {}   {}\n'.format(source_name, fpol_profile, chi_profile, p_profile, std_profile))
        else:
            f.write('{}   {}   {}   {}   {}\n'.format(source_name, fpol_profile, chi_profile, p_profile, '-'))

    '''
    with open('./source_list.txt') as f:
        lines = f.readlines()

    good_lines = []
    for line in lines:
        source_name = line[0:8]
        plot_std_EVPA_map(source_name)
    
    with open(os.path.join(data_folder, "source_list.txt")) as f:
        lines = f.readlines()

    good_lines = []
    for line in lines:
        source_name = line[0:8]
        print()
        print('########################################')
        print('########## Analyzing {} ##########'.format(source_name))
        print('########################################')
        good_lines.append(detect_save_2pike_stdmax_profile(source_name, data_folder, base_dir))

    f = open('./interesting_pol_struct.txt', 'w')
    for ind in good_lines:
        if ind is not None:
            f.write('{}\n'.format(ind))


    source_name = '1641+399'
    detect_save_profile(source_name, contours_mode='P', colors_mode='тщту', vectors_mode='chi')
    plot_any_map_w_ridge(source_name, contours_mode='I', colors_mode='std', vectors_mode='n')


    image_data = fits.getdata('0814+425_stack_p_ang.fits', ext=0)[0][0]
    image_data = np.flip(image_data)
    plt.imshow(image_data)
    # plt.show()
    '''

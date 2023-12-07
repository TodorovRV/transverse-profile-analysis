import numpy as np
import os
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
from profiles import generate_slices
sys.path.insert(0, '/home/rtodorov/jetpol/ve/vlbi_errors')
from utils import find_bbox, find_image_std, mas_to_rad, degree_to_rad
sys.path.insert(0, '/home/rtodorov/ridge-calculation')
from build_ridgeline import get_average_ridgeline
from plot_ridgeline import plot_sourse_map_w_ridge, shift_img


def fix_str(a):
    if len(str(a)) < 4:
        a = str(a)+'0'
    if len(str(a)) > 4:
        a = str(a)[:-1]
    return a


def analyze_richness(source_name, stack_data_folder, std_data_folder, base_dir):
    i_file = '{}/{}_stack_i_true.fits'.format(stack_data_folder, source_name)
    p_file = '{}/{}_stack_p_true.fits'.format(stack_data_folder, source_name)
    chi_file = '{}/{}_stack_p_ang.fits'.format(stack_data_folder, source_name)
    image_data_i = fits.getdata(i_file, ext=0)[0][0]
    image_data_p = fits.getdata(p_file, ext=0)[0][0]
    image_data_chi = fits.getdata(chi_file, ext=0)[0][0] * degree_to_rad
    try:
        std_EVPA_file = '{}/{}_std_EVPA_deg_unbiased_var_2.fits'.format(std_data_folder, source_name)
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

    x_bound = np.abs(mapsize[0] * pix_size[0] / 57.3 / 2)
    y_bound = np.abs(mapsize[1] * pix_size[1] / 57.3 / 2)
    pix_to_mas_x = 2 / mapsize[0] * x_bound / mas_to_rad
    pix_to_mas_y = 2 / mapsize[1] * y_bound / mas_to_rad

    slice_arrs, ridge_space = generate_slices(image_data, ridge, spl, mapsize, pix_size, blc, trc, std, std_p, 
                                                  significant_values_i=3*std, significant_values_p=0)
    
    richness_ = 0 
    slice_arr_ = None
    for slice_arr in slice_arrs:
            richness = np.sum(slice_arr[2] > 3*std)/slice_arr[2].size
            if richness_ < richness:
                richness_ = richness
                slice_arr_ = slice_arr

    if slice_arr_ is None:
        return ('----', '----', round(beam[0], 2), '0.0 ', '0.00')

    if np.sum(slice_arr_[2] > 3*std) > 1:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.plot(slice_arr_[0], slice_arr_[1], color='g')
        # ax.scatter(slice_arr_[0], slice_arr_[1], color='r')
        plot_sourse_map_w_ridge(source_name, stack_data_folder, std_data_folder=std_data_folder, base_dir=base_dir, 
                                outfile="{}.jpg".format(source_name), outdir='./sourse_graphs_richness', contours_mode='P', colors_mode='fpol', 
                                vectors_mode='chi', fig=fig, ax=ax)
        plt.close()

        fig, ax1 = plt.subplots(figsize=(8.5, 6))

        color = 'tab:red'
        ax1.set_ylabel('Polarized intensity (mJy/beam)', color=color, fontsize=20)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xlabel('Transverse distance (mas)')
        slice_len_mas = np.hypot(slice_arr_[0, 0]-slice_arr_[0, -1], slice_arr_[1, 0]-slice_arr_[1, -1])
        linsp = np.linspace(0, slice_len_mas, slice_arr_[2].size)
        ax1.plot(linsp, slice_arr_[2] * 1000, color=color)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('(Degrees)', color=color, fontsize=20)  # we already handled the x-label with ax1
        ax2.plot(linsp, slice_arr_[3] / degree_to_rad,
                color=color, linestyle='dotted', label='EVPA')
        ax2.plot(linsp, slice_arr_[4], color=color,
                linestyle='dashdot', label='STD EVPA')
        ax2.tick_params(axis='y', labelcolor=color)
        plt.legend()         
        plt.tight_layout()
        plt.savefig("./sourse_graphs_richness/{}_slice.jpg".format(source_name), bbox_inches='tight')
        plt.close()

        if beam[0] != beam[1]:
            raise Exception('Beam isnt round, do not support it!')

        return (round(slice_len_mas, 2), round(slice_len_mas/beam[0], 2), round(beam[0], 2), 
                round(richness_*100, 1), round(slice_len_mas*richness_/beam[0], 2))
    else:
        return ('----', '----', round(beam[0], 2), '0.0 ', '0.00')


if __name__ == "__main__":
    stack_data_folder = '/home/rtodorov/vlbi_data/data_stack_fits'
    std_data_folder = '/home/rtodorov/vlbi_data/std_evpa_fits'
    base_dir = '/home/rtodorov/transverse-profile-analysis'

    with open(os.path.join(stack_data_folder, "source_list.txt")) as f:
        lines = f.readlines()

    f = open('./richness-table.txt', 'w')
    f.write('sourse      slice_len(mas) slice_len(beam) beam(mas) polarisation(%) polarisation(beam)\n')
    for line in lines:
        source_name = line[0:8]
        print()
        print('########################################')
        print('########## Analyzing {} ##########'.format(source_name))
        print('########################################')
        a, b, c, d, e = analyze_richness(source_name, stack_data_folder, std_data_folder, base_dir)
        a, b, c, d, e = map(fix_str, (a, b, c, d, e))
        f.write('{}    {}           {}            {}      {}            {}\n'.format(source_name, a, b, c, d, e))
    f.close()
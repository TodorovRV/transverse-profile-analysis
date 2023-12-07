import numpy as np
import os
from scipy.stats import circstd
import matplotlib.pyplot  as plt
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version


def get_std_pa_data(data_dir):
    with open(os.path.join(data_dir, "source_list.txt")) as f:
        lines = f.readlines()

    data_dict = {}

    for line in lines:
        line = line[:-2]
        feature_id, pos_offset, pos_angle = np.genfromtxt(os.path.join(data_dir, '{}.csv'.format(line)), skip_header=1,
                                                        usecols=(2, 3, 4), unpack=True, delimiter=',')
        
        idx = feature_id != 0
        feature_id = feature_id[idx]
        pos_offset = pos_offset[idx]
        pos_angle = pos_angle[idx]

        pos_angle[pos_angle > 360] -= 360
        pos_angle[pos_angle < 0] += 360

        std = circstd(pos_angle, high=360)

        data_dict[line] = std

    return data_dict


def get_sourse_number_data(data_file, is_header=False):
    with open(data_file) as f:
        lines = f.readlines()

    data_dict = {}

    if is_header:
        for line in lines[1:]:
            line = line.split()
            data_dict[line[0]] = float(line[1])
    else:
        for line in lines:
            line = line.split()
            data_dict[line[0]] = float(line[1])

    return data_dict


def get_profile_data(data_file, profile):
    with open(data_file) as f:
        lines = f.readlines()

    head = lines[0]
    head = head.split()
    if profile in head:
        idx = head.index(profile)
    else:
        raise Exception('Wrong type of profile!')
    
    data_dict = {}

    for line in lines[1:]:
        line = line.split()
        if line[idx] in data_dict:
            data_dict[line[idx]].append(line[0])
        else:
            data_dict[line[idx]] = [line[0]]
    
    return data_dict


def get_table_data(data_file, data_dir='./', columns='All', is_header=True, convert_to_float=False):
    with open(os.path.join(data_dir, data_file)) as f:
        lines = f.readlines()

    if is_header:
        keys = lines[0].split()
    else:
        keys = np.arange(len(lines[0]))

    if columns == 'All':
        columns = keys
    else:
        for key in columns:
            if not key in keys:
                raise Exception("Key is absent in data!")

    if is_header:
        data_dict = {}
    else: 
        data_dict = []

    for i, key in enumerate(keys):
        if key in columns:
            data_dict[key] = []
            if is_header:
                for line in lines[1:]:
                    line = line.split()
                    if convert_to_float:
                        data_dict[key].append(float(line[i]))
                    else:
                        data_dict[key].append(line[i])
            else:
                for line in lines:
                    line = line.split()
                    if convert_to_float:
                        data_dict[key].append(float(line[i]))
                    else:
                        data_dict[key].append(line[i])
    
    return data_dict


def plot_data_diagramm(x_data, y_data, z_data, x_label=None, y_label=None, plotrange=None, gridsize=None, 
                       outfile=None, plot_trendline=False):
    if plotrange is None:
        plotrange = [0, 60, 0, 40]
    if gridsize is None:
        gridsize = [5, 5]

    for key in z_data:
        sources = z_data[key]
        ax_x = np.linspace(0, plotrange[1], int(plotrange[1]/gridsize[0]))
        ax_y = np.linspace(0, plotrange[3], int(plotrange[3]/gridsize[1]))
        data_arr = np.zeros((ax_y.size, ax_x.size))
        data_list = [[],[]]
        for source in sources:
            try:
                data_arr[int(y_data[source]/gridsize[1]), int(x_data[source]/gridsize[0])] += 1
                data_list[0].append(x_data[source])
                data_list[1].append(y_data[source])
            except:
                pass
        data_arr = np.flip(data_arr, axis=0)
        plt.imshow(data_arr, aspect='auto', extent=plotrange)
        plt.title(key)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if plot_trendline:
            y_trend = []
            #for column in np.transpose(data_arr):
            #    y_trend.append(np.sum(ax_y*np.flip(column))/np.sum(column))
            solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
            qr = QuantileRegressor(quantile=0.5, alpha=0, solver=solver)
            data_list =np.array(data_list)
            x = data_list[0]
            X = x[:, np.newaxis]
            print(data_list[1])
            y_pred = qr.fit(X, data_list[1]).predict(X)
            # y_trend_fitted = np.polyfit(ax_x, y_trend, 1, rcond=None, full=False, w=None, cov=False)
            plt.plot(X, y_pred, color='red')

        if outfile is None:
            plt.savefig('data_comp_{}.png'.format(key), bbox_inches='tight')
        else:
            plt.savefig(outfile, bbox_inches='tight')
        plt.close()


def plot_data_hist(z_data, x_data, outfname, x_label=None, y_label='Number', plotrange=None, gridsize=None, keys=None):
    if plotrange is None:
        plotrange = 60
    if gridsize is None:
        gridsize = 5

    if keys is None:
        keys = z_data

    for key in keys:
        sources = z_data[key]
        data_arr = []
        for source in sources:
            if source in x_data:
                data_arr.append(x_data[source])

        hist_bins = np.arange(0, plotrange + gridsize, gridsize)
        plt.hist(data_arr, bins=hist_bins, alpha=0.75, label=key)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.savefig(outfname, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    pa_data_dir ='/home/rtodorov/vlbi_data/csv_mojave'
    open_angle_data_file = '/home/rtodorov/vlbi_data/opening_angle_table.txt'
    profile_data_file = '/home/rtodorov/transverse-profile-analysis/profiles-table.txt'
    richness_data_file = 'richness-table.txt'
    z_data_file = '/home/rtodorov/vlbi_data/z_table.txt'
    open_angle_data_dict = get_sourse_number_data(open_angle_data_file)
    std_pa_data_dict = get_std_pa_data(pa_data_dir)
    profile_data_dict = get_profile_data(profile_data_file, 'EVPA-profile')
    plot_data_diagramm(open_angle_data_dict, std_pa_data_dict, profile_data_dict, x_label='open. angle, deg', y_label='std PA, deg')
    plot_data_hist(profile_data_dict, std_pa_data_dict, 'stdPA_hist', x_label='std PA, deg', 
                   keys=['parallel', 'orthogonal', 'fountain', 'bimodal'])
    plot_data_hist(profile_data_dict, open_angle_data_dict, 'open_angle_hist', x_label='open. angle, deg', 
                   keys=['parallel', 'orthogonal', 'fountain', 'bimodal'])
    richness_data_dict = get_table_data(richness_data_file, data_dir='./', columns=['polarisation(beam)'], is_header=True, convert_to_float=True)
    plt.hist(richness_data_dict['polarisation(beam)'], bins=20)
    plt.xlabel('Transverse polarised distance (beam)')
    plt.ylabel('Number')
    plt.savefig('richness.png', bbox_inches='tight')
    plt.close()
    z_data_dict = get_sourse_number_data(z_data_file, is_header=False)
    souses_data = {}
    souses_data[' '] = []
    for key in z_data_dict:
        souses_data[' '].append(key)
    plot_data_diagramm(z_data_dict, open_angle_data_dict, souses_data, x_label='z', plotrange=[0, 2, 0, 60], 
                       gridsize=[0.1, 5], y_label='open. angle, deg', outfile='z-op.angle.png', plot_trendline=True)
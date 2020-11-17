import numpy as np
import segyio
import matplotlib.pyplot as plt
import keras
import copy
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
import seaborn as sn


def generator(coords, batch_size, seis_arr_padded, cube_incr_x, cube_incr_y, cube_incr_z,
                             cube_step_interval, num_classes):
    batch_start = 0
    batch_end = batch_size
    print('num of train examples: ' + str(len(coords)))
    while True:
        limit = min(batch_end, len(coords))
        train_examples = np.empty((limit - batch_start, 2 * cube_incr_x + 1, 2 * cube_incr_y + 1, 2 * cube_incr_z + 1,
                                   seis_arr_padded.shape[3]), dtype=seis_arr_padded.dtype)
        for i, idx in enumerate(coords[batch_start:limit]):
            train_examples[i] = seis_arr_padded[idx[0] - cube_incr_x * cube_step_interval:idx[0] +
                                                            cube_step_interval * cube_incr_x + 1:cube_step_interval, \
                                        idx[1] - cube_step_interval * cube_incr_y:idx[1] +
                                                            cube_step_interval * cube_incr_y + 1:cube_step_interval, \
                                        idx[2] - cube_incr_z:idx[2] + cube_incr_z + 1, :]

        train_labels = coords[batch_start:limit][:, -1]
        train_labels = keras.utils.to_categorical(train_labels, num_classes)

        yield train_examples, train_labels

        batch_start += batch_size
        batch_end += batch_size

        if limit == len(coords):
            batch_start = 0
            batch_end = batch_size


def process_coords(points_coords, seis_spec, cube_incr_x, cube_incr_y, cube_incr_z, cube_step_interval, shuffle=True):
    coords_copy = points_coords
    if shuffle == True:
        np.random.shuffle(coords_copy)

    inline_start = seis_spec.inl_start
    inline_step = seis_spec.inl_step
    xline_start = seis_spec.xl_start
    xline_step = seis_spec.xl_step
    t_start = seis_spec.t_start
    t_step = seis_spec.t_step

    coords_copy_norm = np.array([(coords_copy[:, 0] - inline_start) // inline_step + cube_incr_x * cube_step_interval,
                                 (coords_copy[:, 1] - xline_start) // xline_step + cube_incr_y * cube_step_interval,
                                 (coords_copy[:, 2] - t_start) // t_step + cube_incr_z,
                                 coords_copy[:, 3]]).T

    return coords_copy_norm


def segy_read(segy_file, mode, scale=1, inp_cube=None, read_direc='xline', inp_res=np.float32):

    if mode == 'create':
        print('Starting SEG-Y decompressor')
        output = segyio.spec()

    elif mode == 'add':
        if inp_cube is None:
            raise ValueError('if mode is add inp_cube must be provided')
        print('Starting SEG-Y adder')
        cube_shape = inp_cube.shape
        data = np.empty(cube_shape[0:-1])

    else:
        raise ValueError('mode must be create or add')

    # open the segyfile and start decomposing it
    with segyio.open(segy_file, "r") as segyfile:
        segyfile.mmap()

        if mode == 'create':
            # Store some initial object attributes
            output.inl_start = segyfile.ilines[0]
            output.inl_end = segyfile.ilines[-1]
            output.inl_step = segyfile.ilines[1] - segyfile.ilines[0]

            output.xl_start = segyfile.xlines[0]
            output.xl_end = segyfile.xlines[-1]
            output.xl_step = segyfile.xlines[1] - segyfile.xlines[0]

            output.t_start = int(segyfile.samples[0])
            output.t_end = int(segyfile.samples[-1])
            output.t_step = int(segyfile.samples[1] - segyfile.samples[0])


            # Pre-allocate a numpy array that holds the SEGY-cube
            data = np.empty((segyfile.xline.length,segyfile.iline.length,\
                            (output.t_end - output.t_start)//output.t_step+1), dtype = np.float32)

        # Read the entire cube line by line in the desired direction
        if read_direc == 'inline':
            # Potentially time this to find the "fast" direction
            #start = time.time()
            for il_index in range(segyfile.xline.len):
                data[il_index,:,:] = segyfile.iline[segyfile.ilines[il_index]]
            #print(end - start)

        elif read_direc == 'xline':
            #start = time.time()
            for xl_index in range(segyfile.iline.len):
                data[:,xl_index,:] = segyfile.xline[segyfile.xlines[xl_index]]
            #end = time.time()
            #print(end - start)

        elif read_direc == 'full':
            #start = time.time()
            data = segyio.tools.cube(segy_file)
            #end = time.time()
            #print(end - start)
        else:
            print('Define reading direction(read_direc) using either ''inline'', ''xline'', or ''full''')

        factor = scale/np.amax(np.absolute(data))
        if inp_res == np.float32:
            data = (data*factor)
        else:
            data = (data*factor).astype(dtype = inp_res)

    if mode == 'create':
        output.data = data
        return output
    else:
        return output


def convert(file_list, facies_names):
    adr_list = np.empty([0, 4], dtype=np.int32)

    file_list_by_facie = []
    for facie in facies_names:
        facie_list = []
        for filename in file_list:
            if facie in filename:
                facie_list.append(filename)
        file_list_by_facie.append(facie_list)

    for i, files in enumerate(file_list_by_facie):
        for filename in files:
            a = np.loadtxt(filename, skiprows=0, usecols=range(3), dtype=np.int32)
            adr_list = np.append(adr_list, np.append(a, i*np.ones((len(a), 1), dtype=np.int32), axis=1), axis=0)

    return adr_list


def generate_coordinates(test_coords, segy_obj):
    seis_arr = segy_obj.data

    if test_coords[0][1] == test_coords[1][1]:
        xl = test_coords[0][1]
        t_step = segy_obj.t_step
        first_t = segy_obj.t_start // t_step
        n_il = seis_arr.shape[0]
        n_ts = seis_arr.shape[2]
        predict_coord = []
        for il in range(segy_obj.inl_start, segy_obj.inl_start + n_il):
            for ts in range(first_t, first_t + n_ts):
                predict_coord.append([il, xl, ts * t_step, 0])

    elif test_coords[0][0] == test_coords[1][0]:
        il = test_coords[0][0]
        t_step = segy_obj.t_step
        first_t = segy_obj.t_start // t_step
        n_xl = seis_arr.shape[1]
        n_ts = seis_arr.shape[2]
        predict_coord = []
        for xl in range(segy_obj.xl_start, segy_obj.xl_start + n_xl):
            for ts in range(first_t, first_t + n_ts):
                predict_coord.append([il, xl, ts * t_step, 0])

    else:
        raise ValueError('section_type must be inline or xline')

    return np.array(predict_coord)


def save_test_prediction(test_line_coord, test_line_coord_prep, predictions, cube_incr_x, cube_incr_y,
                         cube_incr_z, cube_step_interval, segy_filename, segy_obj, write_location):
    output_file = write_location + 'test_prediction.sgy'
    if test_line_coord[0][0] == test_line_coord[1][0]:
        mode = 'inline'
    elif test_line_coord[0][1] == test_line_coord[1][1]:
        mode = 'xline'
    else:
        raise ValueError("unrecognized format of test_line_coord")

    test_coords_prep_loc = copy.deepcopy(test_line_coord_prep)
    test_coords_prep_loc[:, 0] -= cube_incr_x * cube_step_interval
    test_coords_prep_loc[:, 1] -= cube_incr_y * cube_step_interval
    test_coords_prep_loc[:, 2] -= cube_incr_z * cube_step_interval
    ind = np.lexsort((test_line_coord[:, 2], test_line_coord[:, 1], test_line_coord[:, 0]))
    test_coords_prep_loc = test_coords_prep_loc[ind]
    predictions = predictions[ind]

    with segyio.open(segy_filename) as src:
        spec = segyio.spec()
        spec.sorting = src.sorting
        spec.format = src.format
        spec.samples = src.samples
        spec.ilines = np.unique(test_line_coord[:, 0])
        spec.xlines = np.unique(test_line_coord[:, 1])
        with segyio.create(output_file, spec) as dst:
            dst.text[0] = src.text[0]
            for i, iline in enumerate((spec.ilines - segy_obj.inl_start) // segy_obj.inl_step):
                for j, xline in enumerate((spec.xlines - segy_obj.xl_start) // segy_obj.xl_step):
                    if mode == 'xline':
                        tr_idx = i
                    elif mode == 'inline':
                        tr_idx = j
                    else:
                        raise ValueError("mode should be xine or inline")
                    dst.header[tr_idx].update(src.header[iline * len(src.xlines) + xline])
                    dst.trace[tr_idx] = np.ones(len(src.samples)) * 9
                    idx = np.all(np.array([test_coords_prep_loc[:, 0] == iline, test_coords_prep_loc[:, 1] == xline]),
                                 axis=0)
                    samples = test_coords_prep_loc[idx][:, 2]
                    trace_vals = dst.trace[tr_idx]
                    trace_vals[samples] = predictions[idx]
                    dst.trace[tr_idx] = trace_vals


def printout(write_location, y_test, pred_test, facies_list, history=None):
    facies_list = ['background'] + facies_list[1:]

    if not os.path.exists(write_location):
        os.makedirs(write_location)

    if history is not None:
        plt.figure()
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('batch #')
        ax1.set_ylabel('loss', color='blue')
        ax1.plot(np.array(history.batch_loss), color='blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('acc', color='red')
        ax2.plot(np.array(history.batch_acc), color='red')
        plt.savefig(write_location + 'batch_history.jpg')

        np.savetxt(write_location + 'history_batch_loss.txt', np.array(history.batch_loss))
        np.savetxt(write_location + 'history_batch_acc.txt', np.array(history.batch_acc))
        if history.val_acc[0] is not None:
            np.savetxt(write_location + 'history_val_acc.txt', np.array(history.val_acc))
            np.savetxt(write_location + 'history_val_loss.txt', np.array(history.val_loss))

    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(15, 15))
    cm = confusion_matrix(y_test.astype(int), pred_test.astype(int), normalize='true')

    df_cm = pd.DataFrame(cm, index=[facies_list[i] for i in list(np.arange(cm.shape[0]))],
                         columns=[facies_list[i] for i in list(np.arange(cm.shape[0]))])
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, fmt='.1%', cmap='Blues')
    plt.savefig(write_location + "confusion_matrix.jpg")

    enc = OneHotEncoder(sparse=False)
    categories = np.arange(0, len(facies_list)).reshape(-1, 1)
    enc.fit(categories)
    y_test = enc.transform(y_test.reshape(-1, 1))
    pred_test = enc.transform(pred_test.reshape(-1, 1))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if not os.path.exists(write_location + 'ROC_curves/'):
        os.makedirs(write_location + 'ROC_curves/')
    for i in range(pred_test.shape[1]):
        if np.sum(y_test[:, i]) == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.6f)' % roc_auc[i])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for a class %d' % i)
        plt.legend(loc="lower right")
        plt.savefig(write_location + 'ROC_curves/curve_%d' % i)
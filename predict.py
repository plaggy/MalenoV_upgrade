from utils import *
import time
import keras


def predict(predict_dict):
    model_path = predict_dict['model_path']
    segy_filename = predict_dict['segy_filename']
    facies_names = predict_dict['facies_names']
    inp_res = predict_dict['inp_res']
    test_files = predict_dict['test_files']
    save_location = predict_dict['save_location']
    batch_size = predict_dict['batch_size']
    cube_step_interval = predict_dict['cube_step_interval']

    model = keras.models.load_model(model_path)
    shape = model.input_shape
    cube_incr_x = shape[1] // 2
    cube_incr_y = shape[2] // 2
    cube_incr_z = shape[3] // 2
    test_coords = convert(test_files, facies_names)

    if type(segy_filename) is str or (type(segy_filename) is list and len(segy_filename) == 1):
        if type(segy_filename) is list:
            segy_filename = segy_filename[0]

        segy_obj = segy_read(segy_file=segy_filename, mode='create', read_direc='full', inp_res=inp_res, scale=127)

        segy_obj.cube_num = 1
        segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))


    elif type(segy_filename) is list:
        i = 0

        # iterate through the list of cube names and store them in a masterobject
        for filename in segy_filename:
            if i == 0:
                segy_obj = segy_read(segy_file=filename, mode='create', read_direc='full', inp_res=inp_res, scale=127)

                segy_obj.cube_num = len(segy_filename)

                print('Starting restructuring to 4D arrays')
                ovr_data = np.empty((list(segy_obj.data.shape) + [len(segy_filename)]))
                ovr_data[:, :, :, i] = segy_obj.data
                segy_obj.data = ovr_data
                ovr_data = None
                print('Finished restructuring to 4D arrays')
            else:
                # Add another cube to the numpy-array
                segy_obj.data[:, :, :, i] = segy_read(segy_file=filename, mode='add', inp_cube=segy_obj.data,
                                                      read_direc='full', inp_res=inp_res)
            i += 1
    else:
        print('The input filename needs to be a string or a list of strings')

    seis_arr = segy_obj.data
    # pad to preserve the cube size after applying convolution
    seis_arr_padded = np.empty((seis_arr.shape[0] + 2 * cube_incr_x * cube_step_interval,
                                seis_arr.shape[1] + 2 * cube_incr_y * cube_step_interval,
                                seis_arr.shape[2] + 2 * cube_incr_z, seis_arr.shape[-1]), dtype=inp_res)
    for i in range(segy_obj.data.shape[3]):
        seis_arr_padded[:, :, :, i] = np.pad(seis_arr[:, :, :, i], pad_width=((cube_incr_x * cube_step_interval,),
                                                                              (cube_incr_y * cube_step_interval,),
                                                                              (cube_incr_z,)),
                                             mode='constant', constant_values=0)

    test_line_coord = generate_coordinates(test_coords, segy_obj)

    test_line_coord_prep = process_coords(test_line_coord, segy_obj, cube_incr_x, cube_incr_y, cube_incr_z,
                                          cube_step_interval, shuffle=False)
    test_gen = generator(test_line_coord_prep, batch_size, seis_arr_padded, cube_incr_x,
                         cube_incr_y, cube_incr_z,
                         cube_step_interval, len(facies_names))

    start = time.time()
    predictions = model.predict_generator(test_gen, steps=np.ceil(len(test_line_coord_prep) / batch_size),
                                          verbose=1)
    predict_time = time.time() - start
    pred_sections = predictions.reshape(-1, seis_arr.shape[2], predictions.shape[1])

    if not os.path.exists(save_location):
        os.makedirs(save_location)

    for j in range(pred_sections.shape[-1]):
        plt.imsave(save_location + f'prob_{j}.jpg', np.transpose(pred_sections[..., j]), vmin=0, vmax=1)

    predictions = np.argmax(predictions, axis=-1)
    plt.imsave(save_location + f'prediction.jpeg', np.transpose(np.argmax(pred_sections, axis=-1)))

    save_test_prediction(test_line_coord, test_line_coord_prep, predictions, cube_incr_x, cube_incr_y,
                         cube_incr_z, cube_step_interval, segy_filename, segy_obj, save_location)

    test_coords = process_coords(test_coords, segy_obj, cube_incr_x, cube_incr_y, cube_incr_z, cube_step_interval,
                                 shuffle=False)
    test_gen = generator(test_coords, batch_size, seis_arr_padded, cube_incr_x, cube_incr_y, cube_incr_z,
                         cube_step_interval, len(facies_names))

    pred_test = model.predict_generator(test_gen, steps=np.ceil(len(test_coords) / batch_size), verbose=1)

    y_test = test_coords[:, -1]

    runtime_file = open(save_location + 'runtimes.txt', 'w+')
    runtime_file.write(f"predict time: {predict_time}")

    if len(pred_test) > 0:
        pred_test = np.rint(pred_test)
        pred_test = np.argmax(pred_test, axis=-1)
        acc = np.sum(pred_test == y_test) / y_test.shape[0]
        acc_file = open(save_location + 'acc.txt', 'w+')
        acc_file.write(f"{acc}\n")

        printout(save_location, y_test, pred_test, facies_names)
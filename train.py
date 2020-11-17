### Function for seismic facies prediction using Convolutional Neural Nets (CNN)
### By: Charles Rutherford Ildstad
### Modified by Iris Yang and Wei Chu 6.2019
### Modified by Sergei Petrov 11.2020

import matplotlib

matplotlib.use('Agg')
import time
from os.path import dirname, abspath
import json

from keras.models import Sequential
from keras.layers import Conv3D, Dense, Activation, Flatten, Dropout, MaxPooling3D
from keras.layers.normalization import BatchNormalization

from utils import *


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.batch_loss = []
        self.batch_acc = []
        self.val_acc = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.batch_loss.append(logs.get('loss'))
        self.batch_acc.append(logs.get('accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.val_acc.append(logs.get('val_accuracy'))
        self.val_loss.append(logs.get('val_loss'))

# Make the network structure and outline, and train it
def train_model(segy_obj, class_array, facies_names, cube_incr_x, cube_incr_y, cube_incr_z, cube_step_interval, num_epochs,
                batch_size, write_location, test_split, segy_filename,validation_split, acc_file, runtime_file,
                test_coords=None, num_channels=1, inp_res=np.float64):

    model = Sequential()

    model.add(Conv3D(8, (5, 5, 5),
                     padding='same',
                     input_shape=(2 * cube_incr_x + 1, 2 * cube_incr_y + 1, 2 * cube_incr_z + 1, num_channels),
                     strides=(1, 1, 1),
                     name='conv_layer1'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(2, 2, 2)))

    model.add(Conv3D(16, (5, 5, 5),
                     strides=(1, 1, 1),
                     padding='same',
                     name='conv_layer2'))
    model.add(Activation('relu'))
    model.add(Conv3D(32, (3, 3, 3),
                     strides=(1, 1, 1),
                     padding='same',
                     name='conv_layer3'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),
                           strides=(2, 2, 2)))

    model.add(Flatten())

    model.add(Dense(120, name='dense_layer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(84, name='attribute_layer'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(len(facies_names), name='pre-softmax_layer'))
    model.add(Activation('softmax'))

    # Compile the model with the desired loss, optimizer, and metric
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.adam(0.001),
                  metrics=['accuracy'])

    seis_arr = segy_obj.data
    seis_arr_padded = np.empty((seis_arr.shape[0] + 2 * cube_incr_x * cube_step_interval,
                                seis_arr.shape[1] + 2 * cube_incr_y * cube_step_interval,
                                seis_arr.shape[2] + 2 * cube_incr_z, num_channels), dtype=inp_res)
    for i in range(segy_obj.data.shape[3]):
        seis_arr_padded[:, :, :, i] = np.pad(seis_arr[:, :, :, i], pad_width=((cube_incr_x * cube_step_interval,),
                                             (cube_incr_y * cube_step_interval,), (cube_incr_z,)),
                                             mode='constant', constant_values=0)

    np.random.shuffle(class_array)
    train_val_coords = class_array
    train_val_coords_prep = process_coords(train_val_coords, segy_obj, cube_incr_x, cube_incr_y, cube_incr_z,
                                           cube_step_interval, shuffle=False)

    if test_coords is None:
        try:
            test_split = float(test_split)
        except:
            raise ValueError("test split should be float")
    else:
        test_split = 0

    num_train_examples = int(np.ceil(len(train_val_coords) * (1 - validation_split - test_split)))
    train_coords_prep = train_val_coords_prep[:num_train_examples]
    print(f"number of training points: {num_train_examples}")

    num_valid_examples = np.floor(len(train_val_coords) * validation_split)
    valid_coords_prep = train_val_coords_prep[num_train_examples:num_train_examples + num_valid_examples]

    if test_coords is None:
        num_test_examples = len(train_val_coords) - num_train_examples - num_valid_examples
        test_coords = train_val_coords[-num_test_examples:]

    history = LossHistory()

    train_gen = generator(train_coords_prep, batch_size, seis_arr_padded, cube_incr_x, cube_incr_y,
                                        cube_incr_z, cube_step_interval, len(facies_names))
    valid_gen = generator(valid_coords_prep, batch_size, seis_arr_padded, cube_incr_x, cube_incr_y,
                                         cube_incr_z, cube_step_interval, len(facies_names))

    labels = class_array[:, -1]
    unique, counts = np.unique(labels, return_counts=True)
    class_weight = {}
    for l, c in zip(unique, counts):
        class_weight[l] = len(labels) / (2 * c)

    start = time.time()
    model.fit_generator(train_gen,
                         steps_per_epoch=int(np.ceil(num_train_examples / batch_size)),
                         validation_data=valid_gen,
                         validation_steps=int(np.ceil(num_valid_examples / batch_size)),
                         callbacks=[history],
                         class_weight=class_weight,
                         epochs=num_epochs)

    train_time = time.time() - start
    print('Saving model')
    model.save(write_location + 'trained.h5')

    with open(write_location + 'subcube_params.txt', 'w+') as wfile:
        wfile.write(f"cube_incr_x: {cube_incr_x}\tcube_incr_y: {cube_incr_y}\t"
                    f"cube_incr_z: {cube_incr_z}")

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

    if not os.path.exists(write_location):
        os.makedirs(write_location)

    for j in range(predictions.shape[-1]):
        plt.imsave(write_location + f'prob_{j}.jpg', predictions[0, ..., j])

    predictions = np.argmax(predictions, axis=1)
    pred_section = predictions.reshape(-1, seis_arr.shape[2])

    plt.imsave(write_location + f'prediction.jpeg', pred_section)

    save_test_prediction(test_line_coord, test_line_coord_prep, predictions, cube_incr_x, cube_incr_y,
                         cube_incr_z, cube_step_interval, segy_filename, segy_obj, write_location)

    test_coords_prep = process_coords(test_coords, segy_obj, cube_incr_x, cube_incr_y, cube_incr_z, cube_step_interval, shuffle=False)
    test_gen = generator(test_coords_prep, batch_size, seis_arr_padded, cube_incr_x, cube_incr_y, cube_incr_z,
                                  cube_step_interval, len(facies_names))

    pred_test = model.predict_generator(test_gen, steps=np.ceil(len(test_coords) / batch_size), verbose=1)

    y_test = test_coords[:, -1]

    runtime_file.write(f"{train_time}\t{predict_time}\n")

    if len(pred_test) > 0:
        pred_test = np.rint(pred_test)
        pred_test = np.argmax(pred_test, axis=-1)
        acc = np.sum(pred_test == y_test)/y_test.shape[0]
        acc_file.write(f"{acc}\n")

        printout(write_location, y_test, pred_test, facies_names, history)

    return train_time, predict_time


### ---- MASTER/MAIN function ----
# Make an overall master function that takes inn some basic parameters,
# trains, predicts, and visualizes the results from a model
def train_wrapper(segy_filename, inp_res, train_dict):
    if type(segy_filename) is str or (type(segy_filename) is list and len(segy_filename) == 1):
        # Check if the filename needs to be retrieved from a list
        if type(segy_filename) is list:
            segy_filename = segy_filename[0]

        # Make a master segy object
        segy_obj = segy_read(segy_file = segy_filename,
                             mode='create',
                               plot_data = False,
                               read_direc = 'full',
                               inp_res = inp_res)

        # Define how many segy-cubes we're dealing with
        segy_obj.cube_num = 1
        segy_obj.data = np.expand_dims(segy_obj.data, axis=len(segy_obj.data.shape))


    elif type(segy_filename) is list:
        # start an iterator
        i = 0

        # iterate through the list of cube names and store them in a masterobject
        for filename in segy_filename:
            # Make a master segy object
            if i == 0:
                segy_obj = segy_read(segy_file = filename,
                                        mode='create',
                                       plot_data = False,
                                       read_direc = 'full',
                                       inp_res = inp_res)

                # Define how many segy-cubes we're dealing with
                segy_obj.cube_num = len(segy_filename)

                # Reshape and preallocate the numpy-array for the rest of the cubes
                print('Starting restructuring to 4D arrays')
                ovr_data = np.empty((list(segy_obj.data.shape) + [len(segy_filename)]))
                ovr_data[:,:,:,i] = segy_obj.data
                segy_obj.data = ovr_data
                ovr_data = None
                print('Finished restructuring to 4D arrays')
            else:
                # Add another cube to the numpy-array
                segy_obj.data[:,:,:,i] = segy_read(segy_file = filename,
                                                   mode='add',
                                                    inp_cube = segy_obj.data,
                                                    read_direc = 'full',
                                                    inp_res = inp_res)
            # Increase the itterator
            i+=1
    else:
        print('The input filename needs to be a string, or a list of strings')

    # Unpack the dictionary of training parameters
    label_list = train_dict['train_files']
    num_epochs = train_dict['epochs']
    batch_size = train_dict['batch_size']
    write_location = train_dict['save_location']
    test_split = train_dict['test_split']
    cube_step_interval = train_dict['subcube_step_interval']
    facies_names = train_dict['facies_names']
    test_files = train_dict['test_files']
    validation_split = train_dict['validation_split']
    cube_incr_x = train_dict['cube_incr_x']
    cube_incr_y = train_dict['cube_incr_y']
    cube_incr_z = train_dict['cube_incr_z']
    n_model_samples = train_dict['n_model_samples']

    print('num epochs:', num_epochs)
    print('batch size:', batch_size)

    cube_incr_x_sample = np.array(cube_incr_x)[np.random.randint(0, len(cube_incr_x), size=n_model_samples)]
    cube_incr_y_sample = np.array(cube_incr_y)[np.random.randint(0, len(cube_incr_y), size=n_model_samples)]
    cube_incr_z_sample = np.array(cube_incr_z)[np.random.randint(0, len(cube_incr_z), size=n_model_samples)]

    dir = dirname(dirname(abspath(__file__)))
    if not os.path.exists(dir + write_location):
        os.makedirs(dir + write_location)

    write_location = dir + write_location

    acc_file = open(write_location + 'acc.txt', 'w+')
    runtime_file = open(write_location + 'runtimes.txt', 'w+')
    runtime_file.write(f" \ttrain_time\tprediction_time\n")

    with open(write_location + "train_dict.json", "w+") as f:
        f.write(json.dumps(train_dict))

    for i in range(n_model_samples):
        # Make the list of class data
        print('Making class-adresses')
        class_array = convert(file_list=label_list,
                              facies_names=facies_names)

        print('Finished making class-adresses')

        if len(test_files) > 0:
            test_coords = convert(test_files, facies_names)
        else:
            test_coords = None

        write_location_sample = write_location + f"set_{i}/"
        if not os.path.exists(write_location_sample):
            os.makedirs(write_location_sample)

        # Time the training process
        start_train_time = time.time()

        # Train a new model/further train the uploaded model and store the result as the model output
        train_model(segy_obj=segy_obj,
                    class_array=class_array,
                    facies_names=facies_names,
                    cube_incr_x=cube_incr_x_sample[i],
                    cube_incr_y=cube_incr_y_sample[i],
                    cube_incr_z=cube_incr_z_sample[i],
                    cube_step_interval=cube_step_interval,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    write_location=write_location_sample,
                    test_split=test_split,
                    segy_filename=segy_filename,
                    validation_split=validation_split,
                    acc_file=acc_file,
                    runtime_file=runtime_file,
                    test_coords=test_coords,
                    num_channels=segy_obj.cube_num,
                    inp_res=inp_res)

        # Time the training process
        end_train_time = time.time()
        train_time = end_train_time-start_train_time # seconds

        # print to the user the total time spent training
        if train_time <= 300:
            print('Total time elapsed during training:',train_time, ' sec.')
        elif 300 < train_time <= 60*60:
            minutes = train_time//60
            seconds = (train_time%60)*(60/100)
            print('Total time elapsed during training:',minutes,' min., ',seconds,' sec.')
        elif 60*60 < train_time <= 60*60*24:
            hours = train_time//(60*60)
            minutes = (train_time%(60*60))*(1/60)*(60/100)
            print('Total time elapsed during training:',hours,' hrs., ',minutes,' min., ')
        else:
            days = train_time//(24*60*60)
            hours = (train_time%(24*60*60))*(1/60)*((1/60))*(24/100)
            print('Total time elapsed during training:',days,' days, ',hours,' hrs., ')

    acc_file.close()
    runtime_file.close()

    return

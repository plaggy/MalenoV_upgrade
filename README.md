### The application of 3D CNN with a fully connected head for seismic segmentation task.

The original implementation is here: https://github.com/bolgebrygg/MalenoV. Was improved by Yang I., Chu W., the details are here: https://cs230.stanford.edu/projects_spring_2018/reports/8291004.pdf.

This tool is essentially a relatively simple convolutional network with a fully connected head based on 3D convolutional layers. Training and prediction is performed point-by-point. The feature of this tool is that around each of the points considered a small 3D volume of data around it is taken into account (subcube). Each training example becomes a small 3D volume of data.

This implementation is focused on treating a subcube size as a hyperparameter and making sense of it.

#### The entire process is run from the main.py file, and all the main settings are defined there. The detailed description of parameters is given in the comments in the file itself.

To run a model with a particular set of parameters set **n_param_samples** to **1** and make  **cube_incr_x**, **cube_incr_y**, **cube_incr_z** to be lists containing a single value.

Training outputs are the following: txt files with accuracy, train and prediction runtimes, batch and validation accuracy and loss values, subcube sizes; images of a batch accuracy/loss history plot, confusion matrix, predicted section; predicted section in segy format; trained model (*.h5 file); a folder with ROC curve plots.

**main.py** – both training and prediction processes are run from this file, all the major settings are defined there  
**train.py** – contains a function that runs a training loop and saves results  
**utils.py** – contains all the miscellaneous functions to read segy and interpretation files, save results 
**predict.py** – contains a function that runs prediction and saves results

Here are some prediction examples obtained with different subcube sizes:
![](readme_images/malenov_predictions.png)



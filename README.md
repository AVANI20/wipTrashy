# TrashyWasteSorter

TRASHY is a general-purpose waste sorting appliance for private houses to detect all types of trash except biodegradable food waste,
and display the waste category and the respective bin color on a 7” Raspberry Pi screen. We have built the prototype TRASHY
on a Raspberry Pi 4 model B that captures images from a Raspberry camera and
classifies objects using an image-recognition pre-trained VGG-19 deep learning model.
TRASHY can sort the 25 pre-defined waste categories, namely glass, paper cups, metal
containers, newspaper, plastic cups, medical, e-waste, plastic bottles, and more, into
five colored bins such as blue, black, grey, yellow, and green. 
The dataset has 20146 images combined from two datasets - [TrashBox](https://github.com/nikhilvenkatkumsetty/TrashBox) and [Kaggle waste pictures](https://www.kaggle.com/datasets/wangziang/waste-pictures) .
A detailed description of waste datasets can be found [here](https://github.com/AgaMiko/waste-datasets-review).
The pre-trained deep learning VGG-19 CNN model achieves a training and test accuracy of 93% for 25 classes, 100 epochs, and 0.0001 learning rate.

# TRASHY classes:
'battery', 'beverage cans', 'cardboard', 'cigarette butt', 'construction scrap', 'crockery', 'electronic device', 'ewaste',
'glass bottles', 'leaflets', 'lightbulb', 'medical', 'medicines', 'metal containers', 'news paper', 'paper', 'paper cups', 'pens',
'plastic bags', 'plastic bottles', 'plastic containers', 'plastic cups', 'small appliances', 'syringe', 'tetra pak'

# TRASHY folders:
TRAIN: 10068 images, TEST: 5045 images, VALIDATION: 5033 images. Each class has about 750-850 images approximately.

# Python Installation

## Download and install [Python](https://realpython.com/installing-python/) for Operating System [Windows/MAC OS](https://www.python.org/downloads/) and open terminal to check version
```
$ python3 --version
$ python3 -m pip --version
```

## Install pip if not supported by OS:
For [Windows OS](https://phoenixnap.com/kb/install-pip-windows), download pip file and get pip file, and check pip version:
```
$ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
$ python get-pip.py
```

For [MAC OS](https://howchoo.com/python/install-pip-python), install homebrew, install python using homwbrew that will have pip inside:
```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
$ brew install python

or
$ sudo easy_install pip
```

## Install [python packages](https://packaging.python.org/en/latest/tutorials/installing-packages/) using pip 
`$ pip3 install <package-name>` or `python -m pip install --upgrade <package-name>`
Ensure that an up-to-date version of setuptools is installed `python3 -m pip install --upgrade pip setuptools wheel`

# JupyterLab Setup
Install [Jupyter Lab](https://jupyter.org/install) via local machine's Terminal or [JupyterLab desktop](https://github.com/jupyterlab/jupyterlab-desktop).
```
Install JupyterLab with pip:

$ pip install jupyterlab

Once installed, launch JupyterLab with:

$ jupyter-lab
It will launch on a local server and takes the PC's storage for processing.
```
Install GitHub Desktop or create a repository online on your GitHub, also can install Git extension on JupyterLab to manage the project.

## Pre-Processing Images
- Upload the dataset from computer/laptop or clone a repository with dataset on JupyterLab,
- Resize and rotate the images using `resize.py` and `resize_constants.py` files,
- Change the number and name of categories according to the dataset, TRASHY has 25 classes,
- Replace the source path for original dataset before resizing and destination path for saving modified dataset after resizing,
  according to your dataset's location on device,
- Import the required libraries and packages after installing on jupyter terminal in the VGG notebook `modelVGGTrashy.ipynb file`,
- Load the dataloaders, split the dataset into train, test, val folders,
- Perform model training, saving, loading, visualizing, and evaluating.


## Resize and Rotate function
1. Read/Load Image: Reads the image file using `iio.imread` and stores it in the variable using the Python Image `iio` library.
2. `fileWalk()` walks through the specified directory and processes the files found within it. It creates the destination directory specified by `destPath` if it doesn't already exist.
3. Resize Image: The `resize()` resizes the input image to the specified dimensions `DIM1=384` and `DIM2=582`.
4. Rotate Image: If `dim1` is greater than `dim2`, rotate the image 90 degrees clockwise using `np.rot90`.
5. Saving Image: Writes the resized image to the destination directory using the `write` function.`write_image_imageio()` writes the processed image `img` using `iio.imwrite` as an image file to the specified `img_file` path.

## Model Training

Build and train a VGG19 model on the TRASHY dataset with resource code scripted in Python using the PyTorch framework on Jupyter Lab IDE:

1. Importing Libraries: Install and import the required libraries, such as `numpy, matplotlib, torch, torchvision`, and some functions from other Python files for image resizing, as described above.
2. Dataset Preparation: Define the relative path to the custom dataset folder with `train, test, and validation` sub-folders.
* Load the image labeled to a corresponding class.
* Convert the image into an array or pixel format (looking into the train, test, val folders, and sub-folders of categories) using `augment() function`.
* Normalize pixel values to a scale of 0 to 1 for clear visibility, resize to the size of 384x582 pixels, and rotate to increase the diversity of training samples.
* Save the images into the newly modified folder after pre-processing.
3. Separate the images into train, test, and val sets for the original dataset by running `split_train_val_test() function` after resizing
4. Device Selection: Using a personal computer for model training, the seamless switching between GPU and CPU is used for computation in PyTorch based on availability.
5. Model Initialization: Initialize and  `load the pre-trained VGG-19 model with pre-trained weights` using Pytorch deep learning framework and load the dataloaders from `dataloaders.py file`.
6. Model Customization: Define the model's layers, input (single image), output (class prediction), loss function, and optimizer, and `set the learning rate, batch size, number of classes, and epochs (training iterations) in the VGG class`. We replace the last fully connected layer with a new layer for the number of categories in the TRASHY dataset (i.e., 25).
7. Model Training `train() function`: The model will learn the weights of the last layer during training to predict the waste class for an input trash image. Iterate through the training set in batches and pass the image dataset and epochs number through the VGG-19 model using vgg class object that saves the best accurate trained model to the given path.
8. Evaluation and Testing `eval() function`: Load the saved model using the `load()` function from the specified path to check the model's summary. We evaluate the trained model periodically on the validation and test set to monitor `average loss, average accuracy, total loss, and predictions`.
9. Hyperparameter Tuning: We also experimented with different hyperparameter settings to find the best combination, like changing the learning rate and number of epochs.
10. Visualization `visualize() function`: It is to `plot a confusion matrix` for the `model's accuracy` and `correct and incorrect predictions` on a subset of images from the test set.

Evaluated the model on learning rate= 0.0001 to 1, number of epochs = 10 to 100 (no early stopping), number of classes = 7 to 25, and employed a batch size = 64 for each training iteration.

## Deploying the pre-trained VGG-19 model on Raspberry Pi 4 model B 8GB (Rpi 4)
1. We write in the Vgg model python notebook to save the trained model with the ".pt" extension, a Pytorch format compatible with the Rpi architecture.
2. The Rpi OS installs all the necessary dependencies, so we open the Rpi terminal to check the OS version ( `uname-a` command) to match with arm64 bullseye operating system and have the latest versions of python3 and pip3.
```
$ sudo apt update 		                            // to update installed packages and library
$ sudo apt upgrade		                            //to upgrade to the latest versions
$ sudo apt install python3 pip3
$ python3 --version				                        //check version
$ sudo pip3 install torch torchvision		          // install required libraries
$ sudo pip3 install numpy --upgrade		            //upgrading a package via pip3
$ python3 $ import torch 		                      //check importing a package
$ sudo apt-get install rpi.gpio	                  // install a package if missing
$ pip install opencv-python	                      // package for image processing
```
3. After installing the required packages, open `Python IDE on Rpi Desktop or terminal` to check to import relevant packages like torch, torchvision, and cv2.
4. If the torch library is missing, here is a [step-by-step blog](https://medium.com/secure-and-private-ai-writing-challenge/a-step-by-step-guide-to-installing-pytorch-in-raspberry-pi-a1491bb80531) 
on torch and torchvision installation.
```
	# install gdown to download from Google Drive
		$ sudo -H pip3 install gdown
	# download the wheel (wheel file to install Python extensions)
		$ gdown https://drive.google.com/uc?id=1uLkZzUdx3LiJC-Sy_ofTACfHgFprumSg
	# install PyTorch 1.13.0
		$ sudo -H pip3 install torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl
	# clean up
		$ rm torch-1.13.0a0+git7c98e70-cp39-cp39-linux_aarch64.whl
```
5. Click on the start menu, then select Preferences, choose Raspberry Pi Configuration, and click on Interfaces to `enable SSH, camera, and GPIO` via Desktop, and another way is using the command `sudo raspi-config` on the terminal to choose interfaces using arrows.
```
# to check the IP address of Rpi at the terminal  
  $ hostname -I
# to connect the computer/laptop to Rpi                                    
  $ ssh [rpi_username]@[hostname].local or ssh [rpi_username]@[rpi_ip address] 			
```
6. Transfer the trained model file to Rpi storage via USB or network transfer like WinSCP by providing the IP address of Rpi.
7. Make a new Python script in Rpi's Python Interpreter to write the code for loading and visualizing the model (`load and visualize` functions as defined in VGG-19 class) and perform inference on the input data by importing all the required libraries* and returning the prediction to check the waste category that triggers LED blinking to inform about the bin color.
8. Using the `picam2 library` to initiate the camera preview, take pictures after pressing the button, and stop the preview after capturing. [Useful links](https://www.tomshardware.com/how-to/use-picamera2-take-photos-with-raspberry-pi) for [Picam2 library](https://pypi.org/project/picamera2/0.2.2/)
9. We checked the model processing firstly on a 2GB RAM Rpi 4, and it failed to load the model and predict the waste category, then used an 8GB RAM Rpi 4 that successfully processed the model and predictions.
10. For testing the model, 
	-> Created a prediction folder (containing a single image of any waste object category) next to the train, test, and val folders, 
	-> Imported open cv to read the image and loaded the model to predict the image from the "pred" folder and map to the respective bin color.

* Libraries required: (numpy, torch, torchvision, pandas, cv2, picamera2, sys, os, pathlib, PIL for image, Rpi.GPIO, gpiozero for Button, LED).


## Waste-To-Bin : Color-coded bins for waste categories in [Germany](https://handbookgermany.de/en/waste-separation).

- Blue Bin = ‘cardboard', 'news paper’, ‘paper’, ‘leaflets’ 
- Glass Bin = ‘glass bottles'
- Special Waste Bin = 'ewaste’, 'electronic device’, ‘battery', 'small appliances’, 'construction scrap’)
- Yellow Bin = 'beverage cans’, 'metal containers’, 'paper cups’, 'tetra pak', 'plastic bags’, 'plastic bottles’, 'plastic containers’, 'plastic cups’
- Black Bin = 'cigarette butt’, ‘medical', ‘crockery', ‘medicines', 'pens’, ‘lightbulb’, ‘syringe’

## Wiring Diagram: [Source article](https://www.instructables.com/Make-a-Pi-Trash-Classifier-With-ML/) See diagram below
The TRASHY circuit design has a push button, LEDs, and resistors on the breadboard that connect GPIO pins on the Rpi board by inserting the male end of jumper wires into the breadboard's holes and the female end mounted on the pins.

Breadboard consists of two vertical trails (+ve for voltage and -ve for ground). Pushbutton has four similar legs, and LEDs have one longer leg (+ve anode) and one shorter leg (-ve cathode). We insert the resistors horizontally in a U-shape on the breadboard to limit the current flow, one end into the GND line and another next to the LED's shorter leg and pushbutton's leg. Here is the connection of the other legs of the Pushbutton and LEDs using Jumper wires:
1. GND and Voltage: We insert the male ends of a black jumper wire into the -ve GND hole and a red wire into the +ve voltage hole at the top of the breadboard, and the female ends of the black wire onto the GPIO GND pin (3rd position from the top on the right side) and red wire onto the GPIO 5V pin (1st position on the right side).
2. Pushbutton: A purple wire's male end next to the second leg of the pushbutton and female end onto the GPIO pin 2 (2nd position on the left side of the pin header).
3. Yellow LED (for Yellow Bin): A yellow wire's male end is inserted next to the longer leg and female side onto the GPIO pin 17 (6th position on the left side).
4. Blue LED (for Blue Bin): A blue wire's male end is inserted next to the longer leg and female side onto the GPIO pin 27 (7th position on the left side).
5. Green LED (for Black Bin): A green wire's male end next to the longer leg and female side onto the GPIO pin 15 (5th position on the right side).
6. Red LED (for Glass Bin): A red wire's male end is inserted next to the longer leg and female side onto the GPIO pin 14 (4th position on the right side).
7. White LED (for Special Waste Bin and also as status LED for button): A white wire's male end next to the longer leg and female side onto the GPIO pin 24 (9th position on the right side).

![Trashy Connections](https://content.instructables.com/FIB/COST/KG6JRJP8/FIBCOSTKG6JRJP8.jpg?auto=webp&frame=1&width=388&height=1024&fit=bounds&md=9ca0e776356c49fbad3090296c3f4dc3)

## Results

The confusion matrix in Figure below shows the number of correct and incorrect predictions and the confusion between the waste categories in the dataset. If two classes have zero samples in the matrix, then the classifier learned the classification between classes very well (For example, cardboard and newspaper, metal containers and plastic bottles, and more). The diagonal represents the correct predictions, and everything else is incorrect or misclassified (For example, plastic bottles and glass bottles, newspaper and paper). The least accurate class is glass due to its transparency.

![Confusion Matrix](https://github.com/AVANI20/TRASHYImages/blob/main/trashyvisualsaccuracynew_page-0002%20(2)%20(1).jpg?raw=true)


Here, we conclude the successful working of the prototype TRASHY. We run the Python file in the Rpi terminal, and 
- a camera preview opens up,
- the white status LED starts blinking,
- asks the user to press a button,
- place an object in front of the camera and check the preview on the desktop,
- the camera takes a picture after pressing the button, and the white LED stops blinking,
- processes model initialization, loading, and prediction,
- the display screen shows the waste category and bin color names, and the LED light lightens up for the respective bin,
- asks the user to either quit or repeat the process again to predict other waste.

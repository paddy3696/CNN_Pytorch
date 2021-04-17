# Image Classification using Convolution Neural Network (CNN)
# Fundamentals of DL course Assignment - 2 


## Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [Pytorch](https://pytorch.org/)
- [wandb](https://wandb.ai/site)
- [wget](https://pypi.org/project/wget/)
- [mapextrackt](https://pypi.org/project/mapextrackt/)

## Code
- #### Neural network Class (MyNN)
The neural network class contains all the functions required for training the model. The function **fit** is the training function which contains the neural network pipeline (Forward_prog, compute_loss, back_prop, update_parameters). The activation functions (Sigmoid, tanh, relu) and weights, bias initialization (xavier, normal) are independently defined. 

The one hot encoding is performed to deal with categorial output variable. and it is defined as a seperate function.

The following line of code is an example to define a model using the MyNN class:

```python
model = MyNN(network_size=layers,network_fns=act,batch_size = 64, loss_fn = 'crossE'
             optimizer='NADAM',regularize= 'l2',alpha = 0, wb_init = 'xavier_uniform',
             learning_rate = 1e-3, max_epoch=5,verbose=1,seed=25)
```
After defining the model, the training of the model can be done using the following command:
```python
model.fit(X,Y,x_valid,y_valid)
```
- #### Wandb configuration
Wandb is a tool for tuning the hyper-parameters of a model. The wandb sweep requires to define a sweep configuaration with hyper-parameters in a dictionary type. The following code snippet is an example of defining the wandb sweep configuration:
```python
sweep_config = {
    'method': 'bayes', #grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'drop_out': {
            'values': [0.2, 0.3]
        },
        'batch_norm': {
            'values': ['Yes', 'No']
        },
        'filter_n': {
            'values': [64, 128]
        
        },
        'batch_size': {
            'values': [16, 32]
        },
        'filter_org': {
            'values': ['same', 'double_up', 'double_down']
        },
        'epoch': {
            'values': [5,10]
        },
        'data_aug': {
            'values': ['Yes', 'No']
        },
        'optimizer': {
            'values': ['SGD','ADAM'] 
        },
        'lr': {
            'values': [0.1,0.01] 
        },
        'hidden_out': {
            'values': [128,196] 
        },
    }
}
```

```
sweep_id = wandb.sweep(sweep_config, entity="paddy3696", project="cnn_inat")
```
- #### Train sweep function
The function **train** is the main function called by the wandb sweep. This function contains the wandb initialization and data pre-processing.  

- #### Testing
The function **model_test** finds the accuracy of the model with test data and plots the Confusion matrix heatmap.


## Run

In a terminal or command window, navigate to the top-level project directory `CNN_Pytorch/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Inat_cnn_train.ipynb
```  
or
```bash
jupyter notebook Inat_cnn_train.ipynb
```
The code for evaluating the perfomance of the custom CNN model with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook Inat_cnn_test.ipynb
``` 
The code for evaluating the perfomance of the pretrained CNN models with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook inat_cnn_pretrained.ipynb
``` 

## Data
The iNaturalist datasetis downloaded directly from the downloadable link using the following the "wget" command:
```python
wget.download('https://storage.googleapis.com/wandb_datasets/nature_12K.zip')
```
### Data Preprocessing
- The iNaturalist dataset contains 9999 training image data and 2000 testing.
- The training data is very split with ratio of 90:10 for training and validation. This is done to avoid overfitting.
- All the test, train and validation data are imported using the data loader function in torch library.
- The transfromers function are used for resizing, cropping, normalizng the images and then convert it to tensors.

## Report link
[wandb_report](https://wandb.ai/paddy3696/cnn_inat/reports/FDL-Assignment-2---Vmlldzo2MDg3Mzg?accessToken=l08ezysoh00yvd68sdpq7r78rvq5l2zjaxbjg6li81d982eu2we6xqky99wuol3r)

## Reference
- Udacity Deep Learning course
- https://youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh
- https://youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4
- https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
- https://pypi.org/project/mapextrackt/
- https://www.kaggle.com/sironghuang/understanding-pytorch-hooks?select=fig1.png
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- https://github.com/ultralytics/yolov5
- https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/

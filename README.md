# CNN
# Fundamentals of DL course Assignment - 2 


## Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [wandb](https://wandb.ai/site)


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
        'max_epoch': {
            'values': [20, 30]
        },
        'wb_init': {
            'values': ['he', 'xavier_uniform']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'hidden_size': {
            'values': [32, 64, 128]
        },
        'n_hidden': {
            'values': [3,4,5]
        },
        'alpha': {
            'values': [0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['SGD','SGDM','RMSP','ADAM','NADAM'] 
        },
        'activation': {
            'values': ['relu','sigmoid','tanh']
        },
    }
}
```
It is to be noted that these parameters can be changed and additional paramters for tuning can also be added.

- #### Train sweep function
The function **train** is the main function called by the wandb sweep. This function contains the wandb initialization and data pre-processing.  

- #### Testing
The function **model_test** finds the accuracy of the model with test data and plots the Confusion matrix heatmap.


## Run

In a terminal or command window, navigate to the top-level project directory `NN_from_scratch/` (that contains this README) and run one of the following commands:

```bash
ipython notebook MNIST_Classification.ipynb
```  
or
```bash
jupyter notebook MNIST_Classification.ipynb
```
The code for evaluating the perfomance of the model with MNIST Fashion data is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook MNIST_fasion_test.ipynb
``` 
The code for evaluating the perfomance of the model with MNIST handwriting data is seperately uploaded and it can be run using the following command:

```bash
jupyter notebook NN_MNIST_HW.ipynb
``` 

## Data
The MNIST fashion dataset is downloaded directly from the Keras library using the following the command:
```python
from keras.datasets import fashion_mnist
```
### Data Preprocessing
- The MNIST fashion dataset contains 80k image data. The data is split in the ratio of 90:10 for training and testing respectively.
- The training data is very split with ratio of 90:10 for training and validation. This is done to avoid Overfitting.
- All the test, train and validation data are normalized for better performance of the neural network

## Reference
- https://ml-cheatsheet.readthedocs.io/en/latest/index.html
- https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
- [Sentdex - Neural Network from scratch](https://youtube.com/playlist?list=PLQVvvaa0QuDcjD5BAw2DxE6OF2tius3V3)
- [CS7015- Deep Learning](https://youtube.com/playlist?list=PLyqSpQzTE6M9gCgajvQbc68Hk_JKGBAYT)
- https://www.deeplearningbook.org/
- https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6






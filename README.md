# 50021AIGrp21

## Description of how to setup your code in order to be able to run the model and/or GUI.
To set up and run the project, follow these steps:

### Clone the Repository
Clone the project repository to your local machine using this command:
```bash
git clone https://github.com/BoweiDu01/50021AIGrp21.git
```
Or click on Code > Download ZIP and extract the folder on your local machine.

### Install Dependencies
Install the required Python packages using:

```bash
pip install -r requirements.txt
```

### Download and Prepare the Dataset

Download the dataset from the project's GitHub repository or the provided source.

Extract or place the dataset inside a folder named dataset located at the root of the project directory.

The expected structure should follow the standard ImageFolder format, with subdirectories for each class.

### Configure Parameters 

You may modify key hyperparameters (e.g., batch size, learning rate, number of epochs) in the script. 

A list of modifiable hyperparameters can be found in the description of hyperparameters section

### Run Training

Execute the training script or run the training cell in the notebook to begin model training.

The model will be trained using early stopping, and training artifacts (e.g. model weights, loss logs) will be saved to the models/ directory
### Visualize and Evaluate

After training is complete:

Run the post-training cell to generate a plot of training and validation loss over epochs.

The best validation loss is highlighted in the plot for easy identification.

A confusion matrix and F1 score are computed on the test set to evaluate model performance.

###Pretrained Weights

Model weights can be accessed through this link. 
https://drive.google.com/drive/folders/1cLbZhj7afSI0w5M-bXx25hB88afFwKUo?usp=sharing

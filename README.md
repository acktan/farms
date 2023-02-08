# farms
This repository is made to help predict the future yield of batches of mushrooms for a mushroom farm in eastern Africa.

## 🚀 Getting started with the repository

To ensure that all libraries are installed pip install the requirements file:
 
```
pip install -r requirements.txt
```

To run the model go to the console and run following command: 
 
```
python main.py
```

You should be at the source of the repository structure (ie. farms) when running the command.

## 🗂 Repository structure

Our repository is structured in the following way:

```
|natives_deephedging
   |--data
   |--output
   |--src
   |-----evaluation
   |-----inference
   |-----loading
   |-----model
   |-----preprocessing
   |-----train
   |--main.py
   |--README.md
   |--requirements.txt
   |--config.json
```

### 📊 Data

The Data folder contains all the datasets used to train the models and the test input datasets used for inference.

### ↗️ Output
The output folder contains two graphs assessing the feature importance of the model input and comparing the train predictions to the actual predictions and including the mean absolute error of the test set. This is only the case when variable 'val' is defined as True in the config file. If not, then the model will be trained on the entire test set and will instead predict the yields for mushrooms that have not yet been harvested.

### ℹ️ src
The src folder contains all the different classes combined in the main.py file. The following is a description of all classes used.


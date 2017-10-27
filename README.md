# Cs 433 Machine Learning Project I : Higgs Challenge
This is the result of the project I for the course cs 433 Machine Learning at EPFL.
The model developed is able to classify the events (signal or noise)
relevant for the detection of the Higgs Boson. More info in https://www.kaggle.com/c/higgs-boson/leaderboard
## Getting Started
Git clone the repository.
https://github.com/Idate96/cs_433_ML_project_1.git
To run the model:
```
python run.py
```
It will reproduce the best results obtained with it.
They are saved in the folder dataset under the name submission_0x.csv.

### Prerequisites
```
python  3.6x
numpy  1.13.x
```

## Running the tests
```
nosetests --verbose Do we need this?
```

## Architecture of the code

```

-src
--run.py # main
--ensemble_log_regression.py # contains the regression model
--utils.py # helper function to process and load data
--implementations.py # contains the implementation of ML_methods
--test_ML_methods # contains the dataset used during the lab sessions
--old_models # old_models considered for the assignment
-dataset # contains the dataset for the competition and the submission for kaggle
-config #where the weight of the model are saved

```

## Authors

* **Lorenzo Terenzi**
* **Alberto Bonifazi**
* **Laura Jou Ferrer**

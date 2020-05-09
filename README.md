# Rumor Detection

## Dependencies
* python == 3.6
* numpy == 1.16.4
* pytorch version == 1.0.1
* matplotlib

## Data
* Download data in the following link (nfold_new and resource) and put the two folders in the same level of `pruning` folder. Name them as `nfold_new` and `resource` respectively.
* Train and test splits are put inside `nfold_new` with additional user features.
* Labels and tf-idf encoding for tweets are put inside `resource`.
* [nfold_new](https://drive.google.com/drive/folders/1Lv3WpKHEkEeGRTBy6blqrzl5LaZmZJ9u?usp=sharing)
* [resource](https://drive.google.com/drive/folders/1ozEHSmxT3bUCK_ROONKnNWY_LHZ0Ynsa?usp=sharing)

## Additioanl User Features
* `Preprocessing.py`- We add below user properties to the input:

1. follower count of user
2. friend count of user
3. ratio of followers and friends
4. whether a verify account or not
5. registration time (year)
6. Number of history tweets

## Experiment Reproduction
### Pruning
* `cd` into `pruning` directory and run `main.py` script.
* Change line 12 in `main.py` to use ***additional user features***.
* Change line 16 in `utililty.py` to ***prune tree*** in 3 different ways.
* Change line 18-20 in `utility.py` to use different hyper parameters for ***tree pruning***.


## Reference
We develop our code based on the previous work of following github repo:

1. https://github.com/majingCUHK/Rumor_RvNN
2. https://github.com/ShuHwaiTeoh/AI_rumor_detection


# Attention version 

## Data
* download the dataset from the link and put it under the RumorDetection folder
* data process: 
under /dataset, twitter15_new and twitter16_new has the data  which has same records with Tree Structure but the format is different
* process.py load the data from the twitter15_new and twitter16_new to pickle file which is used in run.py
* [dataset](https://1drv.ms/u/s!ApBEiUtUqBjHzBdZn_bSE-GfA1Vd?e=aSRSWy)
## Run
* python run.py => this is only attention version.  The main file
* python run_cnn.py => this is the attention version with CNN of user profile. The result is bad, just for reference

## Reference
The code is modified from
https://github.com/chunyuanY/RumorDetection


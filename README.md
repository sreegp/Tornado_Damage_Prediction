# Tornado_Damage_Prediction

Goal: 

I wanted to predict the level of damage tornadoes can cause, when information about an imminent tornado starts coming in. People can be taken to safety sooner and casualties can be treated quickly, with hospitals and emergency services prepared and full equipped to deal at over the maximum capacity.

Problem and Approach: 

The problem was considered as a classification problem, predicting whether the tornado damage would be severe (class 1) or mild (class 0). I divided the data into train, validation and test based on the progression of time, with the model being trained on older information and making predictions on new tornado information. 

I trained logistic regression, decision tree and random forest models to predict tornado casualties. I performed data balancing, cross validation for tuning hyperparameters and evaluated best model using AUC value.

Code is modularized, commented and easy to read. The "Main" iPython notebook calls the preprocessing.py, scaling_and_sampling.py and model_selection.py files.

Credits: 

Data processing ideas were discussed with Orion Taylor and Zane Dennis. Data processing code was a collaboration between me and Orion Taylor. I modularized all the code and worked on sections such as model selection and evaluation.

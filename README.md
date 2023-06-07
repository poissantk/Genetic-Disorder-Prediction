# Genetic-Disorder-Prediction
Dataset from https://www.kaggle.com/datasets/aibuzz/predict-the-genetic-disorders-datasetof-genomes

Genetic-Disorder-Prediction uses a dataset of medical history such as blood cell count, heart rate, symptoms, and parental genetic information in order to predict the genetic disorder and disorder subclass. After cleaning and exploring the data, I ran different models with grid search cross validation in order to find the best model. I ended on a Random Forest model and a Naive Bayes model and achieved an averaged f1 score of .397, compared to a score achieved with random guessing of .198. I'm now working on cleaning the code playing around with column selection.

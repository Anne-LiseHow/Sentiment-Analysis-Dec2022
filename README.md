# Sentiment-Analysis
This repository contains the implementation of various machine learning techniques to classify the sentiment of airline tweets. The data consists of three variables: tweet ID, text and airline sentiment. The tweet ID is a unique identifier for each tweet and can be disregarded since it does not provide any additional information.

## Exploratory Data Analysis
Before analysis, any missing or duplicates were removed from the data. The proportion of positive, neutral and negative tweets are approximately equal for the three data sets. Negative tweets make up the largest majority with at least 60%, while neutral tweets are the smallest group with at most 16.2%. This suggests that we might encounter models that are biased in predicting the negative class due to the latter being the majority.

TO DO: Include pictures of graphs

## Data Preprocessing for Sentiment Analysis
The following pre-processing steps were conducted:
* Convert tweet to lower case
*	Remove hyperlinks
* Convert html entities to string
* Expand contracted words
* Remove punctuations
* Remove numbers
* Remove stop words
* Split into train inputs (X) and train outputs (Y)
* Tokenizing and Normalisation: Lemmatization, Porter Stemming, Lancaster Stemming
* Convert to Numeric Representation: Bag of Words, Term Frequency-Inverse Document Frequency (Tf-Idf)

## Machine Learning Models
Three traditional machine learning models were implemented:
1.	Multinomial Naïve Bayes Classifier (MNB)
2.	Gaussian Naïve Bayes Classifier (GNB)
3.	Random Forest Classifier (RFC)

With different combinations of normalisation methods, convertion to numeric representation methods, and machine learning model, a total of 24 models have been trained. 

Bi-LSTM model was also implemented with the Google Word2Vec embedding matrix and our own created embedding matrix. However, due to imbalance in the dataset, the model was overfitting and predicting the majority class for every input.

### Hyperparameter Tuning
Grid Search and Randomized Search were used for determining optimal values of the models.

### Results
Out of the 24 models, Multinomial Naïve Bayes Classifier with a f1-score of (0.86 0.55 0.65) and the Random Forest Classifier, with (0.85 0.45 0.60).  Both classifiers attained the highest f1-score with the same processing steps: Porter Stemming with the Bag of Words model.

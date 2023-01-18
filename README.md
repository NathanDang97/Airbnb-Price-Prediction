# Airbnb Price Predictor

## Introduction

This project is rooted from the final course challenge for a graduate Machine Learning course. In this final challenge, we were asked to build our own machine learning solution to predict the price of an Airbnb rental given the provided dataset. Pricing a rental property such as an apartment or house on Airbnb is a difficult challenge. A model that accurately predicts the price can potentially help renters and hosts on the platform make better decisions. 

## The provided data

The provided dataset was collected from the Airbnb website for New York, which has a total of 39,981 entries, each with 764 features. The data is split into a training set and a sample test set. The model was evaluated and graded via Mean Squared Error (MSE) score on a hidden test set known only to the teaching staff.

Some minimal data cleaning was already done beforehand, such as converting text fields into categorical values and getting rid of the NaN values. As a quick review, to convert text fields into categorical values, we can use different strategies depending on the field. For example, sentiment analysis was applied to convert user reviews to numerical values (’comments’ column). Different columns were for state names in one-hot-encoding style, ’1’ indicating the location of the property. Column names are included in the data files and are mostly descriptive. Also in this data cleaning step, the price value that we are trying to predict is calculated by taking the log of original price. Hence, minimum value for our output price is around 2.302 and maximum value is around 11.488. All input features have already been scaled to [0, 1].

All data files can be found in the "Data" folder in this repo.

## Implementation

The starting point was to perform more basic data cleanup that had not been done beforehand (i.e. handling NaN). Some data visualization definitely helps in this case and was implemented as well. We were asked not to include data visulaization code in the submission for some auto grading reasons. Thus, the visualization does not appear in the current state of this repo, but it can be easily added back later if needed.

The chosen model was a simple Linear Model (using Linear Regression) with L2-Regularization. Since the number of data points was not large, an alternate implementation using Normal Equation was also provided. An obvious improvement for this project is to implement a more complicated model (e.g. Neural Network) to see if we can achieve a better performance.

The notebook containing all codes can be found in the "Main" folder in this repo.

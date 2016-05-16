
# Coursera: Practical Machine Learning: Week 1: Prediction Models

[https://class.coursera.org/predmachlearn-014/lecture](https://class.coursera.org/predmachlearn-014/lecture)


## Components of a predictor

question -> input data -> features of data -> algorithm -> parameters -> evaluation

* features:
    * important!
    * data compression
    * retain relevant info
    * expert application knowledge
* common mistakes:
    * automating feature selection
    * don't understand why features were selected
* algorithms aren't so important
    * often linear discriminants give nearly as good results
    * linear disc. tend to be easier to interpret
    * interpretability matters.


## In sample vs. out of sample error

* in-sample error: the error rate you get on the dataset used to build the predictor (training)
* out-of-sample error: the error rate on a new dataset (test)
* we care about out-of-sample
* in-sample < out-of-sample
* overfitting



## Prediction Study Design


1. define your error rate
2. split data into:
    * training (60%)
    * test (20%)
    * validation (20%) (optional)
    * AVOID SMALL SAMPLES! more data == better predictor
3. analyze features of the training set
4. create a prediction function on the training set
5. apply prediction function to test set ONLY ONCE
    * if you iterate to try to improve performance on test set, then you're treating it like a training set.
    * could use a validation set


### Predictive analytics competition: [kaggle.com](kaggle.com) 



## Types of Errors


* for binary data (e.g diseased/not diseased):
    * True positive = correctly indentified by the predictor
    * True negative = correctly rejected by the predictor
    * False positive = incorrectly indentified by the predictor
    * False negative = incorrectly rejected by the predictor
* Sensitivity -> P( positive ident | True/correct ) -> TP / TP + FN
    * Sensitvity = "true positive rate"
    * 1 - sensitivity = "false negative rate"
* Specificity -> P( negative reject | True/correct ) -> TN / TN + FP 
    * Specificity = "true negative rate"
    * 1 - specificity = "false positive rate"
* Positive Predictive Value -> P( True/correct | positive ident) -> TP / TP + FP
* Negative Predictive Value -> P( True/correct | negative reject) -> TN / TN + FN
* Accuracy -> P( correct ) -> TP + TN / TP + TN + FP + FN



Suppose that we have created a machine learning algorithm that predicts whether
a link will be clicked with 99% sensitivity and 99% specificity. The rate the
link is clicked is 1/1000 of visits to a website. If we predict the link will
be clicked on a specific visit, what is the probability it will actually be clicked?


               actual (100 actual clicks for every 100 * 1000 visits)
                +      - 
              -------------
    test  +     99     999 
          -     1     98901

    P(actual|test) = 99/(99+999) = 0.09


### Common error measures:

for continuous data:

* MSE (mean-squared error == variance): 1/n SUM( prediction - truth)^2
* RMSE (root mean-squared error == std dev): sqrt(MSE)


1. mean squared error
    * continuous data, sensitive to outliers
2. median absolute deviation
    * continuous data, more robust
3. Sensitivity
    * if you want few missed positives (FALSE NEGATIVES)
4. Specificity
    * if you want few negatives called positives (FALSE POSITIVES)
5. Accuracy     
    * weighs false positives / negatives equally
6. Concordance (?)



## ROC Curves: Receiver Operating Characteristics


* For binary outcomes
    * plots true positive rate (sensitivity) vs false positive rate (1 - specificity)
    * maximize area under the curve 
    * AUC = 0.5 is the 45 degree line. as good as flipping a coin



## Cross-Validation


* subset the training dataset, to avoid problems like overfitting
* K-fold
    * separate dataset into K sets of equal size
    * use 1 set as test, the remaining as train
    * iterate using each set as the test set




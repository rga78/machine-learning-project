
# Coursera: Practical Machine Learning: Week 4: Combining Models, Forecasting

[https://class.coursera.org/predmachlearn-014/lecture](https://class.coursera.org/predmachlearn-014/lecture)



## Regularized Regression: 


1. Fit a regression model
2. Penalize (shrink) large coefficients
    * Pro: helps with bias/variance tradeoff
        * if vars are correlated, they will have high variance
    * Pro: helps with model selection
    * Con: computationally intensive
    * Con: inferior to random forest or boosting


caret train models: lasso, ridge, relaxo



## Combining prediction models


* combine multiple models for better accuracy
* "majority" vote from the models

Procedure:

1. build multiple models on training set
2. combine/"stack" models and train together on the testing set
3. validate combined models on the validation set


#### Example

    R: 
    mod1 <- train(wage ~ ., method="glm", data=training)
    mod2 <- train(wage ~ ., method="rf", data=training)
    pred1 <- predict(mod1, testing)
    pred2 <- predict(mod2, testing)
    qplot(pred1, pred2, colour=wage, data=testing)
    
    
    # train combined model on test set
    predDF <- data.frame(pred1, pred2, wage=testing$wage)
    combModFit <- train(wage ~ ., method="gam", data=predDF)
    combPred <- predict(combModFit, predDF)
    
    
    # test combined model on validation set
    pred1V <- predict(mod1, validation)
    pred2V <- predict(mod2, validation)
    predVDF <- data.frame(pred1=pred1V, pred2=pred2V)
    combPredV <- predict(combModFit, predVDF)




## Forecasting


* Data are dependent over time
* Patterns: Time-series decomposition:
    * trends - long term
    * seasonal - related to time of week, month, year
    * cycles - patterns that rise and fall periodically
* sub-sampling into training/testing
    * need to sample in chunks, because of time dependency
* deps between nearby observations
* location specific effects
* beware of spurious correlation!
* beware extrapolation!
* Moving average
    * take average of previous n data points
* Exponential Smoothing
    * similar to moving average, but weight recent data points heavier than older ones


#### Example

    R:
    library(quantmod)
    from.dat <- as.Date("01/01/08", format="%m/%d/%y")
    to.dat <- as.Date("12/31/13", format="%m/%d/%y")
    getSymbols("GOOG", src="google", from=from.dat, to=to.dat)
    head(GOOG)
    mGoog <- to.monthly(GOOG)
    googOpen <- Op(mGoog)
    ts1 <- ts(googOpen, frequency=12) # time-series object
    plot(ts1,xlab="years + 1", ylab="GOOG")
    
    
    plot( decompose(ts1), xlab="Years 1" )
    
    
    ts1Train <- window(ts1, start=1, end=5)
    ts1Test <- window(ts1, start=5, end=(7-0.01))
    plot(ts1Train)
    lines( ma(ts1Train, order=3), col="red" ) # moving average
    
    
    ets1 <- ets(ts1Train, model="MMM") # exponential smoothing 
    fcast <- forecast(ets1)
    plot(fcast)
    lines( ts1Test, col="red")
    accuracy( fcast, ts1Test )



## Quiz 4 

[https://class.coursera.org/predmachlearn-014/quiz](https://class.coursera.org/predmachlearn-014/quiz)


### Quiz 4, question 1


    library(ElemStatLearn)
    data(vowel.train)
    data(vowel.test) 
    library(caret)
    
    
    vowel.test$y <- factor(vowel.test$y)
    vowel.train$y <- factor(vowel.train$y)
    set.seed(33833)
    mod.rf <- train(y ~ ., data=vowel.train, method="rf")
    mod.gbm <- train(y ~ ., data=vowel.train, method="gbm")
    
    
    pred.rf <- predict(mod.rf, newdata=vowel.test)
    sum( pred.rf == vowel.test$y ) / length(pred.rf)
    # [1] 0.6082251
    
    
    pred.gbm <- predict(mod.gbm, newdata=vowel.test)
    sum( pred.gbm == vowel.test$y ) / length(pred.gbm)
    # [1] 0.5108225
    
    
    pred.agree <- pred.gbm[ pred.gbm == pred.rf ]
    y.agree <- vowel.test$y[ pred.gbm == pred.rf ]
    sum(pred.agree == y.agree) / length(pred.agree)



### Quiz 4, question 2


    library(caret)
    library(gbm)
    set.seed(3433)
    library(AppliedPredictiveModeling)
    data(AlzheimerDisease)
    adData = data.frame(diagnosis,predictors)
    inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
    training = adData[ inTrain,]
    testing = adData[-inTrain,]
    
    
    set.seed(62433)
    mod.rf <- train(diagnosis ~ ., data=training, method="rf")
    mod.gbm <- train(diagnosis ~ ., data=training, method="gbm")
    mod.lda <- train(diagnosis ~ ., data=training, method="lda")
    
    
    pred.rf <- predict(mod.rf, testing)
    pred.gbm <- predict(mod.gbm, testing)
    pred.lda <- predict(mod.lda, testing)
    
    
    # train combined model on test set
    preds.df <- data.frame(pred.rf, pred.gbm, pred.lda, diagnosis=testing$diagnosis)
    mod.all <- train(diagnosis ~ ., method="rf", data=preds.df)
    pred.all <- predict(mod.all, preds.df)
    sum(pred.all == testing$diagnosis) / length(pred.all)
    sum(pred.rf == testing$diagnosis) / length(pred.rf)
    sum(pred.gbm == testing$diagnosis) / length(pred.gbm)
    sum(pred.lda == testing$diagnosis) / length(pred.lda)
    
    
    # try again: training combined model on the training set
    pred.rf <- predict(mod.rf, training)
    pred.gbm <- predict(mod.gbm, training)
    pred.lda <- predict(mod.lda, training)
    
    
    # train combined model on training set
    preds.df <- data.frame(pred.rf, pred.gbm, pred.lda, diagnosis=training$diagnosis)
    mod.all <- train(diagnosis ~ ., method="rf", data=preds.df)
    
    
    # test on testing set.
    pred.rf <- predict(mod.rf, testing)
    pred.gbm <- predict(mod.gbm, testing)
    pred.lda <- predict(mod.lda, testing)
    
    
    preds.df <- data.frame(pred.rf, pred.gbm, pred.lda)
    pred.all <- predict(mod.all, preds.df)
    
    
    sum(pred.all == testing$diagnosis) / length(pred.all)
    sum(pred.rf == testing$diagnosis) / length(pred.rf)
    sum(pred.gbm == testing$diagnosis) / length(pred.gbm)
    sum(pred.lda == testing$diagnosis) / length(pred.lda)



### Quiz 4, question 3


    set.seed(3523)
    library(AppliedPredictiveModeling)
    data(concrete)
    inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
    training = concrete[ inTrain,]
    testing = concrete[-inTrain,]
    
    
    set.seed(233)
    mod <- train( CompressiveStrength ~ ., data=training, method="lasso" )
    plot(mod$finalModel,use.color=T,xvar="penalty")




### Quiz 4, question 4


    dat <- read.csv("gaData.csv")
    training = dat[year(dat$date) < 2012,]
    testing = dat[(year(dat$date)) > 2011,]
    tstrain = ts(training$visitsTumblr)
    
    
    mod <- bats(tstrain)
    fcast <- forecast(mod, h=nrow(testing))
    plot(fcast)
    
    
    tstest <- ts(testing$visitsTumblr, start=366)
    lines(tstest, col="red")
    
    
    sum(testing$visitsTumblr > fcast$lower & testing$visitsTumblr < fcast$upper) / length(testing$visitsTumblr)
    [1] 0.9617021



### Quiz 4, question 5


    set.seed(3523)
    library(AppliedPredictiveModeling)
    data(concrete)
    inTrain = createDataPartition(concrete$CompressiveStrength, p = 3/4)[[1]]
    training = concrete[ inTrain,]
    testing = concrete[-inTrain,]
    set.seed(325)
    library(e1071)
    
    
    mod <- svm( CompressiveStrength ~ ., data=training )
    preds <- predict( mod, newdata=testing )
    RMSE( preds, testing$CompressiveStrength )




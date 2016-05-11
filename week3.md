
# Coursera: Practical Machine Learning: Week 3

[https://class.coursera.org/predmachlearn-014/lecture](https://class.coursera.org/predmachlearn-014/lecture)



## Classification Trees

* non-linear models
* use interactions between variables
* monotone transformations less important
* can be used regression problems
* can use RMSE for purity measure
* R: caret: party, rpart, 
* R: tree


### Basic algorithm:

1. start with all vars in 1 group
2. find the var/split that best separates the outcome
3. divide data into two groups ("leaves") on the split ("node")
4. repeat procedure for each subgroup, until satisfied.


#### Measure of impurity:


    p^_mk = #-of-outcomes-in-class-k
            -------------------------
            total-#-of-outcomes-in-leaf


#### misclassification error:


    1-p^_mk = #-of-outcomes-NOT-in-class-k
                -------------------------
                total-#-of-outcomes-in-leaf


#### gini index (not to be confused with gini coefficient):

    = p^_mk * (1 - p^_mk)


#### deviance/information gain:

    = p^_mk * log_2 (p^_mk)


#### Example

    R:
    library(caret)
    data(iris)
    inTrain <- createDataPartition(y=iris$Species, p=0.7, list=F)
    training <- iris[inTrain,]
    testing <- iris[-inTrain,]
    qplot(Petal.Width, Sepal.Width, colour=Species, data=training)
    // three distinct clusters
    
    
    // rpart = classification and regression trees (CART)
    modFit <- train(Species ~ ., method="rpart", data=training)
    modFit$finalModel
    plot(modFit$finalModel, uniform=T, main="Classification Tree")
    text(modFit$finalModel, use.n=T, all=T, cex=0.8)
    
    
    // prettier
    library(rattle)
    fancyRpartPlot(modFit$finalModel)
    predict(modFit, newdata=testing)




## Bagging / Bootstrap Aggregating


Average models together.  

* similar bias
* reduced variance
* more useful on non-linear models

Procedure:

1. Resample data (w/ replacement) and recalculate predictions
2. Average / majority vote


#### Example

    R:
    library(ElemStatLearn)
    data(ozone, package="ElemStatLearn")
    ozone <- ozone[order(ozone$ozone),]
    ll <- matrix(NA, nrow=10, ncol=155)
    
    
    // loop over 10 re-samples
    for (i in 1:10) {
    ss <- sample(1:dim(ozone)[1], replace=T)
    ozone0 <- ozone[ss,]
    ozone0 <- ozone0[ order(ozone0$ozone), ]
    // loess = smooth curve thru the data
    loess0 <- loess( temperature ~ ozone, data=ozone0, span=0.2)
    ll[i,] <- predict(loess0, newdata=data.frame(ozone=1:155))
    }
    
    
    plot(ozone$ozone, ozone$temperature, pch=19, cex=0.5)
    for(i in 1:10) {
    lines(1:155, ll[i], col="grey", lwd=2)
    }
    // bagged loess curve
    lines(1:155, apply(ll,2,mean), col="red", lwd=2)
    
    
    // alternative
    train (method="bagEarth", "treebag", "bagFDA")
    bag(model)




## Random Forests

Procedure:

1. bootstrap samples
2. at each split, bootstrap variables (select subset of vars (at random))
    * diverse set of trees
3. vote/average trees


Extension to bagging.

* pro: very accurate.
* con: slow
* con: hard to interpret
* con: overfitting (see rfcv function)


#### out-of-bag error: 
    
[http://stackoverflow.com/questions/18541923/what-is-out-of-bag-error-in-random-forests](http://stackoverflow.com/questions/18541923/what-is-out-of-bag-error-in-random-forests)


#### Example

    R:
    data(iris)
    modFit <- train(Species ~ ., data=training, method="rf", prox=T)
    getTree(modFit$finalModel, k=2)
    irisP <- classCenter(training[,c(3,4)], training$Species, modFit$finalModel$prox)
    irisP <- as.data.frame(irisP)
    irisP$Species <- rownames(irisP)
    p <- qplot(Petal.Width, Petal.Length, col=Species, data=training)
    p + geom_point(aes(x=Petal.Width,y=Petal.Length,col=Species), size=5, shape=4, data=irisP)
    
    
    pred <- predict(modFit,testing)
    testing$predRight <- (pred == testing$Species)
    table(pred, testing$Species)
    qplot(Petal.Width, Petal.Length, colour=predRight, data=testing)



## Quiz 3

### Quiz 3, Question 1: 



    library(AppliedPredictiveModeling)
    data(segmentationOriginal)
    library(caret)
    so <- segmentationOriginal
    so.test <- subset(so, Case=="Test")
    so.train <- subset(so, Case=="Train")
    set.seed(125)
    modFit <- train(Class ~ . , method="rpart", data=so.train)
    library(rattle)
    fancyRpartPlot(modFit$finalModel)
    so.1 <- so.train[1,]
    so.1$TotalIntenCh2[1] <- 23000
    so.1$FiberWidthCh1[1] <- 10
    predict(modFit, newdata=so.1)
    [1] PS




### Quiz 3, Question 3:


    library(pgmm)
    data(olive)
    olive = olive[,-1]
    newdata = as.data.frame(t(colMeans(olive)))
    modFit.t <- tree(Area ~ ., data=olive)
    predict(modFit.t, newdata=newdata)




### Quiz 3, Question 4:


    library(ElemStatLearn)
    data(SAheart)
    set.seed(8484)
    train = sample(1:dim(SAheart)[1],size=dim(SAheart)[1]/2,replace=F)
    trainSA = SAheart[train,]
    testSA = SAheart[-train,]
    
    
    missClass = function(values,prediction){sum(((prediction > 0.5)*1) != values)/length(values)}
    
    
    modFit <- train(chd ~ age + alcohol + obesity + tobacco + typea + ldl, data=trainSA, method="glm", family="binomial")
    preds.train <- predict(modFit, newdata=trainSA)
    preds.test <- predict(modFit, newdata=testSA)
    missClass(trainSA$chd, preds.train)
    [1] 0.2727273
    missClass(testSA$chd, preds.test)
    [1] 0.3116883



### Quiz 3, Question 5:


    library(ElemStatLearn)
    data(vowel.train)
    data(vowel.test)
    vowel.train$y <- as.factor(vowel.train$y)
    set.seed(33833)
    modFit <- train(y ~ ., data=vowel.train, method="rf")
    varImp(modFit)





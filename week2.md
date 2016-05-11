
# Coursera: Practical Machine Learning: Week 2

[https://class.coursera.org/predmachlearn-014/lecture](https://class.coursera.org/predmachlearn-014/lecture)


## R: caret package functionality: 


1\. Preprocessing

    preProcess()

     // pair-wise plot of predictors vs outcome
    featurePlot(x=..predictors.., y=..outcome.., plot="pairs")


2\. Data splitting

    createDataPartition()
        - create series of test/training partitions
    createResample()
        - create one or more bootstrap samples
    createFolds()
    createMultiFolds()
        - split data into k groups
    createTimeSlices()
        - creates cross-validation sample info for time-series data


3\. Training/testing

    train()
    predict()


4\. Model comparison

    confusionMatrix()



## Machine Learning Algorithms in R


* Linear Discriminant Analysis
* Regression
* Naive Bayes
* Support Vector Machines
* classification and regression trees
* random forests
* boosting
* etc.



## Spam example 


    R:
    library(caret)
    library(kernlab)
    data(spam)
    inTrain <- createDataPartition(y=spam$type, p=0.75, list=F)
    training <- spam[inTrain,]
    testing <- spam[-inTrain,]
    
    set.seed(1)
    modelFit <- train(type ~ ., data=training, method="glm")
    modelFit // summary info
    modelFit$finalModel // model coefficients
    
    predictions <- predict(modelFit, newdata=testing)
    predictions // outcomes (spam/nonspam)
    
    confusionMatrix(predictions, testing$type)




## Preprocessing


* why? if a predictor is skewed
    * skewed vars are harder to deal with in models
* z-scaling / standardizing
    * "center": x - mean(x)
    * "scale" : x - mean(x) / sd

.

    preProcess(training[,-58], method=c("center", "scale"))



## Covariate creation


* Covariates: Predictors / Features that will be used in model 
    * "covariate" with outcome
* Level 1. Raw data -> covariates (variables that describe raw data)
    * balance between summarization vs info loss
    * err on side of more features (less info loss)
* Level 2. tidy covariates -> new covariates (via transformation)
    * more necessary for regression, svms (models depend on nice data dist)
    * ONLY ON TRAINING SET!!
    * apply same function to testing set later
    * exploratory analysis
    * preProcess()

.

    R:
    library(ISLR)
    library(caret)
    data(Wage)
    inTrain <- createDataPartition(y=Wage$wage, p=0.7, list=F)
    training <- Wage[inTrain,]
    testing <- Wage[-inTrain,]
    
    
    // dummy vars for factors
    dummies <- dummyVars(wage ~ jobclass, data=training)
    predict(dummies, newdata=training)
    
    
    // remove zero covariates
    nsv <- nearZeroVar(training, saveMetrics=T)
    
    
    // curvy model fitting (df=3: age, age^2, age^3)
    library(splines)
    bsBasis <- bs(training$age, df=3)
    lm1 <- lm(wage ~ bsBasis, data=training)
    plot(training$age, training$wage, pch=19, cex=0.5)
    points(training$age, predict(lm1, newdata=training), col="red", pch=19, cex=0.5)
    
    
    // apply bsBasis (derived from training) to testing set
    predict(bsBasis, age=testing$age)



## Principal Component Analysis


* for correlated predictors
* combine correlated vars 
    * reduce # of predictors
    * weighted combination
    * capture as much info as possible

.

1. Find new set of multivariate variables that are uncorrelated and explain as much variance as possible
    * statistical goal
2. Find the best matrix with fewer variables (lower rank) that explains original data
    * data compression


### SVD - singular value decompisition

* `X = UDV^T`
* where,
    * cols of U are orthogonal - left singular vectors
    * cols of V are orthogonal - right singular vectors
    * D is diagonal matrix - singular values

### PCA - principal component analysis

* the PCs are the right singular values (if data is scaled)

.

    R:
    // gen enough PCs to capture thresh=0.8 (80%) of variance
    // note: automatically centers and scales data
    preProcess( training.il, method="pca", thresh=0.8 )
    
    
    library(caret)
    library(kernlab)
    data(spam)
    inTrain <- createDataPartition(y=spam$type, p=0.75, list=F)
    training <- spam[inTrain,]
    testing <- spam[-inTrain,]
    
    
    M <- abs( cor(training[,-58]) ) // -58: remove outcome var
    diag(M) <- 0 // correlation between var and itself -> 0
    which(M > 0.8, arr.ind=T)
    
    
    smallSpam <- spam[,c(34,32)] // 34,32: highly correlated
    pc <- prcomp(smallSpam)
    plot(pc$x[,1], pc$x[,2])
    pc$rotation // shows weights
    
    
    typeColor <- ((spam$type=="spam")*1 + 1)
    pc <- prcomp(log10(spam[,-58]+1)) // pca on all predictors. log10 -> normally distributed vars (needed for pca to make sense)
    plot(pc$x[,1], pc$x[,2], col=typeColor, xlab="pc1", ylab="pc2")
    // PC1: explains the most variance
    // PC2: explains the 2nd most variance
    // PC3: explains the 3rd most ...
    
    
    preproc <- preProcess(log10(spam[,-58]+1)), method="pca", pcaComp=2) // pcaComp=2 number of PCs to compute
    spamPC <- predict(preproc, log10(spam[,-58]+1))
    plot(spamPC[,1], spamPC[,2], col=typeColor)
    
    
    // train a model using just the PCs
    preProc <- preProcess(log10(training[,-58]+1)), method="pca", pcaComp=2) // pcaComp=2 number of PCs to compute
    trainPC <- predict(preProc, log10(training[,-58]+1))
    modelFit <- train(training$type ~ ., method="glm", data=trainPC)
    
    
    // take PCs calc'ed from training set
    testPC <- predict(preProc, log10(testing[,-58]+1))
    pred <- predict(modelFit, testPC)
    confusionMatrix(testing$type, pred)
    
    
    // convenience using train():
    modelFit <- train(type ~ ., method="glm", preProcess="pca", data=training)
    confusionMatrix( testing$type, predict(modelFit, testing) )



## Quiz #2 


    (3)
    il.cols <- grep("^IL", names(training))
    training.il <- training[,c(1,il.cols)]
    pp <- preProcess( training.il[,-1], method="pca", thresh=0.8 )
    pp
    
    
    (4)
    // build model based on IL* cols 
    m1 <- train( diagnosis ~ ., data=training.il, method="glm" )
    m1.preds <- predict(m1, newdata=testing)
    confusionMatrix( m1.preds, testing$diagnosis )
    
    
    // build model based on PCs of IL* cols
    pp <- preProcess( training.il[,-1], method="pca", thresh=0.8 )
    train.pc <- predict(pp, training.il[,-1])
    m2 <- train( training.il$diagnosis ~ ., data=train.pc, method="glm" )
    
    
    // apply PCA from training data to testing data
    test.pc <- predict(pp, testing.il[,-1])
    m2.preds <- predict(m2, test.pc)
    confusionMatrix(m2.preds, testing.il$diagnosis )
    




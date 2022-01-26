# Load libraries needed.
library(tidyverse)
library(psych)
library(robustHD)
library(praznik)
library(randomForest)
library(Hmisc)
library(MASS)
library(dplyr)
library(plyr)

# Import the library and view the dataset.
library(aplore3)
data("myopia")
view(myopia)

# Summary of the dataset.
summary(myopia)
dim(myopia)

# New dataset without the uneeded values.
myopia1 <- myopia
myopia1$id <- c()
myopia1$studyyear <- c()
myopia1$spheq <- c()

# Change variables to binary. 
myopia1$myopic <- revalue(myopia1$myopic, c("Yes"=1))
myopia1$myopic <- revalue(myopia1$myopic, c("No"=0))
head(myopia1$myopic)
myopia1$myopic <- myopia1$myopic - 1

# Convert back into a numeric type argument (real or integer).
for (i in 1:15) myopia1[,i]<-as.numeric(myopia1[,i])

# Create new dataset to store the standardised data.
myopiastnd <- myopia1

# Data normalisation using Z-score.
for (i in 2:15)
  myopiastnd[,i]<-robustHD::standardize(myopiastnd[,i])
describe(myopiastnd)

# Runing the Shapiro-Wilk normality test on every column.
shapiro.test(myopia1$myopic)
shapiro.test(myopia1$age)
shapiro.test(myopia1$gender)
shapiro.test(myopia1$al)
shapiro.test(myopia1$acd)
shapiro.test(myopia1$lt)
shapiro.test(myopia1$vcd)
shapiro.test(myopia1$sporthr)
shapiro.test(myopia1$readhr)
shapiro.test(myopia1$comphr)
shapiro.test(myopia1$tvhr)
shapiro.test(myopia1$diopterhr)
shapiro.test(myopia1$mommy)
shapiro.test(myopia1$dadmy)

# Finding lambda.
transf <- boxcox(myopia1$comphr ~ myopia1$age)
lambda <- transf$x[which.max(transf$y)]

# Box-Cox transformation.
reg1 <- lm(myopia1$comphr~myopia1$age)
transf <- boxcox(myopia1$comphr~myopia1$age)
lambda <- transf$x[which.max(transf$y)]
reg2 <- lm(((myopia1$comphr^lambda-1)/lambda)~myopia1$age)

# Q-Q plots.
par(mfrow = c(1,2))
qqnorm(reg1$residuals)
qqline(reg1$residuals)
qqnorm(reg2$residuals)
qqline(reg2$residuals)

# Splitting the data.
set.seed(134)
ind <- sample(2, nrow(myopiastnd), prob = c(0.7, 0.3), replace = TRUE)
train.data <- myopiastnd[ind == 1,]
test.data <- myopiastnd[ind == 2,]

# Create the functions to be used in feature selection. 
FSCR = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X1<- X[which(Y==0),i]
    X2<- X[which(Y==1),i]
    mu1<- mean(X1); mu2<- mean(X2); mu<- mean(X[,i])
    var1<- var(X1); var2<- var(X2)
    n1<- length(X1); n2<- length(X2)
    J[i]<- (n1*(mu1-mu)^2+n2*(mu2-mu)^2)/(n1*var1+n2*var2)
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

TSCR = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X1<- X[which(Y==0),i]
    X2<- X[which(Y==1),i]
    mu1<- mean(X1); mu2<- mean(X2)
    var1<- var(X1); var2<- var(X2)
    n1<- length(X1); n2<- length(X2)
    J[i]<- (mu1-mu2)/sqrt(var1/n1+var2/n2)
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

WLCX = function(X, Y, k) # X - matrix with predictors, Y - binary outcome, k top candidates
{
  J<- rep(NA, ncol(X))
  names(J)<- colnames(X)
  for (i in 1:ncol(X))
  {
    X_rank<- apply(data.matrix(X[,i]), 2, function(c) rank(c))
    X1_rank<- X_rank[which(Y==0)]
    X2_rank<- X_rank[which(Y==1)]
    mu1<- mean(X1_rank); mu2<- mean(X2_rank); mu<- mean(X_rank)
    n1<- length(X1_rank); n2<- length(X2_rank); N<- length(X_rank)
    num<- (n1*(mu1-mu)^2+ n2*(mu2-mu)^2)
    denom<- 0
    for (j in 1:n1)
      denom<- denom+(X1_rank[j]-mu)^2
    for (j in 1:n2)
      denom<- denom+(X2_rank[j]-mu)^2
    J[i]<- (N-1)*num/denom
  }
  J<- sort(J, decreasing=TRUE)[1:k]
  return(list(score=J))
}

# Feature selection on standardised training set.
Data_X<- train.data[,2:15]
Data_Y<- train.data[,1]
K<- 3

FSCR(Data_X, Data_Y, K)
TSCR(Data_X, Data_Y, K)
WLCX(Data_X, Data_Y, K)

# Create new dataset to store the top 3 features of the TSCR test.
reduced.train <- train.data[c(11,14,15)]
reduced.test <- test.data[c(11,14,15)]

# Logistic Regression for training set.
training_logit <- glm(myopic~., data=train.data, family="binomial")
summary(training_logit)
# Odds Ratios
exp(training_logit$coefficients)
# Confidence intervals around ORs 
exp(confint(training_logit))

# Random Forest for training set.
training_forest <- randomForest(factor(myopic)~., data=train.data, importance=TRUE, ntree=1000, mtry=3, replace=TRUE)
importance(training_forest)[,c(3,4)]

# Logistic Regression for reduced training set.
reduced_logit <- glm(myopic~., data=reduced.train, family="binomial")
summary(reduced_logit)
# Odds Ratios
exp(reduced_logit$coefficients)
# Confidence intervals around ORs 
exp(confint(reduced_logit))

# Random Forest for reduced training set.
reduced_forest <- randomForest(factor(myopic)~., data=reduced.train, importance=TRUE, ntree=1000, mtry=3, replace=TRUE)
importance(reduced_forest)[,c(3,4)]

# Applying the Logistic Regression models to the test set, and the reduced test set.
prGLM1<- predict(training_logit, newdata=test.data, type="response")
prGLM1
prGLM2<- predict(reduced_logit, newdata=reduced.test, type="response")
# Evaluate the performance
somers2(prGLM1, test.data$myopic)
somers2(prGLM2, reduced.test$myopic)
# Comparison of predicted values to observed values
cbind(test.data$myopic, round(prGLM1, digits=3)>0.5)
cbind(reduced.test$myopic, round(prGLM2, digits=3)>0.5)

# Applying the Random Forest models to the test set, and the reduced test set. 
prRF1<- predict(training_forest, newdata=test.data, type="response")
prRF1
prRF2<- predict(reduced_forest, newdata=reduced.test, type="response")
prRF2

# Performance tables for the models.
table(test.data$myopic, prGLM1>0.5)
table(reduced.test$myopic, prGLM2>0.5)
table(test.data$myopic, prRF1)
table(reduced.test$myopic, prRF2)



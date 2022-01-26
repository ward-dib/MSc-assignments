# Import the library and view the dataset.
library(MASS)
data(birthwt)
view(birthwt)

# Summary of the dataset.
summary(birthwt)
dim(birthwt)
sum(is.na(birthwt))

# Remove birth weight in grams from the dataset.
birthwt1 <- birthwt
birthwt1$bwt <- c()

# Create new dataset to store the standardised data.
birthstnd <- birthwt1

# Data normalisation using Z-score.
library(robustHD)
for (i in 2:9)
  birthstnd[,i]<-robustHD::standardize(birthstnd[,i])
library(psych)
describe(birthstnd[, c(1:9)])

# Runing the Shapiro-Wilk normality test on every column.
shapiro.test(birthwt1$low)
shapiro.test(birthwt1$age)
shapiro.test(birthwt1$lwt)
shapiro.test(birthwt1$race)
shapiro.test(birthwt1$smoke)
shapiro.test(birthwt1$ptl)
shapiro.test(birthwt1$ht)
shapiro.test(birthwt1$ui)
shapiro.test(birthwt1$ftv)

# Finding lambda.
library(MASS)
transf <- boxcox(birthwt1$lwt ~ birthwt1$age)
lambda <- transf$x[which.max(transf$y)]

# Box-Cox transformation performed on the standardised dataset.
reg1 <- lm(birthwt1$lwt~birthwt1$age)
transf <- boxcox(birthwt1$lwt~birthwt1$age)
lambda <- transf$x[which.max(transf$y)]
reg2 <- lm(((birthwt1$lwt^lambda-1)/lambda)~birthwt1$age)

# Q-Q plots.
par(mfrow = c(1,2))
qqnorm(reg1$residuals)
qqline(reg1$residuals)
qqnorm(reg2$residuals)
qqline(reg2$residuals)

# Shapiro testing again on transformed values.
shapiro.test((birthwt1$lwt^lambda-1)/lambda)

# Splitting the data.
set.seed(134)
ind <- sample(2, nrow(birthstnd), prob = c(0.7, 0.3), replace = TRUE)
train.data <- birthstnd[ind == 1,]
test.data <- birthstnd[ind == 2,]

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
library(praznik)
Data_X<- train.data[,2:9]
Data_Y<- train.data[,1]
K<- 3

FSCR(Data_X, Data_Y, K)
TSCR(Data_X, Data_Y, K)
WLCX(Data_X, Data_Y, K)

# Create new dataset to store the top 3 features of the TSCR test.
reduced.train <- train.data[c(1,9,2:3)]
reduced.test <- test.data[c(1,9,2:3)]

# Logistic Regression for training set.
training_logit <- glm(low~., data=train.data, family="binomial")
summary(training_logit)
# Odds Ratios.
exp(training_logit$coefficients)
# Confidence intervals around ORs. 
exp(confint(training_logit))

# Logistic Regression for reduced training set.
reduced_logit <- glm(low~., data=reduced.train, family="binomial")
summary(reduced_logit)
# Odds Ratios
exp(reduced_logit$coefficients)
# Confidence intervals around ORs 
exp(confint(reduced_logit))

# Random Forest for training set.
library(randomForest)
training_forest <- randomForest(factor(low)~., data=train.data, importance=TRUE, ntree=1000, mtry=3, replace=TRUE)
importance(training_forest)[,c(3,4)]

# Random Forest for reduced training set.
reduced_forest <- randomForest(factor(low)~., data=reduced.train, importance=TRUE, ntree=1000, mtry=3, replace=TRUE)
importance(reduced_forest)[,c(3,4)]

# Applying the Logistic Regression models to the test set, and the reduced test set.
prGLM1<- predict(training_logit, newdata=test.data, type="response")
prGLM1
prGLM2<- predict(reduced_logit, newdata=reduced.test, type="response")

# Comparison of predicted values to observed values
cbind(test.data$low, round(prGLM1, digits=3)>0.5)
cbind(reduced.test$low, round(prGLM2, digits=3)>0.5)

# Applying the Random Forest models to the test set, and the reduced test set. 
prRF1<- predict(training_forest, newdata=test.data, type="response")
prRF1
prRF2<- predict(reduced_forest, newdata=reduced.test, type="response")
prRF2

# Performance tables for the models.
table(test.data$low, prGLM1>0.5)
table(reduced.test$low, prGLM2>0.5)
table(test.data$low, prRF1)
table(reduced.test$low, prRF2)

100*prop.table(table(prGLM1>0.1, test.data$low), 2)
100*prop.table(table(prGLM1>0.3, test.data$low), 2)
100*prop.table(table(prGLM1>0.5, test.data$low), 2)
100*prop.table(table(prGLM1>0.7, test.data$low), 2)
100*prop.table(table(prGLM1>0.9, test.data$low), 2)

table(prGLM2>0.5, test.data$low)
100*prop.table(table(prGLM2>0.5, test.data$low), 2)

# ROC curves for test set and.
pred1<- prediction(predictions = prGLM1, labels=test.data$low)
pred2<- prediction(predictions = prGLM2, labels=reduced.test$low)
perf1<- performance(pred1, measure="tpr", x.measure = "fpr")
perf2<- performance(pred2, measure="tpr", x.measure = "fpr")

perf1@x.values; perf1@y.values; perf1@alpha.values

plot(perf1, col="red")
plot(perf2, col="blue", add=TRUE)

legend(0.65, 0.3, legend=c("Test Set", "Reduced Test Set"),
       col=c("red", "blue"), lty=1:1, cex=0.8, 
       text.font=4, bg='#f0f0f0')

# AUC value to evaluate performance.
library(Hmisc)
somers2(prGLM1, test.data$low)
somers2(prGLM2, reduced.test$low)

# Calibration test.
library(gbm)
library(ResourceSelection)
hoslem.test(test.data$low, prGLM1)
hoslem.test(reduced.test$low, prGLM2)

par(mfrow=c(2,1))
calibrate.plot(test.data$low, prGLM1, main="full model")
calibrate.plot(reduced.test$low, prGLM2, main="reduced model")


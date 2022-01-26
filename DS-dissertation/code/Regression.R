# Remove C19 unwanted columns.
C19$ID <- c()
C19$Sex2 <- c()


# Splitting the data.
set.seed(143)
ind <- sample(2, nrow(C19), prob = c(0.5, 0.5), replace = TRUE)
train.data <- C19[ind == 1,]
test.data <- C19[ind == 2,]


# Logistic Regression for training set.
training_logit <- glm(Death~., data=train.data, family="binomial")
summary(training_logit)
# Odds Ratios.
exp(training_logit$coefficients)
# Confidence intervals around ORs. 
exp(confint(training_logit))

# Applying the Logistic Regression models to the test set.
prGLM1<- predict(training_logit, newdata=test.data, type="response")
prGLM1

# ROC and AUC value to evaluate performance.
library(pROC)
lrROC <- roc(test.data$Death, prGLM1)
plot(lrROC, col="red", main="Logistic Regression")
auc(lrROC)


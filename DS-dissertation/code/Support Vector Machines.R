# Remove C19 unwanted columns.
C19$ID <- c()
C19$Sex2 <- c()


# Splitting the data.
set.seed(143)
ind <- sample(2, nrow(C19), prob = c(0.5, 0.5), replace = TRUE)
train.data <- C19[ind == 1,]
test.data <- C19[ind == 2,]

library(e1071)
library(caret)
library(pROC)

SVMmodel <- svm(Death ~ ., data =train.data)
svm_prediction <- predict(SVMmodel, test.data)

svmROC <- roc(test.data$Death, svm_prediction)
plot(svmROC, col = "blue", main = "Random Forest")

# Area Under Curve (AUC) for each ROC curve (higher -> better)
auc(svmROC)

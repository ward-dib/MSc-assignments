# Remove C19 unwanted columns.
C19$ID <- c()
C19$Sex2 <- c()

# Splitting the data.
set.seed(143)
ind <- sample(2, nrow(C19), prob = c(0.5, 0.5), replace = TRUE)
train.data <- C19[ind == 1,]
test.data <- C19[ind == 2,]


# Random Forest for training set.
library(randomForest)
RFmodel <- randomForest(factor(Death)~., data=train.data, importance=TRUE, ntree=1000, mtry=3, replace=TRUE)
importance(RFmodel)[,c(3,4)]

rf_prediction <- predict(RFmodel, test.data, type = "prob")

# ROC curves
library(pROC)
rfROC <- roc(test.data$Death, rf_prediction[,2])
plot(rfROC, col = "green", main = "Random Forest")

# Area Under Curve (AUC) for each ROC curve (higher -> better)
auc(rfROC)

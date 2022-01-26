# Import all the needed libraries.#
library(pROC)
library(randomForest)
library(e1071)
library(caret)
library(survival)
library(survminer)
library(dplyr)


# Create new dataframes with desired inputs only.
C19A <- C19
C19B <- C19

# Remove C19 unwanted columns.
C19A$ID <- c()
C19A$Sex2 <- c()
C19A$Days <- c()
C19A$ICU <- c()

C19B$ID <- c()

# Splitting the data for predictive modelling.
set.seed(143)
ind <- sample(2, nrow(C19A), prob = c(0.5, 0.5), replace = TRUE)
train.data <- C19A[ind == 1,]
test.data <- C19A[ind == 2,]

#------- Logestic Regression ------- 

# Logistic Regression for training set.
training_logit <- glm(Death~., data=train.data, family="binomial")
summary(training_logit)

# Odds Ratios.
exp(training_logit$coefficients)
# Confidence intervals around ORs. 
exp(confint(training_logit))

# Applying the Logistic Regression models to the test set.
LRmodel <- predict(training_logit, newdata=test.data, type="response")

#------- Random Forest -------

# Random Forest for training set.
RFmodel <- randomForest(factor(Death)~., data=train.data, importance=TRUE, ntree=1000, mtry=3, replace=TRUE)
importance(RFmodel)[,c(3,4)]
rf_prediction <- predict(RFmodel, test.data, type = "prob")

#------- Support Vector Machines ------- 

SVMmodel <- svm(Death ~ ., data =train.data)
svm_prediction <- predict(SVMmodel, test.data)

# ------ ROC and AUC value to evaluate performance ------

# ROC
lrROC <- roc(test.data$Death, LRmodel)
plot(lrROC, col="red", main="ROC curves")

rfROC <- roc(test.data$Death, rf_prediction[,2])
plot(rfROC, add=TRUE, col = "green")

svmROC <- roc(test.data$Death, svm_prediction)
plot(svmROC, add=TRUE, col = "blue")

legend(0.4, 0.2, legend=c("Logistic Regression", "Random Forest", "SVM", "ANN"),
       col=c("red", "green", "blue", "orange"), lty=1:1, cex=0.8)

# AUC
auc(lrROC)
auc(rfROC)
auc(svmROC)

# Confidence Intervals for each model.
ci.auc(lrROC)
ci.auc(rfROC)
ci.auc(svmROC)

# Sensitivities at fixed specificity.
ci.se(lrROC, specificities=c(0.9))
ci.se(rfROC, specificities=c(0.9))
ci.se(svmROC, specificities=c(0.9))


#------- Prognostic Analysis -------

# Histogram to split data by age.
hist(C19B$Age) 
C19B <- C19B %>% mutate(age_group = ifelse(C19B$Age >=50, "Above 50", "Under 50"))
C19B$age_group <- factor(C19B$age_group)


# Fit survival data using the Kaplan-Meier method.
surv_object <- Surv(time = C19B$Days, event = C19B$Death)
surv_object 

fit1 <- survfit(surv_object ~ age_group, data = C19B)
ggsurvplot(fit1, data = C19B, pval = TRUE)

fit2 <- survfit(surv_object ~ chroniccard_mhyn, data = C19B)
ggsurvplot(fit2, data = C19B, pval = TRUE)

fit3 <- survfit(surv_object ~ hypertension_mhyn, data = C19B)
ggsurvplot(fit3, data = C19B, pval = TRUE)

fit4 <- survfit(surv_object ~ chronicpul_mhyn, data = C19B)
ggsurvplot(fit4, data = C19B, pval = TRUE)

fit5 <- survfit(surv_object ~ renal_mhyn, data = C19B)
ggsurvplot(fit5, data = C19B, pval = TRUE)

fit6 <- survfit(surv_object ~ chronicneu_mhyn, data = C19B)
ggsurvplot(fit6, data = C19B, pval = TRUE)

fit7 <- survfit(surv_object ~ diabetes_mhyn_2, data = C19B)
ggsurvplot(fit7, data = C19B, pval = TRUE)

fit8 <- survfit(surv_object ~ Sex2, data = C19B)
ggsurvplot(fit8, data = C19B, pval = TRUE)


fit <- survfit(Surv(time = C19B$Days, event = C19B$Death) ~ Sex2, data = C19B)
ggsurvplot(fit,
           pval = TRUE, conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           ggtheme = theme_bw(), # Change ggplot2 theme
           palette = c("#E7B800", "#2E9FDF"))

#------- Cox Regression ------- 

fit.coxph <- coxph(surv_object ~ chroniccard_mhyn +
                     hypertension_mhyn + renal_mhyn + diabetes_mhyn_2 + chronicneu_mhyn, data = C19B)
ggforest(fit.coxph, data = C19B)


#--------- End ---------



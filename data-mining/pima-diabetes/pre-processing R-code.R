# Retrieve dataset from library.
library(mlbench)
data("PimaIndiansDiabetes")
View(PimaIndiansDiabetes)

# Full summary of data.
rm(list = ls(all = TRUE))
library(mlbench)
data("PimaIndiansDiabetes")
summary(PimaIndiansDiabetes)

# Contingency table for pregnancy values.
table(PimaIndiansDiabetes$pregnant, PimaIndiansDiabetes$diabetes)

# Histograms of all the columns.
par(mfrow=c(1,3))
hist(PimaIndiansDiabetes$pregnant, main = "Number of Pregnancies",
     col = c("#bcbddc"))
hist(PimaIndiansDiabetes$glucose, main = "Glucose Levels", col = c("#bcbddc"))
hist(PimaIndiansDiabetes$pressure, main = "Diastolic Pressure",
     col = c("#bcbddc"))
hist(PimaIndiansDiabetes$triceps, main = "Tricep Skin Thickness",
     col = c("#bcbddc"))
hist(PimaIndiansDiabetes$insulin, main = "Insulin Levels",
     col = c("#bcbddc"))
hist(PimaIndiansDiabetes$mass, main = "BMI", col = c("#bcbddc"))
hist(PimaIndiansDiabetes$pedigree, main = "Pedigree function",
     col = c("#bcbddc"))
hist(PimaIndiansDiabetes$age, main = "Age in years", col = c("#bcbddc"))

# Min-Max ranges for all columns.
for (i in names(PimaIndiansDiabetes)) {
  s <- sprintf("data filed %s range: [%s-%s]", 
               i, range(PimaIndiansDiabetes[,i])[1], 
               range(PimaIndiansDiabetes[,i])[2])
  print(s)
}

# IQR range calculations.

iqr <- IQR(PimaIndiansDiabetes$pregnant)
Q1 <- quantile(PimaIndiansDiabetes$pregnant, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$pregnant, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$glucose)
Q1 <- quantile(PimaIndiansDiabetes$glucose, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$glucose, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$pressure)
Q1 <- quantile(PimaIndiansDiabetes$pressure, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$pressure, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$triceps)
Q1 <- quantile(PimaIndiansDiabetes$triceps, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$triceps, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$insulin)
Q1 <- quantile(PimaIndiansDiabetes$insulin, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$insulin, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$mass)
Q1 <- quantile(PimaIndiansDiabetes$mass, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$mass, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$pedigree)
Q1 <- quantile(PimaIndiansDiabetes$pedigree, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$pedigree, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

iqr <- IQR(PimaIndiansDiabetes$age)
Q1 <- quantile(PimaIndiansDiabetes$age, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$age, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

# Age value 81 summary.
PimaIndiansDiabetes[460,]

# Create new dataset to store cleaned data with no missing values or outliers.
Pima_NA<-PimaIndiansDiabetes
View(Pima_NA)

# Change specific missing values or outlier ranges to NA. 
class(Pima_NA$pregnant)
Pima_NA$pregnant[Pima_NA$pregnant>13.5] <- "NA"

class(Pima_NA$glucose)
Pima_NA$glucose[Pima_NA$glucose %in% c("0", "<37.125", ">202.125")] <- "NA"

class(Pima_NA$pressure)
Pima_NA$pressure[Pima_NA$pressure %in% c("0", "<35", ">107")] <- "NA"

class(Pima_NA$triceps)
Pima_NA$triceps[Pima_NA$triceps %in% c("0", ">80")] <- "NA"

class(Pima_NA$insulin)
Pima_NA$insulin[Pima_NA$insulin %in% c("0", ">318.125")] <- "NA"

class(Pima_NA$mass)
Pima_NA$mass[Pima_NA$mass %in% c("0", "<13.35", ">50.55")] <- "NA"

class(Pima_NA$pedigree)
Pima_NA$pedigree[Pima_NA$pedigree %in% c("<-0.33", ">1.20")] <- "NA"

# Convert back into a numeric type argument (real or integer).
for (i in 1:8) Pima_NA[,i]<-as.numeric(Pima_NA[,i])

# Summarise the new dataset.
summary(Pima_NA)

# Create and view new dataset for mean imputation.
Pima_mean_imp<- Pima_NA
View(Pima_mean_imp)

# Mean imputation of missing values and outliers.
for (i in 1:7)
  Pima_mean_imp[is.na(Pima_NA[,i]), i]<- mean(Pima_NA[,i], na.rm=TRUE)

# Summary of imputed data.
summary(Pima_mean_imp)

# Create new dataset to store the standardised data.
Pima_stand<- Pima_mean_imp
View(Pima_stand)

# Data normalisation using Z-score.
library(robustHD)
for (i in 1:8)
  Pima_stand[,i]<-robustHD::standardize(Pima_stand[,i])

# Download packages needed for this function.
install.packages("psych")
library(psych)

# Summary of both datasets to compare the changes overall.
describe(Pima_mean_imp)
describe(Pima_stand)

# Or, for continuous columns only.
describe(Pima_mean_imp[, c(1:8)])
describe(Pima_stand[, c(1:8)])

# Runing the Shapiro-Wilk normality test on every column.
shapiro.test(Pima_NA$pregnant)
shapiro.test(Pima_NA$glucose)
shapiro.test(Pima_NA$pressure)
shapiro.test(Pima_NA$triceps)
shapiro.test(Pima_NA$insulin)
shapiro.test(Pima_NA$mass)
shapiro.test(Pima_NA$pedigree)
shapiro.test(Pima_NA$age)

# Finding lambda.
library(MASS)
transf <- boxcox(Pima_NA$glucose ~ Pima_NA$pressure)
lambda <- transf$x[which.max(transf$y)]


# Box-Cox transformation performed on dataset with no zeros or negatives.
library(MASS)
reg1 <- lm(Pima_NA$glucose~Pima_NA$pressure)
transf <- boxcox(Pima_NA$glucose~Pima_NA$pressure)
lambda <- transf$x[which.max(transf$y)]
reg2 <- lm(((Pima_NA$glucose^lambda-1)/lambda)~Pima_NA$pressure)

# Q-Q plots.
par(mfrow = c(1,2))
qqnorm(reg1$residuals)
qqline(reg1$residuals)
qqnorm(reg2$residuals)
qqline(reg2$residuals)

# Shapiro-Wilk test on transformed values.
shapiro.test((Pima_NA$glucose^lambda-1)/lambda)

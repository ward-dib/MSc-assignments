# Load required packages
library(survival)
library(survminer)
library(dplyr)

# Remove C19 unwanted columns.
C19$ID <- c()
C19$Sex2 <- c()

# Data seems to be bimodal
hist(C19$Age) 

C19 <- C19 %>% mutate(age_group = ifelse(C19$Age >=50, "old", "young"))
C19$age_group <- factor(C19$age_group)

# Fit survival data using the Kaplan-Meier method
surv_object <- Surv(time = C19$Days, event = C19$Death)
surv_object 

fit1 <- survfit(surv_object ~ ICU, data = C19)
summary(fit1)

ggsurvplot(fit1, data = C19, pval = TRUE)

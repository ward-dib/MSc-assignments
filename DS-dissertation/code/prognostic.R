# Load required packages
library(survival)
library(survminer)
library(dplyr)

# Remove C19 unwanted columns.
C19$ID <- c()

# Data seems to be bimodal
hist(C19$Age) 

C19 <- C19 %>% mutate(age_group = ifelse(C19$Age >=50, "old", "young"))
C19$age_group <- factor(C19$age_group)

# Fit survival data using the Kaplan-Meier method
surv_object <- Surv(time = C19$Days, event = C19$Death)
surv_object 

fit1 <- survfit(surv_object ~ age_group, data = C19)
summary(fit1)

ggsurvplot(fit1, data = C19, pval = TRUE)


fit1 <- survfit(surv_object ~ Sex2, data = C19)
summary(fit1)

ggsurvplot(fit1, data = C19, pval = TRUE)


fit <- survfit(Surv(time = C19$Days, event = C19$Death) ~ Sex2, data = C19)
ggsurvplot(fit,
           pval = TRUE, conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           ggtheme = theme_bw(), # Change ggplot2 theme
           palette = c("#E7B800", "#2E9FDF"))


fit1 <- survfit(Surv(time = C19$Days, event = C19$Death) ~ age_group, data = C19)
ggsurvplot(fit1,
           pval = TRUE, conf.int = TRUE,
           risk.table = TRUE, # Add risk table
           risk.table.col = "strata", # Change risk table color by groups
           linetype = "strata", # Change line type by groups
           ggtheme = theme_bw(), # Change ggplot2 theme
           palette = c("#E7B800", "#2E9FDF"))


# Fit a Cox proportional hazards model
fit.coxph <- coxph(surv_object ~ age_group + Sex2, 
                   data = C19)
ggforest(fit.coxph, data = C19)


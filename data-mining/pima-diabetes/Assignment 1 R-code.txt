# data preprocessing steps

# retrieve dataset from library
library(mlbench)
data(PimaIndiansDiabetes)
View(PimaIndiansDiabetes)


# full summary of data
rm(list = ls(all = TRUE))
library(mlbench)
data("PimaIndiansDiabetes")
summary(PimaIndiansDiabetes)

  
# contingency table to compare values
table(PimaIndiansDiabetes$pregnant, PimaIndiansDiabetes$diabetes)


# summary by group/column
>
by(PimaIndiansDiabetes, PimaIndiansDiabetes$pregnant, summary)


# summarising data
>
Hmisc::describe(PimaIndiansDiabetes)


# summarising data
>
pastecs::stat.desc(PimaIndiansDiabetes)


# visualising data
# retrieve functions libararies
>
library(jsonlite)
library(ggpubr)


# boxplotting to identify anomalies
>
ggpubr::ggsummarystats(PimaIndiansDiabetes, x = "diabetes", y = "pregnant", ggfunc = ggboxplot, add = "jitter", color = "diabetes", palette = "npg")
 

# violin plot
>
ggpubr::ggsummarystats(PimaIndiansDiabetes, x = ”diabetes”, y = ”pregnant”, ggfunc = ggviolin, add = ”jitter”, color = ”diabetes”, palette = ”npg”)



# IQR-based search
iqr <- IQR(PimaIndiansDiabetes$pregnant) # IQR range
Q1 <- quantile(PimaIndiansDiabetes$pregnant, 0.25)
Q3 <- quantile(PimaIndiansDiabetes$pregnant, 0.75)
as.numeric(c(Q1-1.5*iqr, Q3+1.5*iqr))

summary(PimaIndiansDiabetes$pregnant)

# find out which entries r said outliers
>
which(PimaIndiansDiabetes$pregnant>13.5)

# get info about the entries
>
PimaIndiansDiabetes[c(89, 160, 299, 456),]

# pick some columns
>
PimaIndiansDiabetes <- PimaIndiansDiabetes[,c(2:6)]


# set values = NA
>
class(PimaIndiansDiabetes$pregnant)
# replacing specific value
PimaIndiansDiabetes$pregnant[PimaIndiansDiabetes$pregnant == "0"] ＜- "NA"
# replacing a specified number of values
PimaIndiansDiabetes$pregnant[PimaIndiansDiabetes$pregnant %in% c("17","15","14")] ＜- "NA"
# replacing a range of values less or more than x
PimaIndiansDiabetes$pregnant[PimaIndiansDiabetes$pregnant>13.5] ＜- "NA"




-----------------

# RAWS

> library(robustHD)
Loading required package: ggplot2
Loading required package: perry
Loading required package: parallel
Loading required package: robustbase
> summary(PimaIndiansDiabetes$pregnant)
   Min. 1st Qu.  Median    Mean 
  0.000   1.000   3.000   3.845 
3rd Qu.    Max. 
  6.000  17.000 
> # since the mean and median are close this means the data could be following normal destrubution
> summary(standardize(PimaIndiansDiabetes$pregnant))
   Min. 1st Qu.  Median    Mean 3rd Qu. 
-1.1411 -0.8443 -0.2508  0.0000  0.6395 
   Max. 
 3.9040 
> # the rule of 2.5
> which(standardize(PimaIndiansDiabetes$pregnant)>2.5)
 [1]  29  73  87  89 160 275 299 324 358 456
[11] 519 636 692 745
> PimaIndiansDiabetes[29,]
   pregnant glucose pressure triceps
29       13     145       82      19
   insulin mass pedigree age diabetes
29     110 22.2    0.245  57      neg
> # we'll use mean_imp to compare changes
> summary(mean_imp$pregnant)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
  0.000   1.000   3.000   3.787   6.000  13.000 
> # mean reduced but is still close to median
> summary(standardize(mean_imp$pregnant))
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
-1.1579 -0.8521 -0.2406  0.0000  0.6768  2.8174 
> # the max reduced
> which(standardize(mean_imp$pregnant)>2.5)
 [1]  29  73  87 216 255 275 324 334 358 359 376 437 511 519 583 636 692 745 746
> # seems that all the values equal to 13 pregnancies are reading as outliers. this could be because theyre close to the iqr cut off of 13.5.


----------------









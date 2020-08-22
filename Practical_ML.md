---
title: "Assignment"
author: "Wei Ken Len"
date: "22 Aug 2020"
output: 
  html_document: 
    keep_md: yes
---


### Link to github repo: https://github.com/lenweiken/courserapractical_ML_new

### The objective, as stated in the instructions of this project assignment, is to predict the manner in which this exercise is carried out ("classe" variable) in the training set. After training the model to predict the classe variable, the prediction model will be applied onto the testing dataset. 

### Broadly speaking, I will be carrying this assignment out in the following manner:
#### 1. Setup my libraries and read data
#### 2. Peruse the training data and perform necessary cleansing (including feature selection)
#### 3. Training my model (which involves partitioning my training dataset into 2, so that I can verify the accuracy before applying my model onto the testing data set)
#### 4. Applying my prediction model on the "validation set" to estimate the accuracy
#### 5. If results of validation are sufficient, to proceed with applying the prediction model onto the test set. If not sufficient, I will attempt other prediction models. 


# Step 1: Setup the data and libraries required to conduct the analyses

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(tidyr)
library(plyr)
library(dplyr)
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:plyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(tibble)
library(ggplot2)
library(foreach)
library(iterators)
library(parallel)
library(doParallel)
```


```r
training1 <- read.csv("pml-training.csv")
testing1 <- read.csv("pml-testing.csv")
```


# Step 2 - Check and clean data

## Cursory check on training data reveals key issues:


```r
###  Training data Blank / NA / #DIV/0! as data values
###  Variables that are not expected to contribute to prediction models (i.e., variables with no expected predicting power such as timestamps, columns with all blank vlaues) )

## To standardise the way data captures blank or errors, i.e. #DIV/0! , we replace all these with NA. We do the same for the testing data to ensure consistency by re-reading the data into our environment whilst simultaneously setting errors or blanks to NA 
training <- read.csv("pml-training.csv", na.strings = c("#DIV/0!", ""))
testing <- read.csv("pml-testing.csv", na.strings = c("#DIV/0!", ""))
```


```r
## To remove columns with high amount of NAs / all NAs, we first identify the number of code
allcolumns_nu_na <- as.data.frame(colSums(is.na(training)))
allcolumns_nu_na <- rownames_to_column(allcolumns_nu_na, var = "Variables")
columns_na <- filter(allcolumns_nu_na, allcolumns_nu_na$`colSums(is.na(training))`!=0)
min(columns_na$`colSums(is.na(training))`)
nrow(training)
```
#### We can note that there are 33 variables which have high % of missing values (at least ~97.9%), which can be removed as part of feature selection

```r
list_columnsna <- columns_na[,1]
training <- select(training,-c(list_columnsna))
```

```
## Note: Using an external vector in selections is ambiguous.
## ℹ Use `all_of(list_columnsna)` instead of `list_columnsna` to silence this message.
## ℹ See <https://tidyselect.r-lib.org/reference/faq-external-vector.html>.
## This message is displayed once per session.
```

```r
testing <- select(testing,-c(list_columnsna))
```


```r
## To remove columns with near zero variances, we apply the nearzerovar function

near_zerovar_remove <- nearZeroVar(training)

### Before applying the same data cleansing principles on the test dataset, we check if column names are the same for both training & test (WITHOUT looking at the test data itself)
check_nzv_train <- training[0,near_zerovar_remove]
check_nzv_test <- testing[0,near_zerovar_remove]

training <- training[ , -near_zerovar_remove]
testing <- testing[ , -near_zerovar_remove]
```

#### To remove first 7 columns where it does not logically affect prediction, and check if first 7 columns in testing data has the same header(Without looking at other columns or data in first 7 columns), then apply pre-processing to both data

```r
training[0,c(1,2,3,4,5,6,7)]
testing[0,c(1,2,3,4,5,6,7)]
training <- select(training,-c(1,2,3,4,5,6,7))
testing <- select(testing,-c(1,2,3,4,5,6,7))
```

#### To ensure we know what we are predicting, we first check on nature of the variable we want to predict. 
#### We confirm that there are 5 classes (A,B,C,D,E) for which to predict for

```r
unique(training$classe)
class(training$classe)
### We also change the the class of the variable from character to factor to simplify downstream analysis and visualizations (if requried)
training$classe <- as.factor(training$classe)
```

# Step 3 - Training the model
#### We start off with Random Forest model, which is generally a reliable model for classification (which is what's needed in this case). Note that we run into some minor problems with computing speed but we managed to apply a workaround based on Len greski's suggestions (see below)

```r
## First, we separate training set into two (training & validation) so that we can get an estimate of the out-of-sample error

inTrain <- createDataPartition(training$classe, p = 3/4, list = FALSE)

training2 <- training[ inTrain,]

validation <- training[-inTrain,]

## We then train our model, starting with Random Forest. Our first pass at this yielded unsatisfactory results (long processing times) , primarily due to the bootstrapping method in Random Forest. We apply the suggestions by Len Greski to improve the perforamnce of model training (using cross validation method and parallel processing)

y <- training2[,52] # set up x and y to avoid slowness of caret() with model syntax
x <- training2[,-52]
```



```r
cluster <- makeCluster(detectCores() - 1)  # Parallel processing to quicken training
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv", # Cross validation is used
                           number = 5,
                           allowParallel = TRUE)

fit_rf <- train(x,y,  method="rf",data=training2,trControl = fitControl)

stopCluster(cluster)
registerDoSEQ()
```


# Step 4. Apply prediction model to validation dataset to estimate out of sample error
#### Based on our results below, we find that our accuracy for our validation data is approximately ~99% (implying our out of sample error is would be very low), which is a sufficient result to proceed with applying the model onto the test dataset in step 5

```r
predict_rf <- predict(fit_rf, newdata = validation)

confusionMatrix(predict_rf, validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1391    7    0    0    0
##          B    2  941    2    1    0
##          C    2    1  850    4    0
##          D    0    0    3  799    3
##          E    0    0    0    0  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9925, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9971   0.9916   0.9942   0.9938   0.9967
## Specificity            0.9980   0.9987   0.9983   0.9985   1.0000
## Pos Pred Value         0.9950   0.9947   0.9918   0.9925   1.0000
## Neg Pred Value         0.9989   0.9980   0.9988   0.9988   0.9993
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2836   0.1919   0.1733   0.1629   0.1831
## Detection Prevalence   0.2851   0.1929   0.1748   0.1642   0.1831
## Balanced Accuracy      0.9976   0.9952   0.9962   0.9962   0.9983
```

# Step 5. Here, we can see the results of the prediction model, which will be used in the quiz portion of this assignment

```r
predict_rf <- predict(fit_rf, newdata = testing)
print(predict_rf)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
##### References: 
###### - https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md
###### - https://stackoverflow.com/questions/52807921/number-of-missing-values-in-each-column-in-r
###### - https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-ghPagesSetup.md
###### - http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

rm(list = ls())

#install required packages
library(dplyr)
library(funModeling)
library(ggplot2)
library(corrgram)
#install.packages('caret', dependencies = TRUE)
library(caret)
library(randomForest)
library(tidyr)
#library(Boruta)
#install.packages("Hmisc")
library(Hmisc)
#install.packages("corrplot")
library(corrplot)
#install.packages("pryr")
library(pryr)
#instal.packages("c50")
library(C50)
#install.packages("PRROC")
#install.packages("pROC")
library(PRROC)
library(pROC)



#Set Working directory
setwd("E:/EDW/Projects/Santander Prediction")
#getwd()

#Load data
train_df <- read.csv("train.csv", header = T)
test_df <- read.csv("test.csv", header = T)

train_df$ID_code <- as.character(train_df$ID_code)
train_df$target <- as.character(train_df$target)
train_df$target <- as.factor(train_df$target)

#EDA - Train data
str(train_df)
glimpse(train_df)  #to view data of all columns
df_1 <- df_status(train_df)  #gives no. of zeros(& %value), na values(& % value), infinity values, type and no of unique entries in each column
freq(train_df$target) #percentage of target variable
train_summary <- profiling_num(train_df)  #Summary of train data
# row.names(train_summary) <- train_summary$variable
# train_summary <- train_summary[-1]
# train_summary <- data.frame(t(train_summary))

#EDA - Test Data
glimpse(test_df)  #to view data of all columns
df_2 <- df_status(test_df)  #gives no. of zeros(& %value), na values(& % value), infinity values, type and no of unique entries in each column
test_summary <- profiling_num(test_df)  #Summary of train data
row.names(test_summary) <- test_summary$variable
test_summary <- test_summary[-1]
test_summary <- data.frame(t(test_summary))

###############################################################################################################
#get the variables that have missing values
var_with_na <- subset(df_1, df_1$q_na > 0)
print("Missing Value details:")
var_with_na["variable"]
###############################################################################################################

#Visualizations:
#Density plots of individual features
ggplot(gather(train_df[,3:30]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')
ggplot(gather(train_df[,31:60]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')
ggplot(gather(train_df[,61:90]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')
ggplot(gather(train_df[,91:120]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')
ggplot(gather(train_df[,121:150]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')
ggplot(gather(train_df[,151:180]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')
ggplot(gather(train_df[,181:202]), aes(value)) + geom_freqpoly() + facet_wrap(~key, scales = 'free_x')

#Distribution of mean,standard deviation and ranges:
ggplot(train_summary, aes(x=train_summary$variable, y=train_summary$std_dev)) +geom_point()
ggplot(train_summary, aes(x=train_summary$variable, y=train_summary$mean)) +geom_point(colour = 'red')
ggplot(train_summary, aes(x=train_summary$variable, y=train_summary$range_98)) +geom_point()

##########################################################################################################
#Numerical data:
numeric_data = train_df[,-c(1,2)]
cnames = colnames(numeric_data)

### Correlation Plot 
corr_data <- cor(numeric_data)
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = corr_data, col = palette, symm = TRUE)
# corr_data_1 <- rcorr(as.matrix(numeric_data))
# corrplot(corr_data_1)
############################################################################################################

##Feature Selection - Correlation using Logistic regression 
numeric_data = train_df[,-1]
#numeric_data$target = as.numeric(numeric_data$target)
fit_log = glm(target~., numeric_data, family = binomial)
v_imp <- varImp(fit_log)
v_imp$Variable <- row.names(v_imp)
v_imp <- v_imp[ , c(2,1)]
v_imp <- v_imp[order(-v_imp$Overall), ]
row.names(v_imp) <- NULL
feature <- v_imp$Variable[1:30]

# ##Feature Selection -Random Forest Classifier
# # numeric_data$target <- as.character(numeric_data$target)
# # numeric_data$target <- as.factor(numeric_data$target)
# # rf_model <- randomForest(target~., data = numeric_data)
# 

#New train data with only selected variables 
train_df_1 <- cbind(train_df[,c(1,2)], train_df[,feature])

############################################################################################################

# ## BoxPlots - Distribution and Outlier Check
numeric_index = sapply(train_df_1,is.numeric) #selecting only numeric

numeric_data = train_df_1[,numeric_index]

cnames = colnames(numeric_data)
ln = length(cnames)

for (i in 1:6)
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "target"), data = subset(train_df_1))+
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                        outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="target")+
           ggtitle(paste("Box plot of target for",cnames[i])))
}


gridExtra::grid.arrange(gn1,gn2,gn3,ncol=3)
gridExtra::grid.arrange(gn4,gn5,gn6,ncol=3)
gridExtra::grid.arrange(gn22,gn23,gn24,ncol=3)
gridExtra::grid.arrange(gn25,gn26,gn27,ncol=3)
gridExtra::grid.arrange(gn28,gn29,gn30,ncol=3)


#Outlier:
# # #loop to remove outliers from all variables in feature selected dataset
for(i in (3:length(cnames))){
  print(cnames[i-2])
  val = train_df_1[,i][train_df_1[,i] %in% boxplot.stats(train_df_1[,i])$out]
  print(length(val))
  train_df_1 = train_df_1[which(!train_df_1[,i] %in% val),]
}

# for(i in cnames){
#   q = quantile(train_df_1[,i], c(0.25,0.75), names = FALSE)
#   iqr = q[2] - q[1]
#   mini = q[1] - (iqr*1.5)
#   maxi = q[2] + (iqr*1.5)
#   print(i)
#   print(mini)
#   print(maxi)
#   
# }

nrow(train_df_1) #193026 ## after removal of

###############################################################################################################
#Correlation Plot after removing outliers:
numeric_index = sapply(train_df_1,is.numeric) #selecting only numeric
numeric_data = train_df_1[,numeric_index]
corr_data_1 <- cor(numeric_data)
palette = colorRampPalette(c("green", "white", "red")) (20)
heatmap(x = corr_data_1, col = palette, symm = TRUE)
###############################################################################################################

#Feature Scaling - MinMax Method
for(i in cnames){
  print(i)
  train_df_1[,i] = (train_df_1[,i] - min(train_df_1[,i]))/
    (max(train_df_1[,i] - min(train_df_1[,i])))
}

#########################################################################################################
train_df_1$target <- as.character(train_df_1$target)
train_df_1$target <- as.factor(train_df_1$target)
##Simple Random Sampling
#random_sampled = train_df[sample(nrow(train_df), 100000, replace = F), ]

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(train_df_1$target, p = .70, list = FALSE)
train = train_df_1[ train.index,]
test  = train_df_1[-train.index,]

#Plot of Stratified sample Train and test target variables
freq(train$target) #percentage of target variable
freq(test$target) #percentage of target variable
#######################################################################################################

##*********************************Models************************************##
train = train[, -1]
test = test[, -1]
#Logistic Regression
logit_model = glm(target ~ ., data = train, family = "binomial")

#summary of the model
summary(logit_model)


#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


#Function to calculate metrics
metrics <- function(x_act, x_pred){
  ConfMatrix = table(x_act, x_pred)
  TN = ConfMatrix[1]
  FN = ConfMatrix[3]
  TP = ConfMatrix[4]
  FP = ConfMatrix[2]
  
  print(paste("True Negative : ", TN))
  print(paste("True Positive : ", TP))
  print(paste("False Negative:", FN))
  print(paste("False Positive:", FP))
  
  ACC = (TP+TN)/(TP+TN+FP+FN)
  TPR = TP/(TP+FN)
  FPR = FP/(FP+TN)
  TNR = TN/(TN+FP)
  FNR = FN/(FN+TP)
  Precision = TP/(TP+FP)
  F1 = 2*((Precision*TPR)/(Precision+TPR))
  print(paste("Accuracy :", ACC))
  print(paste("Sensitivity or Recall/TPR :", TPR))
  print(paste("Specificity or TNR :", TNR))
  print(paste("Fall-Out or FPR :", FPR))
  print(paste("Miss rate or False Negative rate :", FNR))
  print(paste("Precision :" , Precision))
  print(paste("F1 Score: " , F1))
  
}


##Evaluate the performance of classification model
ConfMatrix_LOGIT = metrics(test$target, logit_Predictions)
ROC1 <- roc.curve(test$target, logit_Predictions, curve = TRUE)
plot(ROC1, lwd = 4)

# logit_Predictions <- as.character(logit_Predictions)
# logit_Predictions <- as.factor(logit_Predictions)
# confusionMatrix(test$target, logit_Predictions)

#**********************************************************************************************

##Decision tree for classification
#Develop Model on training data
ctrl = C5.0Control(subset = TRUE, CF = 0.5, minCases = 2)
C50_model = C5.0(target ~., train, trials = 2, rules = TRUE, control = ctrl)

#plot(C50_model)
#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50Rules_trial10.txt")

#Lets predict for test cases
C50_Predictions = predict(C50_model, test[,-1], type = "class")

confMatrix_C50 = metrics(test$target, C50_Predictions)

ROC2 <- roc.curve(test$target, C50_Predictions, curve = TRUE)
plot(ROC2)

#*****************************************************************************************


###Random Forest
RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 200)

#Predict test data using random forest model
RF_Predictions = predict(RF_model, test[,-1])

confMatrix_RF = metrics(test$target, RF_Predictions)

ROC3 <- roc.curve(test$target, RF_Predictions, curve = TRUE)
plot(ROC3)

#**********************************************************************************************

# #Naive Bayes Model
# library(e1071)
# #Develop model
# NB_model = naiveBayes(target ~ ., data = train)
# 
# #predict on test cases #raw
# NB_Predictions = predict(NB_model, test[,2:31], type = 'class')
# 
# confMatrix_NB = metrics(test$target, NB_Predictions)
# 
# ROC4 <- roc.curve(test$target, NB_Predictions, curve = TRUE)
# plot(ROC4)

#***********************************************************************************************

# #kNN Model
# library(class)
# #Predict test data
# KNN_Predictions = knn(train[, 2:31], test[, 2:31], train$target, k = 7)
# 
# confMatrix_knn = metrics(test$target, KNN_Predictions)
# 
# ROC5 <- roc.curve(test$target, KNN_Predictions, curve = TRUE)
# plot(ROC5)

#***************************************************************************************

#Downsampling:
train_00 <- train_df_1[train_df_1$target==0, ] #take entries with only target = 0 values
train_11 <- train_df_1[train_df_1$target==1, ] #take entries with only target = 1 values
index = sample(1:nrow(train_00), 0.3 * nrow(train_00))
train_00 = train_00[index,]
undersampled_df = rbind(train_00, train_11)


#Divide undersampled data into train and test using stratified sampling method
set.seed(2222)
train.index = createDataPartition(undersampled_df$target, p = .70, list = FALSE)
train = undersampled_df[ train.index,]
# test  = undersampled_df[-train.index,]

#Plot of Stratified sample Train and test target variables
freq(train$target) #percentage of target variable
freq(test$target) #percentage of target variable

train = train[, -1]
test = test[, -1]
##############################################################################################

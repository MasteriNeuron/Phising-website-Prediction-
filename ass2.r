# Import necessary libraries
library(readr)
library(forcats)
library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(tidyverse)
library(Boruta)
library(mlbench)
library(party)
library(gbm)
library(e1071)
library(class)
library(randomForest)
library(ipred)
library(pROC)

# Load the dataset
Phish <- read_csv("phishingdata.csv", col_names = TRUE)

# Check for missing values
summary(Phish)
miss_pct <- sapply(Phish, function(x) sum(is.na(x)) / length(x) * 100)
print(miss_pct)

# Handle missing data
# Impute missing values using median for numeric features
Phish <- Phish %>%
  mutate_if(is.numeric, ~replace_na(., median(., na.rm = TRUE)))

# Set the random seed using your Student ID
set.seed(100000) # Replace 100000 with your Student ID

# Create a data frame with numbers 1 to 50
L <- as.data.frame(c(1:50))

# Randomly sample 10 rows from L without replacement
L <- L[sample(nrow(L), 10, replace = FALSE), ]

# Filter Phish to only include rows where A01 is in the sampled L
Phish <- Phish[(Phish$A01 %in% L), ]

# Randomly sample 2000 rows from Phish without replacement
PD <- Phish[sample(nrow(Phish), 2000, replace = FALSE), ]

# Proportion of phishing sites to legitimate sites
phish_prop <- PD %>%
  group_by(Class) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

print("Proportion of phishing sites to legitimate sites:")
print(phish_prop)

# Feature Selection
set.seed(100000)
default_br <- Boruta(Class ~ ., data = PD, doTrace = 2, maxRuns = 250)
print(default_br)
plot(default_br, las = 2, cex.axis = 0.7, ylim = c(0, 45))

print(getSelectedAttributes(default_br))
# So, we have 16 Important attributes

# Filtered features after removing highly correlated features
filtered_features <- getSelectedAttributes(default_br)  # Get the confirmed important features

# Select only the filtered features and the target variable
PD <- PD[, c(filtered_features, "Class")]

# Proportion of phishing sites to legitimate sites
phish_prop <- PD %>%
  group_by(Class) %>%
  summarise(count = n()) %>%
  mutate(proportion = count / sum(count))

print("Proportion of phishing sites to legitimate sites:")
print(phish_prop)

# Descriptions of predictor variables
numeric_cols <- sapply(PD, is.numeric)
numeric_data <- PD[, numeric_cols]

for (col in names(numeric_data)) {
  print(paste0("Summary for ", col, ":"))
  print(summary(numeric_data[, col]))
  cat("\n")
}

# Split the data into training and test sets
set.seed(100000)  # For reproducibility
trainIndex <- createDataPartition(PD$Class, p = 0.7, list = FALSE)
trainData <- PD[trainIndex, ]
testData <- PD[-trainIndex, ]

# Check for class imbalance in the training set
table(trainData["Class"])

barplot(prop.table(table(trainData["Class"])),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

# Feature scaling
data.pre <- preProcess(trainData, method = "range")
trainData_scaled <- predict(data.pre, trainData)
testData_scaled <- predict(data.pre, testData)

# Append the target variable back to the scaled datasets
trainData_scaled$Class <- trainData$Class
testData_scaled$Class <- testData$Class

### Train Models ###
library(rpart)

# Train a Decision Tree model
trainData_scaled$Class <- factor(trainData_scaled$Class)
tree <- rpart(Class ~ ., trainData_scaled)

# Train a Naive Bayes model
naive_bayes_model <- naiveBayes(Class ~ ., data = trainData_scaled)

# Train a Bagging model
bagging_model <- bagging(Class ~ ., data = trainData_scaled, nbagg = 25)


library(xgboost)
# Train an XGBoost model
xgb_train <- xgb.DMatrix(data = as.matrix(trainData_scaled[-ncol(trainData_scaled)]), label = as.numeric(trainData_scaled$Class) - 1)
xgb_test <- xgb.DMatrix(data = as.matrix(testData_scaled[-ncol(testData_scaled)]), label = as.numeric(testData_scaled$Class) - 1)
xgb_params <- list(
  booster = "gbtree",
  eta = 0.01,
  max_depth = 8,
  gamma = 4,
  subsample = 0.75,
  colsample_bytree = 1,
  objective = "binary:logistic",
  eval_metric = "logloss"
)
xgb_model <- xgb.train(
  params = xgb_params,
  data = xgb_train,
  nrounds = 5000,
  verbose = 1
)
xgb_model




# Train a Boosting model
#boosting_model <- gbm(Class ~ ., data = trainData_scaled, distribution = "bernoulli", n.trees = 100, interaction.depth = 1, n.minobsinnode = 10, shrinkage = 0.1, cv.folds = 5)

library(randomForest)
# Train a Random Forest model
random_forest_model = randomForest(x = trainData_scaled[-17],
                          y = trainData_scaled$Class,
                          ntree = 500, random_state = 42)

#random_forest_model <- randomForest(Class ~ ., data = trainData_scaled, ntree = 100)

### Make Predictions ###

# Predict using the Decision Tree model
dt_pred <- predict(tree, testData_scaled, type = "class")

# Predict using the Naive Bayes model
nb_pred <- predict(naive_bayes_model, newdata = testData_scaled)

# Predict using the Bagging model
bagging_pred <- predict(bagging_model, newdata = testData_scaled)

# Predict using the Boosting model
#boosting_pred <- predict(boosting_model, newdata = testData_scaled, n.trees = 100, type = "response")
#boosting_pred_class <- ifelse(boosting_pred > 0.5, 1, 0)

# Predict using the XGBoost model
xgb_pred_prob <- predict(xgb_model, newdata = xgb_test)
xgb_pred <- ifelse(xgb_pred_prob > 0.5, 1, 0)


# Predict using the Random Forest model

rf_pred <- predict(random_forest_model, newdata = testData_scaled[-17], type = "class")

### Convert Predictions to Factors ###

testData_scaled$Class <- as.factor(testData_scaled$Class)
levels(testData_scaled$Class) <- c("0", "1")

dt_pred <- as.factor(dt_pred)
levels(dt_pred) <- c("0", "1")

nb_pred <- as.factor(nb_pred)
levels(nb_pred) <- c("0", "1")

bagging_pred <- as.factor(bagging_pred)
levels(bagging_pred) <- c("0", "1")

xgb_pred <- as.factor(xgb_pred)
levels(xgb_pred) <- c("0", "1")

rf_pred <- as.factor(rf_pred)
levels(rf_pred) <- c("0", "1")

### Confusion Matrix and Accuracy ###

# Decision Tree
dt_conf_matrix <- confusionMatrix(dt_pred, testData_scaled$Class)
dt_accuracy <- dt_conf_matrix$overall['Accuracy']

# Naive Bayes
nb_conf_matrix <- confusionMatrix(nb_pred, testData_scaled$Class)
nb_accuracy <- nb_conf_matrix$overall['Accuracy']

# Bagging
bagging_conf_matrix <- confusionMatrix(bagging_pred, testData_scaled$Class)
bagging_accuracy <- bagging_conf_matrix$overall['Accuracy']

# XGBoost
xgb_conf_matrix <- confusionMatrix(xgb_pred, testData_scaled$Class)
xgb_accuracy <- xgb_conf_matrix$overall['Accuracy']


# Random Forest
rf_conf_matrix <- confusionMatrix(rf_pred, testData_scaled$Class)
rf_accuracy <- rf_conf_matrix$overall['Accuracy']

### ROC Curves and AUC ###

# Calculate probabilities
dt_prob <- predict(tree, newdata = testData_scaled, type = "prob")[,2]
nb_prob <- predict(naive_bayes_model, newdata = testData_scaled, type = "raw")[,2]
bagging_prob <- predict(bagging_model, newdata = testData_scaled, type = "prob")[,2]
#boosting_prob <- predict(boosting_model, newdata = testData_scaled, n.trees = 100, type = "response")
rf_prob <- predict(random_forest_model, newdata = testData_scaled, type = "prob")[,2]
xgb_prob <- xgb_pred_prob

# ROC Curves and AUC
#roc
roc_dt <- roc(testData_scaled$Class, dt_prob)
roc_nb <- roc(testData_scaled$Class, nb_prob)
roc_bagging <- roc(testData_scaled$Class, bagging_prob)
#roc_boosting <- roc(testData_scaled$Class, boosting_prob)
roc_rf <- roc(testData_scaled$Class, rf_prob)
roc_xgb <- roc(testData_scaled$Class, as.numeric(xgb_prob))

#auc
auc_dt <- auc(roc_dt)
auc_nb <- auc(roc_nb)
auc_bagging <- auc(roc_bagging)
#auc_boosting <- auc(roc_boosting)
auc_rf <- auc(roc_rf)
auc_xgb <- auc(roc_xgb)

# Plot ROC Curves
plot(roc_dt, col = "red", main = "ROC Curves")
plot(roc_nb, col = "blue", add = TRUE)
plot(roc_bagging, col = "green", add = TRUE)
#plot(roc_boosting, col = "purple", add = TRUE)
plot(roc_rf, col = "orange", add = TRUE)
plot(roc_xgb, col = "brown", add = TRUE)
legend("bottomright", legend = c("Decision Tree", "Naive Bayes", "Bagging",  "Random Forest", "XGBoost"), col = c("red", "blue", "purple", "orange", "brown"), lwd = 2)

### Comparison Table ###

comparison_table <- data.frame(
  Model = c("Decision Tree", "Naive Bayes", "Bagging", "Random Forest", "XGBoost"),
  Accuracy = c(dt_accuracy, nb_accuracy, bagging_accuracy, rf_accuracy, xgb_accuracy),
  AUC = c(auc_dt, auc_nb, auc_bagging, auc_rf, auc_xgb)
)

print(comparison_table)

# Determine the best classifier based on Accuracy
best_classifier <- comparison_table[which.max(comparison_table$Accuracy),]
print(best_classifier)


# So, we have best model Random Forest having 80.33% of accuracy and Auc of 82.4%

#####################task 8:
# Random Forest feature importance
rf_importance <- importance(random_forest_model)
rf_importance_df <- data.frame(Variable = rownames(rf_importance), Importance = rf_importance[, 1])
rf_importance_df <- rf_importance_df[order(-rf_importance_df$Importance), ]

# XGBoost feature importance
xgb_importance <- xgb.importance(model = xgb_model)
xgb_importance_df <- as.data.frame(xgb_importance)
xgb_importance_df <- xgb_importance_df[order(-xgb_importance_df$Gain), ]

# Combine and compare importance
combined_importance <- merge(rf_importance_df, xgb_importance_df, by.x = "Variable", by.y = "Feature", all = TRUE)
combined_importance <- combined_importance[order(-combined_importance$Importance), ]

print("Combined Feature Importance:")
print(combined_importance)

# Identify least important features
least_important_rf <- tail(rf_importance_df, 5)
least_important_xgb <- tail(xgb_importance_df, 5)

print("Least Important Features in Random Forest:")
print(least_important_rf)

print("Least Important Features in XGBoost:")
print(least_important_xgb)


###########task:9
# Create a simple Decision Tree model
simple_tree_model <- rpart(Class ~ ., data = trainData_scaled, control = rpart.control(maxdepth = 2))
#rpart.plot(simple_tree_model)

# Predict using the simple tree model
simple_tree_pred <- predict(simple_tree_model, newdata = testData_scaled, type = "class")

# Evaluate performance
simple_tree_conf_matrix <- confusionMatrix(simple_tree_pred, testData_scaled$Class)
simple_tree_accuracy <- simple_tree_conf_matrix$overall['Accuracy']
simple_tree_roc <- roc(testData_scaled$Class, as.numeric(predict(simple_tree_model, newdata = testData_scaled)[, 2]))
simple_tree_auc <- auc(simple_tree_roc)

print(paste("Simple Tree Accuracy:", simple_tree_accuracy))
print(paste("Simple Tree AUC:", simple_tree_auc))

# Compare with other models
comparison_table <- rbind(comparison_table, data.frame(
  Model = "Simple Tree",
  Accuracy = simple_tree_accuracy,
  AUC = simple_tree_auc
))

print(comparison_table)


########################task :10 Create the Best Tree-Based Classifier
# Cross-validation and parameter tuning for Random Forest
control <- trainControl(method = "cv", number = 5)
tunegrid <- expand.grid(.mtry = c(2, 4, 6, 8, 10))

set.seed(100000)
rf_optimized <- train(Class ~ ., data = trainData_scaled, method = "rf", trControl = control, tuneGrid = tunegrid)

# Get the best model
best_rf_model <- rf_optimized$finalModel

# Predict using the optimized Random Forest model
best_rf_pred <- predict(best_rf_model, newdata = testData_scaled)

# Evaluate performance
best_rf_conf_matrix <- confusionMatrix(best_rf_pred, testData_scaled$Class)
best_rf_accuracy <- best_rf_conf_matrix$overall['Accuracy']
best_rf_roc <- roc(testData_scaled$Class, as.numeric(predict(best_rf_model, newdata = testData_scaled, type = "prob")[, 2]))
best_rf_auc <- auc(best_rf_roc)

print(paste("Optimized Random Forest Accuracy:", best_rf_accuracy))
print(paste("Optimized Random Forest AUC:", best_rf_auc))

# Compare with other models
comparison_table <- rbind(comparison_table, data.frame(
  Model = "Optimized Random Forest",
  Accuracy = best_rf_accuracy,
  AUC = best_rf_auc
))

print(comparison_table)


#step:11 Implement an Artificial Neural Network (ANN) Classifier
library(nnet)
# Install caret package if not already installed
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}

# Load caret package
library(caret)


# Convert the target variable to a factor
trainData_scaled$Class <- as.factor(trainData_scaled$Class)
testData_scaled$Class <- as.factor(testData_scaled$Class)

# Train the ANN model
set.seed(100000)
ann_model <- nnet(Class ~ ., data = trainData_scaled, size = 10, maxit = 200, linout = FALSE)

# Predict using the ANN model
ann_pred_prob <- predict(ann_model, newdata = testData_scaled, type = "raw")
ann_pred <- ifelse(ann_pred_prob > 0.5, 1, 0)

# Evaluate performance
ann_conf_matrix <- confusionMatrix(as.factor(ann_pred), testData_scaled$Class)
ann_accuracy <- ann_conf_matrix$overall['Accuracy']
ann_roc <- roc(testData_scaled$Class, as.numeric(ann_pred_prob))
ann_auc <- auc(ann_roc)

print(paste("ANN Accuracy:", ann_accuracy))
print(paste("ANN AUC:", ann_auc))

# Compare with other models
comparison_table <- rbind(comparison_table, data.frame(
  Model = "ANN",
  Accuracy = ann_accuracy,
  AUC = ann_auc
))

print(comparison_table)



###Step 12: Implement a New Classifier (Support Vector Machine with Radial Basis Function Kernel)
# Install and load required packages
if (!requireNamespace("e1071", quietly = TRUE)) {
  install.packages("e1071")
}

library(e1071)
# Ensure the caret package is installed and loaded
if (!requireNamespace("caret", quietly = TRUE)) {
  install.packages("caret")
}



# Train the SVM model with RBF kernel
set.seed(100000)
# Load necessary library
library(e1071)

# Define and train the SVM model
svm_model <- svm(Class ~ ., data = trainData_scaled, probability = TRUE)

# Predict using the SVM model
svm_pred <- predict(svm_model, newdata = testData_scaled, probability = TRUE)
svm_prob <- attr(svm_pred, "probabilities")[, 2]

# Calculate performance metrics
svm_conf_matrix <- confusionMatrix(as.factor(svm_pred), as.factor(testData_scaled$Class))
svm_roc <- roc(testData_scaled$Class, svm_prob)
svm_auc <- auc(svm_roc)

# Print results
svm_accuracy <- svm_conf_matrix$overall['Accuracy']
print(paste("SVM Accuracy:", svm_accuracy))
print(paste("SVM AUC:", svm_auc))


# Compare with other models
comparison_table <- rbind(comparison_table, data.frame(
  Model = "SVM",
  Accuracy = svm_accuracy,
  AUC = svm_auc
))

print(comparison_table)




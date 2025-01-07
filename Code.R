library(tidyverse)
library(dplyr)
library(ggplot2)
data <- read.csv("~/Documents/Jacqueline/duke/Fall Term/Data Science for Business/Final Project -- Fraud/fraudTest.csv")
df <- data
clean_df <- na.omit(df) #Drop NAs 

library(randomForest)
library(caret)
library(tidyverse)
library(dplyr)
library(lubridate)

#####DATA CLEANING AND PREPARATION 

#drop columns that may not be useful for predicting fraud detection 
data <- data[, !(names(data) %in% c("first", "last", "street", "lat", "long", "cc_num", "trans_num", "unix_time", "merch_lat","X", "merch_long"))]
# Convert the transaction dates and time variable to two separate columns for date time 
data[c("trans_date", "trans_time")] <- str_split_fixed(data$trans_date_trans_time, " ", 2)
data$trans_date <- as.Date(data$trans_date)
data$dob <- as.Date(data$dob)
# Assign a column for Age
data$age <- interval(as.Date(data$dob), today()) / years(1)
# Clean the Merchant Column 
data$merchant <- str_remove(data$merchant, "^fraud_")
# Assign factors to relevant categorical variables 
data$category <- as.factor(data$category)
data$gender <- as.factor(data$gender)
data$state <- as.factor(data$state)
#Assign binary dummy variable to Gender column where 0 is Male and 1 is Female 
data$gender <- ifelse(data$gender=="F", 1, 0)
# Classify the fraud column into factors 
data$is_fraud <- factor(data$is_fraud, levels = c(0, 1))
data_clean <- data

#####RANDOM FOREST WITH DOWNSAMPLING 
##Random Forest Model with Downsampling 
# Separate the fraudulent and non-fraudulent transactions from data_clean
fraudulent_data <- data_clean %>% filter(is_fraud == 1)
non_fraudulent_data <- data_clean %>% filter(is_fraud == 0)

# Downsample the non-fraudulent transactions to match the number of fraudulent ones
set.seed(123)  # For reproducibility
downsampled_non_fraud <- non_fraudulent_data %>%
  sample_n(10*nrow(fraudulent_data))

# Combine the downsampled non-fraudulent data with the fraudulent data
downsampled_data <- bind_rows(fraudulent_data, downsampled_non_fraud)

# Shuffle the combined dataset
downsampled_data <- downsampled_data %>% sample_frac(1)

# Split the data into training and testing sets (80% train, 20% test)
set.seed(123)
train_indices <- sample(seq_len(nrow(downsampled_data)), size = 0.8 * nrow(downsampled_data))
train <- downsampled_data[train_indices, ]
test <- downsampled_data[-train_indices, ]

# Train Random Forest model
set.seed(123)
rf_model <- randomForest(is_fraud ~ ., 
                         data = train, 
                         ntree = 100, 
                         mtry = sqrt(ncol(train) - 1), 
                         nodesize = 5, 
                         importance = TRUE)

# Make predictions on the test set
rf_predictions <- predict(rf_model, newdata = test)

# Create confusion matrix to evaluate the model
confusion_matrix <- confusionMatrix(rf_predictions, test$is_fraud)
print(confusion_matrix)

# Assuming confusion_matrix is from the confusionMatrix function
confusion_matrix <- confusionMatrix(rf_predictions, test$is_fraud)

# Extract the confusion matrix table
conf_matrix_table <- confusion_matrix$table

# True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
TP <- conf_matrix_table[2, 2]
FP <- conf_matrix_table[1, 2]
TN <- conf_matrix_table[1, 1]
FN <- conf_matrix_table[2, 1]

# Calculate Precision, Recall, Accuracy, and F1-Score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
accuracy <- (TP + TN) / (TP + TN + FP + FN)
f1_score <- 2 * ((precision * recall) / (precision + recall))

# Print the results
cat("Precision: ", round(precision, 4), "\n")
cat("Recall: ", round(recall, 4), "\n")
cat("Accuracy: ", round(accuracy, 4), "\n")
cat("F1-Score: ", round(f1_score, 4), "\n")


#############################################################
library(caret)

set.seed(123)
cv_control <- trainControl(method = "cv", number = 5)

rf_cv_model <- train(
  is_fraud ~ ., 
  data = downsampled_data,
  method = "rf",
  trControl = cv_control,
  ntree = 100,
  tuneLength = 5
)

print(rf_cv_model)


plot(rf_cv_model)

cv_results <- rf_cv_model$results
mean_accuracy <- mean(cv_results$Accuracy)
mean_kappa <- mean(cv_results$Kappa)
cat("Mean Accuracy: ", round(mean_accuracy, 4), "\n")
cat("Mean Kappa: ", round(mean_kappa, 4), "\n")





#Confusion Matrix 
conf_matrix <- as.data.frame(as.table(conf_matrix_table))
colnames(conf_matrix) <- c("Prediction", "Reference", "Freq")

ggplot(data = conf_matrix, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_text(aes(label = Freq), vjust = 1, color = "white", size = 6) +
  theme_minimal() +
  labs(title = "Confusion Matrix", x = "Predicted Class", y = "Actual Class")



## Feature Importance
# Variable Importance
var_importance <- as.data.frame(importance(rf_model))
var_importance$Variable <- rownames(var_importance)
rownames(var_importance) <- NULL
var_importance <- var_importance[order(var_importance$MeanDecreaseGini, decreasing = TRUE), ]

# Feature Importance
ggplot(var_importance, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance", x = "Variables", y = "Mean Decrease in Gini")





test$prediction <- rf_predictions
ggplot(test %>% filter(amt < 2000), aes(x = amt, y = fraud_prob, color = prediction != is_fraud)) +
  geom_point(alpha = 0.6) +
  labs(title = "Misclassification Analysis (Transaction Amount < 2000)", 
       x = "Transaction Amount", 
       y = "Fraud Probability") +
  scale_color_manual(values = c("grey", "red"), labels = c("Correct", "Incorrect")) +
  theme_minimal()




#
state_fraud <- data %>%
  filter(is_fraud == 1) %>%
  group_by(state) %>%
  summarise(fraud_count = n())


#install.packages("usmap")
library(usmap)
library(ggplot2)

plot_usmap(data = state_fraud, values = "fraud_count", color = "white") +
  scale_fill_continuous(low = "lightblue", high = "darkblue", name = "Fraud Count") +
  labs(title = "Fraud Incidents by State") +
  theme_minimal()








date_fraud <- data %>%
  filter(is_fraud == 1) %>%
  group_by(trans_date) %>%
  summarise(fraud_count = n())

ggplot(date_fraud, aes(x = trans_date, y = fraud_count)) +
  geom_line(color = "red") +
  labs(title = "Daily Fraud Incidents Over Time", x = "Date", y = "Fraud Count") +
  theme_minimal()




install.packages("lubridate")
library(lubridate)

data$trans_date_trans_time <- ymd_hms(data$trans_date_trans_time)

data$day_of_week <- wday(data$trans_date_trans_time, label = TRUE, abbr = FALSE)

fraud_by_day <- data %>%
  filter(is_fraud == 1) %>%
  group_by(day_of_week) %>%
  summarise(fraud_count = n())

ggplot(fraud_by_day, aes(x = day_of_week, y = fraud_count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Fraud Incidents by Day of the Week", x = "Day of the Week", y = "Fraud Count") +
  theme_minimal()



library(lubridate)

data$week <- floor_date(data$trans_date, unit = "week")

weekly_fraud <- data %>%
  filter(is_fraud == 1) %>%
  group_by(week) %>%
  summarise(fraud_count = n())
ggplot(weekly_fraud, aes(x = week, y = fraud_count)) +
  geom_line(color = "red") +
  labs(title = "Weekly Fraud Incidents Over Time", x = "Week", y = "Fraud Count") +
  theme_minimal()



### Calculate the oos r^2

test$is_fraud <- as.numeric(as.character(test$is_fraud))
rf_predictions <- as.numeric(as.character(rf_predictions))

train_mean <- mean(train$is_fraud, na.rm = TRUE)

ss_total <- sum((test$is_fraud - train_mean)^2)

ss_residual <- sum((test$is_fraud - rf_predictions)^2)

R2_OOS <- 1 - (ss_residual / ss_total)

cat("Out-of-Sample R^2: ", round(R2_OOS, 4), "\n")



### ROC & AUC
library(pROC)

rf_probabilities <- predict(rf_model, test, type = "prob")[,2]

roc_obj <- roc(test$is_fraud, rf_probabilities)


plot(roc_obj, col = "blue", main = "ROC Curve")
auc_value <- auc(roc_obj)
cat("AUC: ", round(auc_value, 4), "\n")

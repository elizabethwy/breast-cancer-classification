# Load libraries
install.packages("glmnet")
install.packages("randomForest")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("nnet")       # neural network
install.packages("caret")      # for train/test split and evaluation

library(nnet)
library(caret)
library(rpart)
library(rpart.plot)
library(class)
library(data.table)
library(ggplot2)
library(caret)   
library(car)
library(glmnet)
library(randomForest)
library(knitr)


myDF <- fread("wdbc.data")
data <- myDF  

# Create a summary table
data_summary <- data.frame(
  Feature = c("Total Observations", 
              "Features (Variables)", 
              "Target Variable", 
              "Benign Cases", 
              "Malignant Cases", 
              "Feature Types", 
              "ID Column", 
              "Missing Values"),
  Value = c("569", 
            "30 numeric features + Diagnosis", 
            "Diagnosis (Benign or Malignant)", 
            "357 (approx. 63%)", 
            "212 (approx. 37%)", 
            "All numeric", 
            "Removed / Not used", 
            "None")
)


kable(data_summary, caption = "Summary of the Wisconsin Breast Cancer Dataset")

names(myDF) <- c(
  "ID", "Diagnosis", 
  "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", 
  "concavity1", "concave_points1", "symmetry1", "fractal_dimension1",
  
  "radius2", "texture2", "perimeter2", "area2", "smoothness2", "compactness2", 
  "concavity2", "concave_points2", "symmetry2", "fractal_dimension2",
  
  "radius3", "texture3", "perimeter3", "area3", "smoothness3", "compactness3", 
  "concavity3", "concave_points3", "symmetry3", "fractal_dimension3"
)
##Basic plots for data
myDF$Diagnosis <- factor(myDF$Diagnosis, levels = c("B", "M"), labels = c("Benign", "Malignant"))

ggplot(myDF, aes(x = radius1, y = texture1, color = Diagnosis)) +
  geom_point(alpha = 0.7, size = 2.5) +
  labs(
    title = "Tumor Classification: Radius vs. Texture",
    x = "Mean Radius",
    y = "Mean Texture",
    color = "Diagnosis"
  ) +
  scale_color_manual(
    values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")
  ) +
  theme_minimal(base_size = 14)

ggplot(myDF, aes(x = Diagnosis, y = radius1, fill = Diagnosis)) +
  geom_boxplot(alpha = 0.7, outlier.color = "grey30") +
  scale_fill_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(title = "Radius (mean) by Diagnosis", y = "Mean Radius") +
  theme_minimal(base_size = 14)

###BOXPLOT
ggplot(myDF, aes(x = Diagnosis, fill = Diagnosis)) +
  geom_bar(width = 0.6) +
  scale_fill_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(title = "Distribution of Diagnosis", x = "Diagnosis", y = "Count") +
  theme_minimal(base_size = 14)

# Standardize features
features <- myDF[, 3:32]
features_scaled <- scale(features)
pca_result <- prcomp(features_scaled)

####PCA
# Combine with labels
pca_df <- data.frame(pca_result$x[, 1:2], Diagnosis = myDF$Diagnosis)

ggplot(pca_df, aes(x = PC1, y = PC2, color = Diagnosis)) +
  geom_point(size = 2, alpha = 0.7) +
  scale_color_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(title = "PCA: First 2 Principal Components", x = "PC1", y = "PC2") +
  theme_minimal(base_size = 14)


features <- scale(myDF[, 3:32])  # standardize

pca_result <- prcomp(features)

# View the amount of variance explained
summary(pca_result)

# Extract loadings (rotation matrix)
loadings <- pca_result$rotation

# Get top 10 absolute contributors to PC1
top_PC1 <- sort(abs(loadings[, "PC1"]), decreasing = TRUE)[1:10]
top_PC2 <- sort(abs(loadings[, "PC2"]), decreasing = TRUE)[1:10]

# Display
### tells what the PC1 contributors are
cat("Top contributors to PC1:\n")
print(top_PC1)
#### same for PC2
cat("\nTop contributors to PC2:\n")
print(top_PC2)


screeplot(pca_result, 
          type = "lines", 
          main = "Scree Plot: Variance Explained")



####Logistic regression

features <- myDF[, 3:32]
features$Diagnosis <- myDF$Diagnosis  

features$Diagnosis <- as.factor(features$Diagnosis)

# Split data into training and testing (80/20 split)
set.seed(123) 
trainIndex <- createDataPartition(features$Diagnosis, p = 0.8, list = FALSE)
trainData <- features[trainIndex, ]
testData <- features[-trainIndex, ]

# Fit the Logistic Regression model
# Check multicollinearity using VIF

vif_model <- glm(Diagnosis ~ ., data = trainData, family = binomial)
vif_values <- vif(vif_model)
print(vif_values)

# Optionally filter out variables with very high VIF
high_vif <- names(vif_values[vif_values > 10])
cat("High VIF features:", high_vif, "\n")

logistic_model <- glm(Diagnosis ~ ., data = trainData, family = binomial)

# Print the model summary
summary(logistic_model)

# Predict on the test data
predictions <- predict(logistic_model, testData, type = "response")

# Convert probabilities to binary classification (0 or 1)
predicted_class <- ifelse(predictions > 0.5, "Malignant", "Benign")

# Create confusion matrix to evaluate the model
confusion <- confusionMatrix(factor(predicted_class), testData$Diagnosis)
print(confusion)


logistic_model_concavity1 <- glm(Diagnosis ~ concavity1, data = trainData, family = binomial)

plot_data <- data.frame(
  radi1 = testData$concavity1,
  predicted = predict(logistic_model_concavity1, testData, type = "response")
)

# Visualize the logistic regression decision boundary (optional)
# Note: This is just an example for 2 features, you can modify based on which features you want to visualize
ggplot(data = plot_data, aes(x = radi1, y = predicted, color = predicted)) +
  geom_point(alpha = 0.7, size = 3) +
  geom_smooth(method = 'glm', method.args = list(family = 'binomial'), se =F) +
  labs(title = "Logistic Regression Decision Boundary",
       x = "Radius 1",
       y = "Diagnosis") +
  theme_minimal(base_size = 14)
# Predict on the test data
test_predictions <- predict(logistic_model, testData, type = "response")

# Convert probabilities to binary classification (0 or 1)
predicted_class_test <- ifelse(test_predictions > 0.5, "Malignant", "Benign")

# Create confusion matrix to evaluate the model
confusion <- confusionMatrix(factor(predicted_class_test), testData$Diagnosis)
print(confusion)

######

# Select your two features + target
df_2d <- myDF[, c("fractal_dimension1", "smoothness3", "Diagnosis")]

# Fit the logistic regression model
logit_model <- glm(Diagnosis ~ fractal_dimension1 + smoothness3, data = df_2d, family = "binomial")

# Create a grid for prediction
x_seq <- seq(min(df_2d$fractal_dimension1), max(df_2d$fractal_dimension1), length.out = 200)
y_seq <- seq(min(df_2d$smoothness3), max(df_2d$smoothness3), length.out = 200)

grid <- expand.grid(fractal_dimension1 = x_seq, smoothness3 = y_seq)

# Predict probability of Malignant on the grid
grid$prob_malignant <- predict(logit_model, newdata = grid, type = "response")

# Plot data points + decision boundary
ggplot() +
  geom_point(data = df_2d, aes(x = fractal_dimension1, y = smoothness3, color = Diagnosis), size = 2.5, alpha = 0.8) +
  geom_contour(data = grid, aes(x = fractal_dimension1, y = smoothness3, z = prob_malignant),
               breaks = 0.5, color = "black", linetype = "dashed") +
  scale_color_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(
    title = "Logistic Regression (fractal_dimension1 vs. smoothness3)",
    x = "Mean fractal_dimension1",
    y = "Mean smoothness3"
  ) +
  theme_minimal(base_size = 14)


###Lasso



# Prepare data for glmnet (x = matrix, y = factor response)
x <- as.matrix(myDF[, 3:32])  # all 30 original features
y <- myDF$Diagnosis

# Encode response as numeric (required for glmnet)
y_bin <- ifelse(y == "Malignant", 1, 0)

# Train/test split
set.seed(123)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x[trainIndex, ]
x_test <- x[-trainIndex, ]
y_train <- y_bin[trainIndex]
y_test <- y[trainIndex]  # for confusion matrix later

# LASSO with cross-validation to find optimal lambda
cvfit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial")

# Best lambda
best_lambda <- cvfit$lambda.min

# Final LASSO model
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")

# Predict on test set
pred_probs <- predict(lasso_model, newx = x_test, type = "response")
pred_class_lasso <- ifelse(pred_probs > 0.5, "Malignant", "Benign")

cat("\nConfusion Matrix: LASSO Logistic Regression\n")
confusionMatrix(factor(pred_class_lasso), myDF$Diagnosis[-trainIndex])

x <- as.matrix(myDF[, -1])  # Drop the Diagnosis column
y <- myDF$Diagnosis         # Target variable

# Encode y as numeric for glmnet (0/1)
y_numeric <- ifelse(y == "Benign", 0, 1)

# Fit LASSO model (alpha = 1)
lasso_model <- glmnet(x, y_numeric, alpha = 1, family = "binomial")

# Plot the LASSO path
plot(lasso_model, xvar = "lambda", label = TRUE)
title("LASSO Paths (L1 Regularization)")
abline(v = log(best_lambda), col = "red", lty = 2, lwd = 2)
# Assuming you already have your cv.glmnet result and best_lambda

abline(v = log(best_lambda), col = "red", lty = 2, lwd = 2)  # add vertical line


####decision tree
# Install packages if needed
install.packages("rpart")
install.packages("rpart.plot")

# Load libraries
library(rpart)
library(rpart.plot)

# Prepare data
data <- myDF  # your preprocessed dataset
data$Diagnosis <- as.factor(data$Diagnosis)

# Split into training and testing (80/20)
set.seed(123)
trainIndex <- createDataPartition(data$Diagnosis, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train decision tree model
tree_model <- rpart(Diagnosis ~ ., data = trainData[, -1], method = "class")

# Plot the decision tree
rpart.plot(tree_model, type = 2, extra = 106, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)

# Predict and evaluate
tree_preds <- predict(tree_model, testData, type = "class")
conf_matrix <- confusionMatrix(tree_preds, testData$Diagnosis)
print(conf_matrix)


####RANDOM FOREST?

rf_model <- randomForest(Diagnosis ~ ., data = trainData, ntree = 500, importance = TRUE)
rf_preds <- predict(rf_model, newdata = testData)
confusionMatrix(rf_preds, testData$Diagnosis)

######Decision Boundaries
# Select only two features + target
df_2d <- myDF[, c("radius1", "texture1", "Diagnosis")]

# Fit logistic regression model
model_2d <- glm(Diagnosis ~ radius1 + texture1, data = df_2d, family = "binomial")

# Create a grid of values across the feature space
x_min <- min(df_2d$radius1)
x_max <- max(df_2d$radius1)
y_min <- min(df_2d$texture1)
y_max <- max(df_2d$texture1)

# Generate a grid
grid <- expand.grid(
  radius1 = seq(x_min, x_max, length.out = 200),
  texture1 = seq(y_min, y_max, length.out = 200)
)

# Predict probabilities on the grid
grid$prob <- predict(model_2d, newdata = grid, type = "response")

# Plot
ggplot() +
  geom_point(data = df_2d, aes(x = radius1, y = texture1, color = Diagnosis), size = 2, alpha = 0.8) +
  geom_contour(data = grid, aes(x = radius1, y = texture1, z = prob), 
               breaks = 0.5, color = "black", linetype = "dashed") +
  scale_color_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(title = "Decision Boundary (Logistic Regression)",
       x = "Radius (mean)",
       y = "Texture (mean)") +
  theme_minimal(base_size = 14)
####### again with different variables
df_2d <- myDF[, c("area1", "concavity1", "Diagnosis")]
logit_model <- glm(Diagnosis ~ ., data = df_2d, family = "binomial")
x_min <- min(df_2d$area1)
x_max <- max(df_2d$area1)
y_min <- min(df_2d$concavity1)
y_max <- max(df_2d$concavity1)

grid <- expand.grid(
  area1 = seq(x_min, x_max, length.out = 200),
  concavity1 = seq(y_min, y_max, length.out = 200)
)
grid$prob_malignant <- predict(logit_model, newdata = grid, type = "response")
ggplot() +
  geom_point(data = df_2d, aes(x = area1, y = concavity1, color = Diagnosis), size = 2.5, alpha = 0.8) +
  geom_contour(data = grid, aes(x = area1, y = concavity1, z = prob_malignant), 
               breaks = 0.5, color = "black", linetype = "dashed") +
  scale_color_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(
    title = "Decision Boundary: Logistic Regression (Area1 vs Concavity1)",
    x = "Area (mean)",
    y = "Concavity (mean)"
  ) +
  theme_minimal(base_size = 14)
######
logit_model <- glm(Diagnosis ~ area1 + concavity1, data = df_2d, family = "binomial")

# Create a grid for prediction
x_seq <- seq(min(df_2d$area1), max(df_2d$area1), length.out = 200)
y_seq <- seq(min(df_2d$concavity1), max(df_2d$concavity1), length.out = 200)

grid <- expand.grid(area1 = x_seq, concavity1 = y_seq)

# Predict probability of Malignant on the grid
grid$prob_malignant <- predict(logit_model, newdata = grid, type = "response")

# Plot data points + decision boundary
ggplot() +
  geom_point(data = df_2d, aes(x = area1, y = concavity1, color = Diagnosis), size = 2.5, alpha = 0.8) +
  geom_contour(data = grid, aes(x = area1, y = concavity1, z = prob_malignant),
               breaks = 0.5, color = "black", linetype = "dashed") +
  scale_color_manual(values = c("Benign" = "#1f77b4", "Malignant" = "#d62728")) +
  labs(
    title = "Logistic Regression Decision Boundary (area1 vs. concavity1)",
    x = "Mean Area",
    y = "Mean Concavity"
  ) +
  theme_minimal(base_size = 14)

#KNN
features <- myDF[, 3:32]  # Exclude ID and Diagnosis
labels <- myDF$Diagnosis

# Standardize the features
features_scaled <- scale(features)

# Train/test split (same as before)
set.seed(123)
trainIndex <- createDataPartition(labels, p = 0.8, list = FALSE)
trainData <- features_scaled[trainIndex, ]
testData <- features_scaled[-trainIndex, ]
trainLabels <- labels[trainIndex]
testLabels <- labels[-trainIndex]

# Fit KNN model (k = 5 is common)
knn_pred <- knn(train = trainData, test = testData, cl = trainLabels, k = 5)
knn_pred <- knn(train = trainData, test = testData, cl = trainLabels, k = 13)


# Evaluate the model
conf_matrix_knn <- confusionMatrix(knn_pred, testLabels)
print(conf_matrix_knn)


######10 fold cross validation
# Remove ID column if present
data <- myDF[, -1]

# Make sure outcome is a factor
data$Diagnosis <- as.factor(data$Diagnosis)
set.seed(123)  # for reproducibility
cv_control <- trainControl(method = "cv", number = 10)
####Log model for cross validation
logit_model <- train(
  Diagnosis ~ ., 
  data = data,
  method = "glm",
  family = "binomial",
  tlorControl = cv_control
)
###Random Forest cross validation
rf_model <- train(
  Diagnosis ~ ., 
  data = data,
  method = "rf",
  trControl = cv_control
)
####KNN cross validation
knn_model <- train(
  Diagnosis ~ ., 
  data = data,
  method = "knn",
  trControl = cv_control,
  tuneLength = 10  # number of K values to try
)
###SVM
install.packages("caret")  # Only if you haven't installed it yet
library(caret)

install.packages("e1071")
library(e1071)

# Train SVM
# Check the column names to ensure 'Diagnosis' exists
set.seed(123)
trainIndex <- createDataPartition(myDF$Diagnosis, p = 0.8, list = FALSE)
trainData <- myDF[trainIndex, ]
testData <- myDF[-trainIndex, ]
svm_model <- svm(Diagnosis ~ ., data = trainData, kernel = "linear", probability = TRUE)
svm_predictions <- predict(svm_model, newdata = testData)
confusionMatrix(svm_predictions, testData$Diagnosis)

###Neural Network
myDF$Diagnosis <- as.factor(myDF$Diagnosis)
set.seed(123)
trainIndex <- createDataPartition(myDF$Diagnosis, p = 0.8, list = FALSE)
trainData <- myDF[trainIndex, ]
testData <- myDF[-trainIndex, ]
nn_model <- nnet(Diagnosis ~ ., data = trainData, size = 5, decay = 0.1, maxit = 200)
pred_probs <- predict(nn_model, testData, type = "class")

pred_probs <- factor(pred_probs, levels = levels(testData$Diagnosis))
test_labels <- factor(testData$Diagnosis, levels = levels(testData$Diagnosis))

confusionMatrix(pred_probs, test_labels)
######
cv_model <- cv.glmnet(x, y, alpha = 1)

#find optimal lambda value that minimizes test MSE
best_lambda <- cv_model$lambda.min
best_lambda

###Ridge
# Split predictors and outcome
library(caret)
library(glmnet)

# Assume myDF is your full dataset
# If predicting Diagnosis (must be numeric or factor)
y <- ifelse(myDF$Diagnosis == "Malignant", 1, 0)   # Make it numeric 1/0
X <- as.matrix(myDF[, -which(names(myDF) == "Diagnosis")])  # All other variables

# Now split
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)

# Create training and testing sets
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)
plot(ridge_model)

# Fit Ridge model (alpha = 0)
ridge_model <- glmnet(x, y_numeric, alpha = 0, family = "binomial")

# Plot the Ridge path
x <- model.matrix(response ~ ., data = myDF)[, -1]  # predictors (drop intercept)
y <- myDF$response                                  # response variable

# Fit Ridge Regression (alpha = 0)
ridge_model <- glmnet(x, y, alpha = 0)

# Plot the ridge coefficient paths
plot(ridge_model, xvar = "lambda", label = TRUE)
plot(ridge_model, xvar = "lambda", label = TRUE)
title("Ridge Paths (L2 Regularization)")

set.seed(123)

# Split data
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- data[train_index, ]
y_train <- data[train_index]

# Train Ridge Regression with Cross Validation (alpha = 0 for Ridge)
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0)

# Plot the cross-validation curve
plot(ridge_model)
title("Ridge Regression: Mean Squared Error vs Lambda", line = 2.5)

####
set.seed(123)  # for reproducibility

# Cross-validated LASSO
cv_lasso <- cv.glmnet(x, y_numeric, alpha = 1, family = "binomial")

# Plot cross-validation error vs lambda
plot(cv_lasso)
title("Cross-Validation for LASSO (L1 Regularization)")

set.seed(123)

# Cross-validated Ridge
cv_ridge <- cv.glmnet(x, y_numeric, alpha = 0, family = "binomial")

# Plot
plot(cv_ridge)
title("Cross-Validation for Ridge (L2 Regularization)")
# 1. Get the best lambda value
best_lambda <- cv_ridge$lambda.min

# 2. Fit the final Ridge model with best lambda
final_ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda, family = "binomial")

# 3. Make predictions
ridge_predictions <- predict(final_ridge_model, newx = x_test, type = "response")


#####LDA
# 1. Load necessary package
library(MASS)

# 2. Fit LDA model
lda_model <- lda(Diagnosis ~ ., data = myDF)

# 3. See LDA model summary
print(lda_model)

# 4. Make predictions
lda_predictions <- predict(lda_model, newdata = myDF)

# 5. View predicted classes
head(lda_predictions$class)

# 6. Confusion Matrix
table(Predicted = lda_predictions$class, Actual = myDF$Diagnosis)
# Plot LDA projections
plot(lda_model)
set.seed(123)  # For reproducibility
train_indices <- createDataPartition(myDF$Diagnosis, p = 0.8, list = FALSE)

train_data <- myDF[train_indices, ]
test_data  <- myDF[-train_indices, ]

# 3. Train the LDA model
lda_model <- lda(Diagnosis ~ ., data = train_data)

# 4. Predict on the test data
lda_predictions <- predict(lda_model, newdata = test_data)

# 5. View first few predicted classes
head(lda_predictions$class)

# 6. Confusion matrix to evaluate performance
confusionMatrix(lda_predictions$class, test_data$Diagnosis)




####Ridge confusion matrix
# Load necessary libraries
library(glmnet)
library(caret)
library(e1071)

# Assume your dataset is already loaded and cleaned
# X = predictor variables, y = response variable (factor: 'Benign', 'Malignant')

# Convert to matrix for glmnet
X <- as.matrix(myDF[, -1])  # Replace -1 with actual column index of diagnosis if needed
y <- myDF$Diagnosis         # Assuming Diagnosis is a factor with 2 levels

# Encode response as numeric: glmnet needs binary factor (0/1)
y_numeric <- ifelse(y == "Malignant", 1, 0)

# Split data
set.seed(123)
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y_numeric[train_index]
y_test <- y_numeric[-train_index]

# Ridge regression (alpha = 0)
ridge_model <- cv.glmnet(X_train, y_train, alpha = 0, family = "binomial")

# Predict on test set using best lambda
ridge_pred_prob <- predict(ridge_model, s = ridge_model$lambda.min, newx = X_test, type = "response")
ridge_pred_class <- ifelse(ridge_pred_prob > 0.5, "Malignant", "Benign")

# Convert test labels to original labels
y_test_label <- ifelse(y_test == 1, "Malignant", "Benign")

# Confusion matrix
confusionMatrix(factor(ridge_pred_class, levels = c("Benign", "Malignant")),
                factor(y_test_label, levels = c("Benign", "Malignant")))






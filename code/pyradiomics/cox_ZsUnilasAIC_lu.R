
library(car)
library(survival)
library(glmnet)
library(survC1)
library(MASS)
library(ggplot2)
set.seed(42)

best_aic <- Inf
best_model <- NULL


data <- read.csv("/data/Bladder/pyradiomics/ROI/QDUH_merged_radiomics_features.csv")

col_to_remove = c('影像号', 'GROUP', 
                  '死亡', 'mask', 
                  'event_status','survival_time')

event_status <- data$event_status
survival_time <- data$survival_time

data_value <- data[, !(names(data) %in% col_to_remove)]

data_value_scores <- scale(data_value)

na_columns <- colSums(is.na(data_value_scores)) == nrow(data_value_scores)

data_value_scores <- data_value_scores[, !na_columns]

data_matrix <- cbind(event_status,survival_time,data_value_scores)

data_matrix <- as.data.frame(data_matrix)

data_value_scores_filled <- data_value_scores
data_value_scores_filled[is.na(data_value_scores_filled)] <- 0

selected_features <- c()

for (feature in colnames(data_matrix)[-c(1,2)]) {
  formula <- as.formula(paste("Surv(survival_time, event_status) ~", feature))

  model_univ <- coxph(formula, data = data_matrix)
  
  summary_model <- summary(model_univ)

  p_value <- summary_model$coefficients[1, 5]  # p 值
  

  if (p_value <= 0.05) {
    selected_features <- c(selected_features, feature)
  }
}

print(length(selected_features))

data_matrix_filtered <- data_matrix[, c('survival_time', 'event_status', selected_features)]

n = 0

for (i in 1:5) {
  n = n + 1
  print(n)
  
  x <- makeX(data_matrix_filtered[, selected_features], na.impute = TRUE)
  y <- with(data_matrix_filtered, Surv(survival_time, event_status))
  
  fit <- glmnet(x, y, family = "cox", alpha=1)
  pdf("/home/lab/R/11/coef_path_plot.pdf", width = 800, height = 600)
  plot(fit, xvar = "lambda", label = TRUE, xaxt = "s", yaxt = "s")  # "s"表示显示坐标轴
  dev.off()

  cv_fit <- cv.glmnet(x, y, family = "cox", alpha=1,nfolds=5)
  pdf("/home/lab/R/11/coef_.pdf", width = 800, height = 600)
  plot(cv_fit)
  dev.off()
  lambda_min <- cv_fit$lambda.min
  important_features <- coef(cv_fit, s = "lambda.min")
  
  feature_index <- which(as.numeric(important_features) != 0)


  feature_coef <- as.numeric(important_features)[feature_index]
  feature_name <- rownames(important_features)[feature_index]
  len_feature_name <- paste("lasso-cox选择的特征数量是：", length(feature_name))
  print(length(feature_name))

  data_AIC = makeX(data_matrix_filtered[, c('survival_time', 
                                      'event_status', feature_name)], na.impute = TRUE)
  data_AIC_df <- as.data.frame(data_AIC)
  final_model <- coxph(Surv(survival_time, event_status) ~ ., 
                       data = data_AIC_df)

  step_model <- stepAIC(final_model, direction = "both", trace = 0) 
  
  current_aic <- AIC(step_model)
  if (current_aic < best_aic) {
    best_aic <- current_aic
    best_model <- step_model
  }
}
  

summary_best_model <- summary(best_model)
print(summary_best_model)

print(best_aic)





library(survival)
library(rms)
library(glmnet)

rm(list = ls())

options(warn = -1)

train_data <- read.csv("/data/Bladder/pyradiomics/ROI/QDUH_merged_radiomics_features.csv")

col_to_remove_train = c('影像号', 'GROUP', 
                        '死亡', 'mask', 
                        'event_status','survival_time')

event_status_train <- train_data$event_status
survival_time_train <- train_data$survival_time

data_value_train <- train_data[, !(names(train_data) %in% col_to_remove_train)]

data_value_scores_train <- scale(data_value_train)

na_columns_train <- colSums(is.na(data_value_scores_train)) == nrow(data_value_scores_train)

data_value_scores_train <- data_value_scores_train[, !na_columns_train]

data_matrix_train <- cbind(event_status_train,survival_time_train,data_value_scores_train)

data_matrix_train <- as.data.frame(data_matrix_train)

test_data <- read.csv("/data/Bladder/pyradiomics/ROI/GDPH_merged_radiomics_features.csv")
col_to_remove_test = c('影像号', 'GROUP', 
                       '死亡', 'mask', 
                       'event_status','survival_time')

event_status_test <- test_data$event_status
survival_time_test <- test_data$survival_time
patient_id <- test_data$影像号

data_value_test <- test_data[, !(names(test_data) %in% col_to_remove_test)]

data_value_scores_test <- scale(data_value_test)

na_columns_test <- colSums(is.na(data_value_scores_test)) == nrow(data_value_scores_test)

data_value_scores_test <- data_value_scores_test[, !na_columns_test]

data_matrix_test <- cbind(event_status_test,survival_time_test,data_value_scores_test)

data_matrix_test <- as.data.frame(data_matrix_test)
data_matrix_test[is.na(data_matrix_test)] <- 0

cox_model <- coxph(Surv(survival_time_train, event_status_train) ~ 
        shape_Sphericity_original + glszm_SmallAreaHighGrayLevelEmphasis_original + 
        glcm_ClusterProminence_exponential + glrlm_ShortRunLowGrayLevelEmphasis_exponential + 
        glszm_LowGrayLevelZoneEmphasis_log.sigma.1.0.mm.3D + ngtdm_Strength_log.sigma.1.0.mm.3D + 
         glszm_SmallAreaEmphasis_logarithm + 
        gldm_DependenceVariance_squareroot + firstorder_Maximum_wavelet.HLL + 
        glszm_ZoneEntropy_wavelet.HLH + ngtdm_Strength_wavelet.HLH + 
        glszm_GrayLevelNonUniformityNormalized_log.sigma.3.0.mm.3D + glszm_SmallAreaEmphasis_log.sigma.5.0.mm.3D,
        data = data_matrix_train)
summary(cox_model)


aic_value <- AIC(cox_model)

data_matrix_test$predicted_risk <- predict(cox_model, newdata = data_matrix_test, type = "risk")

concordance <- survConcordance(Surv(survival_time_test, event_status_test) ~ data_matrix_test$predicted_risk, data = data_matrix_test)
c_index <- concordance$concordance

compute_c_index <- function(data, indices) {
 
  d <- data[indices, ]

  concordance <- survConcordance(Surv(d$survival_time_test, d$event_status_test) ~ d$predicted_risk)
  return(concordance$concordance)
}

library(boot)
set.seed(123) 

results <- boot(data = data_matrix_test, statistic = compute_c_index, R = 1000)

ci <- boot.ci(results, type = "perc")

print(paste("GDPH:", c_index))
cat("C-index 95% 置信区间: [", ci$percent[4], ", ", ci$percent[5], "]\n")





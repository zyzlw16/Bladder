#####IBS#####
library(pec)
library(riskRegression)
library(rms)
library(Hmisc)
library(survival)
setwd("C:/Users/")
dt_train_pre<- read.csv('.csv')
dd <- datadist(dt_train_pre)  
options(datadist = "dd") 
Models <- list("Model1"= coxph(Surv(TIME,Status)~BCDL, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model2"= coxph(Surv(TIME,Status)~DenseNet121, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model3"= coxph(Surv(TIME,Status)~Resnet18, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model4"= coxph(Surv(TIME,Status)~Resnet34, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model5"= coxph(Surv(TIME,Status)~Swin, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model6"= coxph(Surv(TIME,Status)~Radiomics, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model7"= coxph(Surv(TIME,Status)~Clinical, data=dt_train_pre, x=TRUE, y=TRUE),
               "Model8"= coxph(Surv(TIME,Status)~Combined, data=dt_train_pre, x=TRUE, y=TRUE))
p <- pec(object = Models,formula=Surv(TIME,Status)~BCDL, data=dt_train_pre, splitMethod="Boot632plus", B=1000, reference = TRUE)
print(p, times=seq(5,60,5))

opar <- par(no.readonly=TRUE)
par(mfrow = c(1, 1))

plot(p,type="l",smooth=TRUE,legend = FALSE,xlim=c(0,60),axis1.at=seq(0,60,12), ylim=c(0,0.2),
     xlab="Follow-up Time (months)", ylab="Prediction error",col = c("Red", "Orange", "Yellow", "Green","Cyan","Blue","Purple","Brown"),
     lwd = c(3,3,3,3,3,3,3,3),lty = c(1,1,1,1,1,1,1,1))
legend("topleft", c("BCDL", "DenseNet121", "Resnet18", "Resnet34","Swin","Radiomics","Clinical","Combined"),
       lty = c(1,1,1,1,1,1,1,1), lwd = c(3,3,3,3,3,3,3,3), col = c("Red", "Orange", "Yellow", "Green","Cyan","Blue","Purple","Brown"), bty = "n")

#####TIME-ROC#####
library(timeROC)
library(survival)
library(risksetROC)
library(openxlsx)

options(warn = -1)
setwd("C:/Users/")
data_BCDL <- read.csv('BCDL.csv')
data_DenseNet121 <- read.csv('DenseNet121.csv')
data_Resnet34 <- read.csv('Resnet34.csv')
data_Resnet18 <- read.csv('Resnet18.csv')
data_Swin <- read.csv('Swin.csv')
data_Radiomics <- read.csv('Radiomics.csv')
data_Clinical <- read.csv('Clinical.csv')
data_Combined <- read.csv('Combined.csv')

# ROC
roc_BCDL <- timeROC(T=data_BCDL$TIME,           # 生存时间
                    delta=data_BCDL$Status,        # 事件状???
                    marker=data_BCDL$BCDL, # 风险分数
                    # weighting="cox",
                    cause=1,                               
                    times=seq(12,60,1),        # 计算时间序列
                    iid = TRUE)                   # 是否计算AUC的独立同分布估计

roc_DenseNet121 <- timeROC(T=data_DenseNet121$TIME,           # 生存时间
                           delta=data_DenseNet121$Status,        # 事件状???
                           marker=data_DenseNet121$DenseNet121, # 风险分数
                           # weighting="cox",
                           cause=1,                               
                           times=seq(12,60,1),        # 计算时间序列
                           iid = TRUE) 

roc_Resnet34 <- timeROC(T=data_Resnet34$TIME,           # 生存时间
                        delta=data_Resnet34$Status,        # 事件状???
                        marker=data_Resnet34$Resnet34, # 风险分数
                        # weighting="cox",
                        cause=1,                               
                        times=seq(12,60,1),        # 计算时间序列
                        iid = TRUE)

roc_Resnet18 <- timeROC(T=data_Resnet18$TIME,           # 生存时间
                        delta=data_Resnet18$Status,        # 事件状???
                        marker=data_Resnet18$Resnet18, # 风险分数
                        # weighting="cox",
                        cause=1,                               
                        times=seq(12,60,1),        # 计算时间序列
                        iid = TRUE)

roc_Swin <- timeROC(T=data_Swin$TIME,           # 生存时间
                    delta=data_Swin$Status,        # 事件状???
                    marker=data_Swin$Swin, # 风险分数
                    # weighting="cox",
                    cause=1,                               
                    times=seq(12,60,1),        # 计算时间序列
                    iid = TRUE)

roc_Radiomics <- timeROC(T=data_Radiomics$TIME,           # 生存时间
                         delta=data_Radiomics$Status,        # 事件状???
                         marker=data_Radiomics$Radiomics, # 风险分数
                         # weighting="cox",
                         cause=1,                               
                         times=seq(12,60,1),        # 计算时间序列
                         iid = TRUE)

roc_Clinical <- timeROC(T=data_Clinical$TIME,           # 生存时间
                        delta=data_Clinical$Status,        # 事件状???
                        marker=data_Clinical$Clinical, # 风险分数
                        # weighting="cox",
                        cause=1,                               
                        times=seq(12,60,1),        # 计算时间序列
                        iid = TRUE)
roc_Combined <- timeROC(T=data_Combined$TIME,           # 生存时间
                        delta=data_Combined$Status,        # 事件状???
                        marker=data_Combined$Combined, # 风险分数
                        # weighting="cox",
                        cause=1,                               
                        times=seq(12,60,1),        # 计算时间序列
                        iid = TRUE)
auc_BCDL <- round(roc_BCDL$AUC[c('t=60')],4)
auc_BCDL <- sprintf("%.5f", auc_BCDL)
print(auc_BCDL)

auc_DenseNet121 <- round(roc_DenseNet121$AUC[c('t=60')],4)
auc_DenseNet121 <- sprintf("%.5f", auc_DenseNet121)
print(auc_DenseNet121)
#AUC and 95%CI       # 735   1825
auc_Resnet34 <- round(roc_Resnet34$AUC[c('t=60')],4)  
auc_Resnet34 <- sprintf("%.5f", auc_Resnet34)
print(auc_Resnet34)
# confint(roc_clin,level = 0.95)$CI_AUC
auc_Resnet18 <- round(roc_Resnet18$AUC[c('t=60')],4)  
auc_Resnet18 <- sprintf("%.5f", auc_Resnet18)
print(auc_Resnet18)

auc_Swin <- round(roc_Swin$AUC[c('t=60')],4)  
auc_Swin <- sprintf("%.5f", auc_Swin)
print(auc_Swin)

auc_Radiomics <- round(roc_Radiomics$AUC[c('t=60')],4)  
auc_Radiomics <- sprintf("%.5f", auc_Radiomics)
print(auc_Radiomics)

auc_Clinical <- round(roc_Clinical$AUC[c('t=60')],4)  
auc_Clinical <- sprintf("%.5f", auc_Clinical)
print(auc_Clinical)

auc_Combined <- round(roc_Combined$AUC[c('t=60')],4)  
auc_Combined <- sprintf("%.5f", auc_Combined)
print(auc_Combined)
# Plot the ROC curve
#pdf("D:/R_project/MPIS-LUNG-main/R_codes/ROC_1825.pdf", width = 6, height = 4.5, family = "Times", onefile = FALSE)
# plot(dt, xlab="Risk scores for OS", ylab="Risk scores for DFS", col="blue", cex.axis=1.5,cex.lab=1.5,cex.main=2)
plot(roc_BCDL,time=60,col="Red",lwd=2,title = FALSE)
plot(roc_DenseNet121,time=60,col="Orange",lwd=2,title = FALSE, add=TRUE)
plot(roc_Resnet34,time=60,col="Yellow",lwd=2,title = FALSE, add=TRUE)
plot(roc_Resnet18,time=60,col="Green",lwd=2,title = FALSE, add=TRUE)
plot(roc_Swin,time=60,col="Cyan",lwd=2,title = FALSE, add=TRUE)
plot(roc_Radiomics,time=60,col="Blue",lwd=2,title = FALSE, add=TRUE)
plot(roc_Clinical,time=60,col="Purple",lwd=2,title = FALSE, add=TRUE)
plot(roc_Combined,time=60,col="Brown",lwd=2,title = FALSE, add=TRUE)
title(main=list("ROC(D1)",cex=1.25,col="black",font=1))
legend("bottomright", c(paste("BCDL (AUC = ", auc_BCDL, ")", sep = ""),
                        paste("DenseNet121 (AUC = ", auc_DenseNet121, ")", sep = ""),
                        paste("Resnet34 (AUC = ", auc_Resnet34, ")", sep = ""),
                        paste("Resnet18 (AUC = ", auc_Resnet18, ")", sep = ""),
                        paste("Swin (AUC = ", auc_Swin, ")", sep = ""),
                        paste("Radiomics (AUC = ", auc_Radiomics, ")", sep = ""),
                        paste("Clinical (AUC = ", auc_Clinical, ")", sep = ""),
                        paste("Combined (AUC = ", auc_Combined, ")", sep = "")),
       col=c("Red","Orange","Yellow","Green","Cyan","Blue","Purple","Brown"),lty=1,lwd=2,bty ="n")
#dev.off()

# Plot the AUC curve
# png(filename="D:/projects/R_project/R_lung_TILs/OS_ROCAUC_figs/AUC-gdph_m48.png",width = 400,height = 280)
#pdf("D:/R_project/MPIS-LUNG-main/R_codes/AUC_1825.pdf", width = 6, height = 4.5, family = "Times", onefile = FALSE)
plotAUCcurve(roc_BCDL,col = "Red")  # 修改plotAUCcurve库文件的xy轴标???
plotAUCcurve(roc_DenseNet121,col ="Orange",add=TRUE)
plotAUCcurve(roc_Resnet34,col ="Yellow",add=TRUE)
plotAUCcurve(roc_Resnet18,col ="Green",add=TRUE)
plotAUCcurve(roc_Swin,col ="Cyan",add=TRUE)
plotAUCcurve(roc_Radiomics,col ="Blue",add=TRUE)
plotAUCcurve(roc_Clinical,col ="Purple",add=TRUE)
plotAUCcurve(roc_Combined,col ="Brown",add=TRUE)
title(main=list("AUC(D1)",cex=1.25,col="black",font=1))
legend("topright",c("BCDL","DenseNet121","Resnet34","Resnet18","Swin","Radiomics","Clinical","Combined"), col=c("Red","Orange","Yellow","Green","Cyan","Blue","Purple","Brown"),lty=1,lwd=2,bty="n")
#dev.off()

#####Cindex的bootstrap#####
# 安装并加载必要的包
library(survival)
library(ggplot2)

# 读取数据
data <- read.csv("")

# 提取时间和状态
time <- data$TIME
status <- data$Status

# 准备模型名称
model_columns <- colnames(data)[3:ncol(data)] # 从第三列到最后一列

# 创建一个空的数据框用于存储结果
results <- data.frame(Model = character(),
                      C_Index = numeric(),
                      CI_Lower = numeric(),
                      CI_Upper = numeric(),
                      stringsAsFactors = FALSE)

# 计算每个模型的C-index和置信区间
for (model in model_columns) {
  predictions <- data[[model]]
  
  # 创建生存对象
  surv_obj <- Surv(time, status)
  
  # 计算C-index
  c_index <- survConcordance(surv_obj ~ predictions)
  
  # 计算C-index值
  c_index_value <- c_index$concordance
  
  # Bootstrapping方法计算置信区间
  set.seed(123) # 为可重复性设置随机种子
  n_iterations <- 2000
  boot_c_indices <- numeric(n_iterations)
  
  for (i in 1:n_iterations) {
    # 重采样
    sample_indices <- sample(1:length(time), replace = TRUE)
    sample_time <- time[sample_indices]
    sample_status <- status[sample_indices]
    sample_predictions <- predictions[sample_indices]
    
    # 创建生存对象
    boot_surv_obj <- Surv(sample_time, sample_status)
    
    # 计算Bootstrapped C-index
    boot_c_index <- survConcordance(boot_surv_obj ~ sample_predictions)
    boot_c_indices[i] <- boot_c_index$concordance
  }
  
  # 计算置信区间
  ci_lower <- quantile(boot_c_indices, 0.025)
  ci_upper <- quantile(boot_c_indices, 0.975)
  
  # 将结果添加到数据框
  results <- rbind(results, data.frame(Model = model,
                                       C_Index = c_index_value,
                                       CI_Lower = ci_lower,
                                       CI_Upper = ci_upper))
}

# 打印结果
print(results)
# 随机重采样100次并绘制黑点
set.seed(456)
random_samples <- data.frame()

for (model in model_columns) {
  predictions <- data[[model]]
  
  for (i in 1:100) {
    sample_indices <- sample(1:length(time), replace = TRUE)
    sample_time <- time[sample_indices]
    sample_status <- status[sample_indices]
    sample_predictions <- predictions[sample_indices]
    
    sample_surv_obj <- Surv(sample_time, sample_status)
    sample_c_index <- survConcordance(sample_surv_obj ~ sample_predictions)$concordance
    
    random_samples <- rbind(random_samples, data.frame(Model = model, C_Index = sample_c_index))
  }
}

# 自定义颜色
custom_colors <- c("BCDL" = "#fccccb", 
                   "DenseNet121" = "#bdb5e1", 
                   "Resnet10" = "#b0d992", 
                   "Resnet34" = "#f9d580",
                   "Swin" = "#99b9e9", 
                   "Radiomics" = "#e3716e", 
                   "Clinical" = "#eca680",
                   "Combined"="#00A087B2")

# 按C_Index升序排列结果
results <- results[order(results$C_Index), ]

# 绘制柱状图，确保模型按照C-index升序排列
ggplot(results, aes(x = reorder(Model, C_Index), y = C_Index, fill = Model)) +
  geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2, color = "black") +
  geom_jitter(data = random_samples, aes(y = C_Index), shape = 21, color = "black", fill = "black", size = 0.8, alpha = 0.8, width = 0.2) +
  labs(title = "C-index for Each Model with CI and Random Samples",
       x = "Model",
       y = "C-index") +
  scale_fill_manual(values = custom_colors) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 保存图像
ggsave("", plot = p, width = 8, height = 6)


##### Calibration plot#####
library(tidyverse)
library(caret)
library(pROC)
library(glmnet)
library(DMwR2)
library(rmda)
library(ggpubr)
library(ModelGood)
library(rms)
library(mRMRe)
library(DescTools)
library(Boruta)
library(sva)
library(e1071)
library(survcomp)

setwd("C:/Users/")
dt_train_pre1 <- read_csv('BCDL.csv')  
dt_train_pre2 <- read_csv('DenseNet121.csv')  
dt_train_pre3 <- read_csv('Resnet34.csv')  
dt_train_pre4 <- read_csv('Resnet18.csv')  
dt_train_pre5 <- read_csv('Swin.csv')  
dt_train_pre6 <- read_csv('Radiomics.csv')  
dt_train_pre7 <- read_csv('Clinical.csv')  
dt_train_pre8 <- read_csv('Combined.csv')  

units(dt_train_pre1$TIME) <- "Month"
dd=datadist(dt_train_pre1)
options(datadist="dd")
f1 <- cph(Surv(dt_train_pre1$TIME,dt_train_pre1$Status==1)~BCDL, data=dt_train_pre1, x=TRUE, y=TRUE, surv = TRUE, time.inc = 60)
cal1 <- calibrate(f1, cmethod = "KM", method="boot",u=60,m=14,B=1000)
#m*5近似等于样本量，评价时间点不能小于最大时间1/25，不能超过最大随访时间

units(dt_train_pre2$TIME) <- "Month"
dd=datadist(dt_train_pre2)
options(datadist="dd")
f2 <- cph(Surv(dt_train_pre2$TIME,dt_train_pre2$Status==1)~DenseNet121, data=dt_train_pre2, x=TRUE, y=TRUE, surv = TRUE, time.inc = 60)
cal2 <- calibrate(f2, cmethod = "KM", method="boot",u=60,m=14,B=1000)

units(dt_train_pre3$TIME) <- "Month"
dd=datadist(dt_train_pre3)
options(datadist="dd")
f3 <- cph(Surv(dt_train_pre3$TIME,dt_train_pre3$Status==1)~Resnet34, data=dt_train_pre3, x=TRUE, y=TRUE, surv = TRUE, time.inc = 60)
cal3 <- calibrate(f3, cmethod = "KM", method="boot",u=60,m=14,B=1000)

units(dt_train_pre4$TIME) <- "Month"
dd=datadist(dt_train_pre4)
options(datadist="dd")
f4 <- cph(Surv(dt_train_pre4$TIME,dt_train_pre4$Status==1)~Resnet18, data=dt_train_pre4, x=TRUE, y=TRUE, surv = TRUE, time.inc =60)
cal4 <- calibrate(f4, cmethod = "KM", method="boot",u=60,m=14,B=1000)

units(dt_train_pre5$TIME) <- "Month"
dd=datadist(dt_train_pre5)
options(datadist="dd")
f5 <- cph(Surv(dt_train_pre5$TIME,dt_train_pre5$Status==1)~Swin, data=dt_train_pre5, x=TRUE, y=TRUE, surv = TRUE, time.inc =60)
cal5 <- calibrate(f5, cmethod = "KM", method="boot",u=60,m=14,B=1000)

units(dt_train_pre6$TIME) <- "Month"
dd=datadist(dt_train_pre6)
options(datadist="dd")
f6 <- cph(Surv(dt_train_pre6$TIME,dt_train_pre6$Status==1)~Radiomics, data=dt_train_pre6, x=TRUE, y=TRUE, surv = TRUE, time.inc =60)
cal6 <- calibrate(f6, cmethod = "KM", method="boot",u=60,m=14,B=1000)

units(dt_train_pre7$TIME) <- "Month"
dd=datadist(dt_train_pre7)
options(datadist="dd")
f7 <- cph(Surv(dt_train_pre7$TIME,dt_train_pre7$Status==1)~Clinical, data=dt_train_pre7, x=TRUE, y=TRUE, surv = TRUE, time.inc =60)
cal7 <- calibrate(f7, cmethod = "KM", method="boot",u=60,m=14,B=1000)

units(dt_train_pre8$TIME) <- "Month"
dd=datadist(dt_train_pre8)
options(datadist="dd")
f8 <- cph(Surv(dt_train_pre8$TIME,dt_train_pre8$Status==1)~Combined, data=dt_train_pre8, x=TRUE, y=TRUE, surv = TRUE, time.inc =60)
cal8 <- calibrate(f8, cmethod = "KM", method="boot",u=60,m=14,B=1000)

#校准曲线训练???
opar <- par(no.readonly=TRUE)
par(mfrow = c(1, 2))

plot(cal1, errbar.col = "Red",lwd = 2,lty=2, cex.axis =1.2, cex.lab = 1.2,xlab="Model predicted survival probability", ylab="Observed survival (probability)", xlim = c(0,1),ylim = c(0,1), subtitles = FALSE)
lines(cal1[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Red",pch = 16)
par(new=TRUE)
lines(cal2[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Orange",pch = 16)
par(new=TRUE)
lines(cal3[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Yellow",pch = 16)
par(new=TRUE)
lines(cal4[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Green",pch = 16)
par(new=TRUE)
lines(cal5[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Cyan",pch = 16)
par(new=TRUE)
lines(cal6[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Blue",pch = 16)
par(new=TRUE)
lines(cal7[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Purple",pch = 16)
par(new=TRUE)
lines(cal8[,c("mean.predicted","KM")],type = "b",lwd = 2,col = "Brown",pch = 16)
par(new=TRUE)
legend("topleft", c("BCDL","DenseNet141","Resnet34","Resnet18","Swin","Radioics","Clinical","Combined"),
       lty = c(1,1,1,1,1,1,1,1), lwd = c(2,2,2,2,2,2,2,2), col = c("Red","Orange","Yellow","Green","Cyan","Blue","Purple","Brown"), bty = "n")
abline(0,1,col="black",lty=2,lwd=1)


#####DCA#####
library(survival)
source("/dca.R")
source("/stdca.R")
df_surv <- read.csv("C:/Users/",header = T)
# 查看数据结构
dim(df_surv)
str(df_surv)
# 建立多个模型
BCDL <- coxph(Surv(TIME, Status) ~ BCDL, data = df_surv)
DenseNet121 <- coxph(Surv(TIME, Status) ~ DenseNet121, data = df_surv)
Resnet18 <- coxph(Surv(TIME, Status) ~ Resnet18, data = df_surv)
Resnet34 <- coxph(Surv(TIME, Status) ~ Resnet34, data = df_surv)
Swin <- coxph(Surv(TIME, Status) ~ Swin, data = df_surv)
Radiomics <- coxph(Surv(TIME, Status) ~ Radiomics, data = df_surv)
Clinical <- coxph(Surv(TIME, Status) ~ Clinical, data = df_surv)
Combined <- coxph(Surv(TIME, Status) ~ Combined, data = df_surv)
# 计算每个模型的概???
df_surv$BCDL <- c(1-(summary(survfit(BCDL, newdata=df_surv), times=60)$surv))
df_surv$DenseNet121 <- c(1-(summary(survfit(DenseNet121, newdata=df_surv), times=60)$surv))
df_surv$Resnet18 <- c(1-(summary(survfit(Resnet18, newdata=df_surv), times=60)$surv))
df_surv$Resnet34 <- c(1-(summary(survfit(Resnet34, newdata=df_surv), times=60)$surv))
df_surv$Swin <- c(1-(summary(survfit(Swin, newdata=df_surv), times=60)$surv))
df_surv$Radiomics <- c(1-(summary(survfit(Radiomics, newdata=df_surv), times=60)$surv))
df_surv$Clinical <- c(1-(summary(survfit(Clinical, newdata=df_surv), times=60)$surv))
df_surv$Combined <- c(1-(summary(survfit(Combined, newdata=df_surv), times=60)$surv))
# 画图
stdca(data=df_surv, 
      outcome="Status", 
      ttoutcome="TIME", 
      timepoint=60, 
      predictors=c("BCDL","DenseNet121","Resnet18","Resnet34","Swin","Radiomics","Clinical","Combined"),
      smooth=FALSE)
#####KM#####
require("survival")  
library(survminer)  

# 读取 CSV 格式的数据文件   
mydata <- read.csv("C:/Users")  

cutoff <- 4.02
mydata$risk_group <- ifelse(mydata$pre1 <= cutoff, "Low", "High")
mydata$risk_group <- factor(mydata$risk_group, levels = c("Low", "High"))

# 拟合生存模型  
fit <- survfit(Surv(TIME, Status) ~ risk_group, data = mydata) 
summary(fit)

# 拟合 Cox 比例风险模型  
model <- coxph(Surv(TIME, Status) ~ risk_group, data = mydata)  

# 计算HR及其95%可信区间  
cox_summary <- summary(model)  
hr <- exp(cox_summary$coefficients[1,"coef"]) 
hr_lower <- cox_summary$conf.int[1, "lower .95"]  
hr_upper <- cox_summary$conf.int[1, "upper .95"]


# 计算 log-rank 检验的 p 值  
survdiff_result <- survdiff(Surv(TIME, Status) ~ risk_group, data = mydata)  
p_value <- 1 - pchisq(survdiff_result$chisq, df = 1)  

# 绘制生存曲线  
ggsurv <- ggsurvplot(  
  fit,  
  data = mydata,  
  risk.table = TRUE,  
  pval = FALSE,  # 不在图中自动显示 p 值  
  conf.int = TRUE,  
  palette = c("#2E9FDF", "#E7B800"),  
  xlim = c(0,60),  
  xlab = "TIME",  
  break.time.by = 10,  
  ggtheme = theme_light(),  
  risk.table.y.text.col = TRUE,  
  risk.table.height = 0.25,  
  risk.table.y.text = FALSE,  
  ncensor.plot = TRUE,  
  ncensor.plot.height = 0.25,  
  conf.int.style = "ribbon",  
  surv.median.line = "hv",  
  legend.labs = c("Low", "High")  
)  
# 添加风险比 (HR) 和 p 值到生存曲线图  
ggsurv$plot <- ggsurv$plot +  
  ggplot2::annotate("text", x = 5, y = 0.1,  
                    label = paste("HR =", round(hr, 5),  
                                  "\np =", sprintf("%.5f", p_value)),  
                    size = 4.5)  
print(ggsurv)  
# 进行生存差异检验  
print(survdiff_result) 
print(fit) # 这会显示包含中位生存时间的摘要信息  
# 或者更具体地：  
summary(fit)$table["median"]
median_surv <- surv_median(fit)  
print(median_surv)
# 按风险组分别计算中位生存时间  
median_by_group <- summary(fit)$table[,"median"]  
print(median_by_group)

######PSM######
library(tidyverse)  
library(MatchIt)  
library(compareGroups)  

# 读取数据  
data <- read.csv("C:/Users/.csv", header = TRUE)  

# 数据预处理  
data <- data %>%   
  #select(-ID) %>%  # 移除ID列  
  mutate(TREAT = as.factor(treatment),  
         Label = as.factor(Status))  

# 查看处理组的分布  
table(data$TREAT)  

# 匹配前的描述性统计  
descrTable(risk ~ ., data = data)  

# 进行倾向性评分匹配  
match.it <- matchit(TREAT ~ Age + Gender + Focus.location + Shape + Size +   
                      Calcification + Cystic.necrosis + Tumor.boundary +   
                      Number + Stalk + Extramural.infiltration +   
                      CT.value.of.lesion.in.nephrographic.phase + MIS,  
                    data = data,  
                    method = "nearest", # 最近邻匹配法  
                    ratio = 1)  # 1:1匹配  

# 输出匹配结果摘要  
summary(match.it)  

# 可视化匹配结果  
plot(match.it, type = "jitter", interactive = FALSE)  

# 提取匹配后的数据  
matchdata <- match.data(match.it)  

# 匹配后的描述性统计  
descrTable(risk ~ ., data = matchdata)  

# 保存匹配后的数据  
write.csv(matchdata, "C:/Users/PSM/Group-ALL.csv", row.names = FALSE)


#####KM#####
# 加载必要的包  
library(survminer)  
library(survival)  

# 读取数据  
mydata <- read.csv("C:/Users", header = TRUE)  

# 查看数据结构  
str(mydata)  

# 创建生存对象  
surv_object <- Surv(mydata$TIME, mydata$Status)  

# 拟合Cox比例风险模型以计算HR  
fit_cox <- coxph(surv_object ~ risk, data = mydata)  

# 提取HR和p值  
summary_cox <- summary(fit_cox)  
hr <- exp(coef(fit_cox))  
p_value <- summary_cox$coefficients[,'Pr(>|z|)']  

# 拟合生存曲线  
fit <- survfit(surv_object ~ risk, data = mydata)  

# 绘制K-M曲线并添加HR信息  
gg <- ggsurvplot(  
  fit,   
  data = mydata,  
  risk.table = TRUE,  
  pval = FALSE, # 若您使用log-rank检验，该p值会自动显示  
  conf.int = TRUE,  
  palette = c("#2E9FDF", "#E7B800"),  
  xlim = c(0, 60),  
  xlab = "TIME",  
  break.time.by = 10,  
  ggtheme = theme_light(),  
  risk.table.y.text.col = TRUE,  
  risk.table.height = 0.25,  
  risk.table.y.text = FALSE,  
  ncensor.plot = TRUE,  
  ncensor.plot.height = 0.25,  
  conf.int.style = "ribbon",  
  surv.median.line = "hv",  
  legend.labs = c("None", "Adjuvant therapy")  
)  

# 在图中添加HR和p值  
gg$plot <- gg$plot +  
  ggplot2::annotate("text", x = 30, y = 0.2,   
                    label = paste("HR =", round(hr, 2),   
                                  "\np =", sprintf("%.2f", p_value)),  
                    size = 4.5)  
# 打印图形  
print(gg)

"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity.

"""
## LOAD PACKAGES
libs <- c("dplyr","ggplot2","tidyr","wesanderson","grid","gridExtra","viridis","stringr","flextable","irrNA","jrt","DescTools","extrafont")
new_libs <- libs[!(libs %in% installed.packages()[,"Package"])]
if(length(new_libs)) install.packages(new_libs)
lapply(libs, library, character.only = T)

## DEF FUNCTIONS
clip <- function(value, UB=1, LB=-1) pmax(LB, pmin(value, UB)) ##clipping function for disattenuated correlations

## IMPORT FONTS
font_import()

#########
#######
###
###  COMPUTE CORRELATIONS & MSE
###
#######
#########
val <- read.csv("validation_storage_dataframe_revised.csv")
test <- read.csv("test_storage_dataframe_revised.csv")
rg1 <- read.csv("rg1_output_dataframe_revised.csv")
rg2 <- read.csv("rg2_output_dataframe_revised.csv")
fg <- read.csv("fargen_output_dataframe_revised.csv")
rg1$ratings <- as.numeric(gsub("\\[|\\]", "", rg1$ratings))
rg2$ratings <- as.numeric(gsub("\\[|\\]", "", rg2$ratings))
fg$ratings <- as.numeric(gsub("\\[|\\]", "", fg$ratings))
rg1$predictions <- as.numeric(gsub("\\[|\\]", "", rg1$predictions))
rg2$predictions <- as.numeric(gsub("\\[|\\]", "", rg2$predictions))
fg$predictions <- as.numeric(gsub("\\[|\\]", "", fg$predictions))
rg1$ink <- as.numeric(gsub("\\[|\\]", "", rg1$ink))
rg2$ink <- as.numeric(gsub("\\[|\\]", "", rg2$ink))
fg$ink <- as.numeric(gsub("\\[|\\]", "", fg$ink))

## validation cor + 95% CI, and mean squared error
(valcor <- cor.test(val$ratings, val$predictions)$estimate)
(vallb <- cor.test(val$ratings, val$predictions)$conf.int[1]) # lower bound
(valub <- cor.test(val$ratings, val$predictions)$conf.int[2]) # upper bound
(valmse <- mean((val$ratings-val$predictions)^2)) # MSE

## validation ink cor + 95% CI
(valinkcor <- cor.test(val$ratings, val$ink)$estimate)
(valinklb <- cor.test(val$ratings, val$ink)$conf.int[1])
(valinkub <- cor.test(val$ratings, val$ink)$conf.int[2])

## test cor + 95% CI, and mean squared error
(testcor <- cor.test(test$ratings, test$predictions)$estimate)
(testlb <- cor.test(test$ratings, test$predictions)$conf.int[1])
(testub <- cor.test(test$ratings, test$predictions)$conf.int[2])
(testmse<- mean((test$ratings-test$predictions)^2))

## test ink cor + 95% CI
(testinkcor <- cor.test(test$ratings, test$ink)$estimate)
(testinklb <- cor.test(test$ratings, test$ink)$conf.int[1])
(testinkub <- cor.test(test$ratings, test$ink)$conf.int[2])

## rater gen 1 cor + 95% CI, and mean squared error
(rg1cor <- cor.test(rg1$ratings, rg1$predictions)$estimate)
(rg1lb <- cor.test(rg1$ratings, rg1$predictions)$conf.int[1])
(rg1ub <- cor.test(rg1$ratings, rg1$predictions)$conf.int[2])
(rg1mse <- mean((rg1$ratings-rg1$predictions)^2))

## rater gen 1 ink cor + 95% CI
(rg1inkcor <- cor.test(rg1$ratings, rg1$ink)$estimate)
(rg1inklb <- cor.test(rg1$ratings, rg1$ink)$conf.int[1])
(rg1inkub <- cor.test(rg1$ratings, rg1$ink)$conf.int[2])

## rater gen 2 cor + 95% CI, and mean squared error
(rg2cor <- cor.test(rg2$ratings, rg2$predictions)$estimate)
(rg2lb <- cor.test(rg2$ratings, rg2$predictions)$conf.int[1])
(rg2ub <- cor.test(rg2$ratings, rg2$predictions)$conf.int[2])
(rg2mse <- mean((rg2$ratings-rg2$predictions)^2))

## rater gen 2 ink cor + 95% CI
(rg2inkcor <- cor.test(rg2$ratings, rg2$ink)$estimate)
(rg2inklb <- cor.test(rg2$ratings, rg2$ink)$conf.int[1])
(rg2inkub <- cor.test(rg2$ratings, rg2$ink)$conf.int[2])

## far gen cor + 95% CI, and mean squared error
(fgcor <- cor.test(fg$ratings, fg$predictions)$estimate)
(fglb <- cor.test(fg$ratings, fg$predictions)$conf.int[1])
(fgub <- cor.test(fg$ratings, fg$predictions)$conf.int[2])
(fgmse <- mean((fg$ratings-fg$predictions)^2))

## far gen ink cor + 95% CI
(fginkcor <- cor.test(fg$ratings, fg$ink)$estimate)
(fginklb <- cor.test(fg$ratings, fg$ink)$conf.int[1])
(fginkub <- cor.test(fg$ratings, fg$ink)$conf.int[2])


#########
#######
###
###    CREATE TABLE 1: DATASET DESCRIPTORS AND CORRELATIONS
###
#######
#########

## Code to create dfs for val/test splits for raw rating data
orig <- read.csv('primary_rating_raw.csv')
orig$blockid <- ifelse(orig$Block == "Block1", "_4", ifelse(orig$Block == "Block2","_7",ifelse(orig$Block == "Block3","_8", ifelse(orig$Block == "Block4","_9", ifelse(orig$Block == "Block5","_11",ifelse(orig$Block == "Block6","_12",ifelse(orig$Block == "Block7","_13",ifelse(orig$Block == "Block8","_3",ifelse(orig$Block == "Block9","_15", ifelse(orig$Block == "Block10","_17",ifelse(orig$Block == "Block11","_19",ifelse(orig$Block == "Block12","_56", ifelse(orig$Block == "Common","_4","error")))))))))))))
orig$fname <- paste(orig$ImageID,orig$blockid,sep='')

## Save subsetted dataframes
val_ratings_raw <- subset(orig, fname %in% val$fname)
test_ratings_raw <- subset(orig, fname %in% test$fname)
train_ratings_raw <- subset(orig, !(fname %in% val$fname) & !(fname %in% test$fname))
write.csv(val_ratings_raw, 'primary_val_rating_raw.csv')
write.csv(test_ratings_raw, 'primary_test_rating_raw.csv')
write.csv(train_ratings_raw, 'primary_train_rating_raw.csv')

## Train ICC (not included in paper)
ratings <- train_ratings_raw[,startsWith(names(train_ratings_raw), "rater")]
ratings <- mutate_all(ratings, function(x) as.numeric(as.character(x)))
iccdf <- as.data.frame(iccNA(ratings)$ICCs)
colnames(iccdf) <- make.names(names(iccdf))
icc <- round(iccdf$ICC[6],2) ## get average consistency ICC--ICC(C,k)
lb <- round(iccdf$lower.CI.limit[6],2) ## upper bound
ub <- round(iccdf$upper.CI.limit[6],2) ## lower bound

## Set up table variables
check <- "\U2713"
ex <- "\U2717"
set_variables <- list(
  c("Primary: validation","1104","50","0.81\n[.79, .83]","primary_val_rating_raw.csv",ex,check,ex, ".53\n[.49, .57]"),
  c("Primary: test", "2216","50","0.80\n[.79, .82]","primary_test_rating_raw.csv",ex, check,ex, ".41\n[.37, .44]"),
  c("Rater generalization 1","670", "3","0.76\n[.73, .79]","ratergen1_raw.csv",ex, check, check, ".55\n[.5, .6]"),
  c("Rater generalization 2", "722", "6","0.65\n[.61, .69]","ratergen2_raw.csv",ex,check,check, ".33\n[.26, .39]"),
  c("Rater + task generalization","679", "3","0.49\n[.43, .54]","fargen_raw.csv",check, check, check,".40\n[.34, .46]")
)

## Generate new dataframe with ICC (for val/test/generalization)
allcors <- c()  ## empty vector to store (only correlations relevant to training task)
allns <- c()  ## empty vector to store n per dataset (only datasets taht match training set)
for (i in 1:length(set_variables)) {
  cur_vars <- set_variables[[i]]
  dataset_name <- cur_vars[1]
  num_items <- cur_vars[2]
  num_raters <- cur_vars[3]
  correlation <- cur_vars[4]
  ink <- cur_vars[9]
  if(i < length(set_variables)){
    allcors[i] <- as.numeric(correlation)
    allns[i] <- as.numeric(num_items)
    }
  cur_dataset <- read.csv(cur_vars[5])
  new_items <- cur_vars[7]
  new_raters <- cur_vars[8]
  new_task <- cur_vars[6]
  ratings <- cur_dataset[,startsWith(names(cur_dataset), "rater")]
  ratings <- mutate_all(ratings, function(x) as.numeric(as.character(x)))
  iccdf <- as.data.frame(iccNA(ratings)$ICCs)
  colnames(iccdf) <- make.names(names(iccdf))
  icc <- round(iccdf$ICC[6],2) ## get average consistency ICC--ICC(C,k)
  lb <- round(iccdf$lower.CI.limit[6],2) ## upper bound
  ub <- round(iccdf$upper.CI.limit[6],2) ## lower bound
  dcorrelation <- as.character(round(clip(as.numeric(correlation)/icc),2)) ## disattenuated correlation
  if (i == 1) {
    desctable <- data.frame("Dataset" = dataset_name, "New Task" =new_task, "New Drawings" = new_items, "New Raters"= new_raters,"Drawings" = num_items, "Raters" = num_raters, "R" = correlation, "r_ink" = ink, "ICC" =paste0(icc, "\n[", lb, ", ", ub, "]"))
  }else {
    tmp<- data.frame("Dataset" = dataset_name, "New Task" =new_task, "New Drawings" = new_items, "New Raters"= new_raters,"Drawings" = num_items, "Raters" = num_raters, "R" = correlation, "r_ink" = ink,"ICC" =paste0(icc, "\n[", lb, ", ", ub, "]"))
    desctable <- rbind(desctable, tmp)
  }
}
## create APA style table and save
outtable <- flextable(desctable)
outtable <- set_header_labels(outtable,
                  values = list(
                    Dataset = "Dataset",
                    New.Task = "New task?",
                    New.Drawings = "New drawings?",
                    New.Raters = "New raters?",
                    Drawings = "Drawings",
                    Raters = "Raters",
                    R = "r",
                    ICC = "ICC")
                  )
outtable <- bold(outtable, bold = TRUE, part = "header")
outtable <- fontsize(outtable, size = 12, part = "all")
outtable <- autofit(outtable, add_w = 0.05, add_h = 0.05)
outtable <- align(outtable, align = "center", part = "all")
save_as_image(outtable, 'descriptives_table_revised.png', webshot = "webshot2")

## Compute avg correlation
fisher_zcors <- FisherZ(allcors)
mean_zcor <- mean(fisher_zcors)
(mean_cor <- FisherZInv(mean_zcor))

#########
#######
###
###  SUPPLEMENTARY MATERIALS: JRT INFORMATION PLOTS
###
#######
#########
val <- read.csv('primary_val_rating_raw.csv')
test <- read.csv('primary_test_rating_raw.csv')
rg1 <- read.csv('ratergen1_raw.csv')
rg2 <- read.csv('ratergen2_raw.csv')
rg2$rater3 <- as.numeric(rg2$rater3)
fg <- read.csv('fargen_raw.csv')

## Compute JRT values
valjrt <- jrt(val[,6:ncol(val)-2], irt.model = "GRM", plots = FALSE)
testjrt <- jrt(test[6:ncol(test)-2], irt.model = "GRM", plots = FALSE)
rg1jrt <- jrt(rg1[,3:ncol(rg1)], irt.model = "GRM", plots = FALSE)
rg2jrt <- jrt(rg2[,6:ncol(rg2)], irt.model = "GRM", plots = FALSE)
fgjrt <- jrt(fg[,3:ncol(fg)], irt.model = "GRM", plots = FALSE)

## Plot Reliability
valplot <- info.plot(valjrt, type = "reliability", title = "Primary: Validation", y.limits = c(0,1), text.size = 12, font.family = "Arial")
valplot
testplot <- info.plot(testjrt, type = "reliability", title = "Primary: Test", y.limits = c(0,1), text.size = 12, font.family = "Arial")
testplot
rg1plot <- info.plot(rg1jrt, type = "reliability", title = "Rater Generalization 1", y.limits = c(0,1), text.size = 12, font.family = "Arial")
rg1plot
rg2plot <- info.plot(rg2jrt, type = "reliability", title = "Rater Generalization 2", y.limits = c(0,1), text.size = 12, font.family = "Arial")
rg2plot
fgplot <- info.plot(fgjrt, type = "reliability", title = "Rater & Task Generalization", y.limits = c(0,1), text.size = 12, font.family = "Arial")
fgplot

allplots <- grid.arrange(valplot, testplot, rg1plot, rg2plot, fgplot, nrow = 3, top = textGrob("JRT Conditional Reliabilities by Dataset", gp = gpar(fontsize=25, fontface = 'bold')))
allplots
ggsave(filename = "reliability_plots_jrt_revised.png", allplots, height = 7, width = 12)


#########
#######
###
###  MAIN ANALYSIS PLOTS
###
#######
#########

##   ******************************
## ***    VALIDATION ANALYSES    ******
##   ******************************

## TIMECOURSE-CORRELATION PLOT
cd <- read.csv('val_corrs_ada_and_ink.csv')
cd$Factor <- ifelse(cd$Factor == "ADA", "AuDrA", cd$Factor)
cd$Factor <- factor(cd$Factor, levels = c("AuDrA","Ink"))
ggplot(cd, aes(x = Step, y = Value, color = Factor, fill = Factor))+
  geom_point()+
  ## stat_smooth(color = "#FDE725FF", fill = "#e6e3be")+
  ## stat_smooth(color = "#2A788EFF", fill = "#ebb905")+
  geom_smooth()+
  scale_color_manual(name = 'Prediction Basis', values = c("#2A788EFF", "#000000"), breaks= c("AuDrA","Ink"), labels = c('AuDrA', 'Ink Baseline'))+
  scale_fill_manual(name = 'Prediction Basis', values = c("#ebb905", "#ebb905"), breaks= c("AuDrA","Ink"), labels = c('AuDrA', 'Ink Baseline'))+
  labs(x = 'training step', y = 'correlation', title = 'correlation: predicted vs. actual creativity ratings\n on validation set') +
  coord_cartesian(xlim = c(0,65900), clip = "off")+
  theme(plot.title = element_blank(),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        legend.title = element_text(size = 30, family = "Arial"),
        legend.text = element_text(size = 25, family = "Arial"),
        legend.position = "bottom",
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
ggsave('AuDrA_val_corr_timecourse_fig.png', height = 9, width = 14)


## QQ PLOT
q <- read.csv('validation_storage_dataframe_revised.csv')
q$ratings <- as.numeric(gsub("\\[|\\]", "", q$ratings))
q$predictions <- as.numeric(gsub("\\[|\\]", "", q$predictions))
ggplot(q, aes(x = ratings, y = predictions))+
  geom_point()+
  stat_smooth(method = "lm", color = "#2A788EFF", fill = "#FDE725FF")+
  labs(x = 'human rating', y = 'model rating', title = 'prediction accuracy by rating value') +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1))+
  annotate("text", x = 0.8, y = 0.93, label = "r = 0.81", fontface = 2, size = 7)+
  geom_abline(slope = 1, intercept = 0)+
  theme(plot.title = element_blank(),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
ggsave('AuDrA_val_QQ_fig_revised.png', height = 9, width = 14)



##   ******************************
## ***    TEST ANALYSES    ******
##   ******************************

## QQ PLOT
q <- read.csv('test_storage_dataframe_revised.csv')
## q$ratings <- as.numeric(gsub("\\[|\\]", "", q$ratings))
## q$predictions <- as.numeric(gsub("\\[|\\]", "", q$predictions))
ggplot(q, aes(x = ratings, y = predictions))+
  geom_point()+
  stat_smooth(method = "lm", color = "#2A788EFF", fill = "#FDE725FF")+
  labs(x = 'human rating', y = 'model rating', title = 'prediction accuracy by rating value') +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1))+
  annotate("text", x = 0.8, y = 0.93, label = "r = 0.80", fontface = 2, size = 7)+
  geom_abline(slope = 1, intercept = 0)+
  theme(plot.title = element_blank(),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
ggsave('AuDrA_test_QQ_fig_revised.png', height = 9, width = 14)

##   ******************************
## ***    RATER GEN TEST ANALYSES    ******
##   ******************************

## QQ PLOT
rg1 <- read.csv('rg1_output_dataframe_revised.csv')
rg1$ratings <- as.numeric(gsub("\\[|\\]", "", rg1$ratings))
rg1$predictions <- as.numeric(gsub("\\[|\\]", "", rg1$predictions))
## write.csv(q,'gen_test_dataframe_fixed.csv',row.names = FALSE)
rg1plot <- ggplot(rg1, aes(x = ratings, y = predictions))+
  geom_point()+
  stat_smooth(method = "lm", color = "#2A788EFF", fill = "#FDE725FF")+
  labs(x = 'human rating', y = 'model rating', title = 'dataset one') +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1))+
  annotate("text", x = 0.6, y = 0.93, label = "r = 0.76", fontface = 2, size = 7)+
  geom_abline(slope = 1, intercept = 0)+
  theme(plot.title = element_text(size = 40, family = "Arial", face = "bold", hjust=0.5),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
rg1plot
ggsave('AuDrA_ratergen1_study1_abs_QQ_fig_revised.png', height = 9, width = 14)

## QQ PLOT
rg2 <- read.csv('rg2_output_dataframe_revised.csv')
rg2$ratings <- as.numeric(gsub("\\[|\\]", "", rg2$ratings))
rg2$predictions <- as.numeric(gsub("\\[|\\]", "", rg2$predictions))
## write.csv(q,'gen_test_dataframe_fixed.csv',row.names = FALSE)
rg2plot <- ggplot(rg2, aes(x = ratings, y = predictions))+
  geom_point()+
  stat_smooth(method = "lm", color = "#2A788EFF", fill = "#FDE725FF")+
  labs(x = 'human rating', y = 'model rating', title = 'dataset two') +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1))+
  annotate("text", x = 0.6, y = 0.93, label = "r = 0.65", fontface = 2, size = 7)+
  geom_abline(slope = 1, intercept = 0)+
  theme(plot.title = element_text(size = 40, family = "Arial", face = "bold", hjust=0.5),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
ggsave('AuDrA_ratergen2_abs_QQ_fig_revised.png', height = 9, width = 14)

genplot <- grid.arrange(rg1plot, rg2plot, nrow = 1, ncol = 2)
ggsave(file = "ratergen_combined_plot_revised.png", genplot, height = 9, width = 14)

##   ******************************
## ***    FAR GEN ANALYSES    ******
##   ******************************

## QQ PLOT
q <- read.csv('fargen_output_dataframe_revised.csv')
q$ratings <- as.numeric(gsub("\\[|\\]", "", q$ratings))
q$predictions <- as.numeric(gsub("\\[|\\]", "", q$predictions))
## write.csv(q,'gen_test_dataframe_fixed.csv',row.names = FALSE)
ggplot(q, aes(x = ratings, y = predictions))+
  geom_point()+
  stat_smooth(method = "lm", color = "#2A788EFF", fill = "#FDE725FF")+
  labs(x = 'human rating', y = 'model rating', title = 'prediction accuracy by rating value') +
  coord_cartesian(xlim = c(0,1), ylim = c(0,1))+
  annotate("text", x = 0.8, y = 0.93, label = "r = 0.49", fontface = 2, size = 7)+
  geom_abline(slope = 1, intercept = 0)+
  theme(plot.title = element_blank(),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
q
ggsave('AuDrA_far_gen_study1_obj_QQ_fig_revised.png', height = 9, width = 14)


##   ********************************************
## ***    TOP 10 HYPERPARAMETER SETTINGS TABLE ******
##   ********************************************
hp <- read.csv('top5_hp_settings.csv')
## Loop over rows of dataframe
for (i in 1:nrow(hp)) {
  cur_mse <- round(hp[i,]$value, 6)
  cur_size <- hp[i,]$params_architecture
  cur_batchsize <- hp[i,]$params_batch_size
  cur_lr <- round(hp[i,]$params_learning_rate,6)
  cur_pretrain <- hp[i,]$params_pretrained

  if (i == 1) {
    top5_hp_table <- data.frame("MSE" = cur_mse, "Model Depth" =cur_size,"Pre-trained?" = cur_pretrain, "Learning Rate" = cur_lr, "Batch Size" = cur_batchsize)
  }else {
    tmp<- data.frame("MSE" = cur_mse, "Model Depth" =cur_size,"Pre-trained?" = cur_pretrain, "Learning Rate" = cur_lr, "Batch Size" = cur_batchsize)
    top5_hp_table <- rbind(top5_hp_table, tmp)
  }
}

## create APA style table and save
hptable <- flextable(top5_hp_table)
hptable <- set_header_labels(hptable,
                  values = list(
                    MSE = "MSE",
                    Model.Depth = "Model Depth",
                    Pre.trained.= "Pre-trained?",
                    Learning.Rate= "Learning Rate",
                    Batch.Size = "Batch Size")
                  )
hptable <- font(hptable, fontname = "Arial")
hptable <- bold(hptable, bold = TRUE, part = "header")
hptable <- fontsize(hptable, size = 12, part = "all")
hptable <- autofit(hptable, add_w = 0.05, add_h = 0.05)
hptable <- align(hptable, align = "center", part = "all")
save_as_image(hptable, 'hyperparams_table.png', webshot = "webshot2")

##   ********************************************
## ***    TRAINING RATING DISTRIBUTION    ******
##   ********************************************
t <- read.csv('training_storage_dataframe_revised.csv')
t$rating <- t$ratings
t <- t[c('predictions','rating')]

ggplot(data=t, aes(x=rating)) +
  geom_histogram(alpha=0.4, position = 'identity') +
  geom_vline(aes(xintercept = mean(rating)),col='red',size=2)+
  geom_vline(aes(xintercept = median(rating)),col='red',size=2)+
  theme(plot.title = element_text(size = 40, hjust = 0.5, face = "bold", family = "Arial"),
        axis.title = element_text(size = 40, family = "Arial"),
        axis.text = element_text(size = 25, family = "Arial"),
        legend.text = element_text(size = 25, family = "Arial"),
        legend.title = element_text(size = 30, family = "Arial"),
        strip.text.x = element_text(size=25, face = "bold", family = "Arial"))
ggsave('training_ratings_distribution_fig.png', height = 9, width = 14)



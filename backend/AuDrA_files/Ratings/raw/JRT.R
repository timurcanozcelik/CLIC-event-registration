"""

All code in this file is licensed to John D. Patterson from The Pennsylvania State University, 11-30-2022, under the Creative Commons Attribution-NonCommerical-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Link to License Deed https://creativecommons.org/licenses/by-nc-sa/4.0/

Link to Legal Code https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Please cite Patterson, J. D., Barbot, B., Lloyd-Cox, J., & Beaty, R. (2022). AuDrA: An Automated Drawing Assessment Platform for Evaluating Creativity.

"""
##  LOAD PACKAGES
libs <- c("plyr","dplyr","mirt","psych","jrt","stringr")
new_libs <- libs[!(libs %in% installed.packages()[,"Package"])]
if(length(new_libs)) install.packages(new_libs)
lapply(libs, library, character.only = T)

##  IMPORT RATING DATASETS
orig <- read.csv('AllBlocksTogether_FIXED.csv')
s1abs <- read.csv('study1_abstract_raw.csv')
s3abs <- read.csv('study3_abstract_ratings.csv')
s1obj <- read.csv('study1_object_raw.csv')

## PRIMARY DATASET JRT
orig_ratings <- orig[,3:ncol(orig)]
fit_orig <- jrt(orig_ratings, irt.model = "GRM", plots = FALSE)
factor_scores <- fit_orig@factor.scores.vector
orig$fscores <- factor_scores
orig$blockid <- ifelse(orig$Block == "Block1", "_4", ifelse(orig$Block == "Block2","_7",ifelse(orig$Block == "Block3","_8", ifelse(orig$Block == "Block4","_9", ifelse(orig$Block == "Block5","_11",ifelse(orig$Block == "Block6","_12",ifelse(orig$Block == "Block7","_13",ifelse(orig$Block == "Block8","_3",ifelse(orig$Block == "Block9","_15", ifelse(orig$Block == "Block10","_17",ifelse(orig$Block == "Block11","_19",ifelse(orig$Block == "Block12","_56", ifelse(orig$Block == "Common","_4","error")))))))))))))
orig$filename <- paste(orig$ImageID,orig$blockid,sep='')
orig <- orig[order(orig$ImageID),]
orig_jrt <- data.frame("filename"=orig$filename, "fscores"=orig$fscores)
write.csv(orig_jrt,'primary_jrt.csv',row.names = FALSE) # save data
## write.table(orig_jrt,'primary_jrt.csv', sep = ",", row.names = FALSE, col.names = FALSE) # use this one to get a csv with no header (which AuDrA needs for train/test)


## RATER GENERALIZATION 1 JRT
s1absratings <- s1abs[,5:7]
fit_s1 <- jrt(s1absratings, irt.model = "GRM", plots = FALSE)
s1abs$fscores <- fit_s1@factor.scores.vector
s1abs$filename <- paste(s1abs$type,'_',s1abs$ImageID, sep = '')
s1absnew <- data.frame("filename" = s1abs$filename, "fscores"=s1abs$fscores)
filtrationset <- read.csv('study2_ratings_fixed.csv')
colnames(filtrationset) <- c('filename','raw')
x <- merge(s1absnew, filtrationset)
s1absnew <- subset(x, select = c(filename, fscores))
write.csv(s1absnew,'rg1_jrt.csv', row.names = FALSE)
## write.table(s1absnew,'rg1_jrt.csv', sep = ",", row.names = FALSE, col.names = FALSE) # use this one to get a csv with no header (AuDrA train/test)


## RATER GENERALIZATION 2 PREPROCESSING & JRT
s3abs <- s3abs[1:778,]
s3abs$Kylee.Rating <- as.integer(s3abs$Kylee.Rating)
s3abs <- subset(s3abs, Exclude == 0)
s3abs$filename <- paste('Drawing ','(',s3abs$Drawing,')', '.jpg', sep = '')
write.csv(s3abs,'s3abs_filtered.csv', row.names = FALSE) # save csv to use python and remove imgs

s3ratings <- s3abs[,7:12]
fit_s3 <- jrt(s3ratings, irt.model = "GRM", plots = FALSE)
s3abs$fscores <- fit_s3@factor.scores.vector
s3absnew <- data.frame("filename"=s3abs$filename,"fscores"=s3abs$fscores)
s3absnew$filename <- unlist(strsplit(s3absnew$filename,".jpg"))
write.csv(s3absnew,'rg2_jrt.csv', row.names = FALSE)
## write.table(s3absnew,'rg2_jrt.csv', sep = ",", row.names = FALSE, col.names = FALSE) # use for header-free version (AuDrA training/testing)

## FAR GENERALIZATION JRT
s1ratings <- s1obj[,5:7]
fit_s1 <- jrt(s1ratings, irt.model = "GRM", plots = FALSE)
s1obj$fscores <- fit_s1@factor.scores.vector
s1obj$filename <- paste('O',s1obj$Type,'_',s1obj$ImageID,sep='')
s1objnew <- data.frame("filename" = s1obj$filename, "fscores"=s1obj$fscores)
write.csv(s1objnew,'fg_jrt.csv', row.names = FALSE)
## write.table(s1objnew,'fg_jrt.csv', sep = ",", row.names = FALSE, col.names = FALSE) # use for header-free (AuDrA train/test)

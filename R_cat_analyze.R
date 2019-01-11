options(stringsAsFactors = F)
setwd('~/catpics/')
require('raster')
require('rgdal')
require('randomForest')
require('rpart')
#require('spatialEco')

positives <- list.files(path = './positive/', full.names = T, pattern = '*diff.png')
negatives <- list.files(path = './negative/', full.names = T, pattern = '*diff.png')

kernel <- matrix(c(1,1,1,1,1,1,1,1,1), byrow = T, nrow = 3)
tri <- function(mat){
  #print(mat)
  cent <- mat[5]
  tri <- sum(abs(mat-cent ))
  return(tri)
}
trisum <- function(rast, kernel = kernel){
  sum(matrix(focal(raster(rast), w = kernel, fun = tri, pad = T, padValue = 0)) > 0, na.rm = T)
}

positives <- data.frame(file = positives, cat = T)
negatives <- data.frame(file = negatives, cat = F)

training <- rbind(positives, negatives)

features <- training
for(f in 1:nrow(training)){
  rast <- as.matrix(raster(training$file[f]) > 1)
  features$sum[f] <- sum(rast)/length(rast)
  features$tri_sum[f] <- trisum(rast, kernel)
  features$door_sum[f] <- sum(rast[nrow(rast),])
  features$house_sum[f] <- sum(rast[1,])
}


boxplot(sum ~ cat, data = features)

hist(features$sum[sums$cat], breaks = seq(0,1,0.05), ylim = c(0,30), main ='')
par(new = T)
hist(features$sum[!sums$cat], breaks = seq(0,1,0.05), ylim = c(0,30), border = 'red')

rf <- randomForest(cat ~ . , data = features[,-1], ntree = 100)
rf
varImpPlot(rf)

boxplot(sum ~ cat, data = features)
boxplot(door_sum ~ cat, data = features)
boxplot(house_sum ~ cat, data = features)
boxplot(tri_sum ~ cat, data = features)

door_sum_q80 <- quantile(features$door_sum[features$cat],0.80)
sum_q80 <- quantile(features$sum[features$cat],0.80)
length(features$door_sum[!features$cat & (features$door_sum > door_sum_q80 | features$sum > sum_q80)])
length(features$door_sum[features$cat & (features$door_sum < q80 & features$sum < sum_q80)])


hist(features$door_sum[features$cat], breaks = 40)

cat_tree <- rpart(cat ~ . , data = features[,-1])
summary(cat_tree)

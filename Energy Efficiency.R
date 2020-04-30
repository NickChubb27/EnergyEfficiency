library(corrplot)
library(ggplot2)
library(grid)
library(gridExtra)
library(neuralnet)

# import data
enerdat <- read.csv('ENB2012_data.csv')

# clean data
enerdat <- enerdat[, c(1:7, 9:10)]

# exploratory analysis
# tableau (histograms, scatterplots with y1 and y2)

# correlations among predictors
# corplot package
corr <- round(cor(enerdat), 4)
corrplot(corr, type='lower', order='hclust', tl.col='black', tl.srt=45)

# combining x2 and x4 into 1 var called x24
enerdat$x24 <- (enerdat$X2 + enerdat$X4)/2
enerdat <- enerdat[, c(-2, -4)]

# run correlations again
corr <- round(cor(enerdat), 4)
corrplot(corr, type='lower', order='hclust', tl.col='black', tl.srt=45)

# split data into training and testing
pct <- 0.70 # train on 70% of data, test on the remaining 30%

N <- nrow(enerdat)
n_train <- round(N*pct)
idx <- sample(1:N, n_train)

traindat <- enerdat[idx, ]
testdat <- enerdat[-idx, ]

# explore data skewness
ggplot(traindat, aes(Y1)) + geom_density(fill="blue")
ggplot(traindat, aes(log(Y1))) + geom_density(fill="blue")
ggplot(traindat, aes(sqrt(Y1))) + geom_density(fill="blue")

ggplot(traindat, aes(Y2)) + geom_density(fill="blue")
ggplot(traindat, aes(log(Y2))) + geom_density(fill="blue")
ggplot(traindat, aes(sqrt(Y2))) + geom_density(fill="blue")

# create predictive models for y1 and y2
y1mod1 <- lm(Y1 ~ X1 + X3 + X5 + X6 + X7 + x24, data = traindat)
summary(y1mod1)
plot(y1mod1)

y2mod1 <- lm(Y2 ~ X1 + X3 + X5 + X6 + X7 + x24, data = traindat)
summary(y2mod1)

# remove X6 bc it has a non significant p value
#update predictive models for y1 and y2
y1mod1 <- lm(Y1 ~ X1 + X3 + X5 + X7 + x24, data = traindat)
summary(y1mod1)
plot(y1mod1)

y2mod1 <- lm(Y2 ~ X1 + X3 + X5 + X7 + x24, data = traindat)
summary(y2mod1)

# look at residual plots of all predictor vars
# make sure residendual plots don't have unwanted patterns
require(gridExtra)
plot1 = ggplot(traindat, aes(X1, residuals(y1mod1))) + geom_point() + geom_smooth()
plot2 = ggplot(traindat, aes(X3, residuals(y1mod1))) + geom_point() + geom_smooth()
plot3 = ggplot(traindat, aes(X5, residuals(y1mod1))) + geom_point() + geom_smooth()
plot4 = ggplot(traindat, aes(X7, residuals(y1mod1))) + geom_point() + geom_smooth()
plot5 = ggplot(traindat, aes(x24, residuals(y1mod1))) + geom_point() + geom_smooth()
grid.arrange(plot1,plot2,plot3,plot4,plot5, ncol=3,nrow=2)

# select 2-3 good models

# evaluate predictive metrics
# MAE, MSE, RMSE, MAPE

pred1 <- predict(y1mod1, newdata = testdat)
pred2 <- predict(y1mod1, newdata = traindat)
summary(pred1)
summary(pred2)
rmse <- sqrt(sum((pred1 - testdat$Y1)^2)/length(testdat$Y1))
c(RMSE = rmse, R2=summary(y1mod1)$r.squared)

rmse <- sqrt(sum((pred2 - testdat$Y2)^2)/length(testdat$Y2))
c(RMSE = rmse, R2=summary(y2mod1)$r.squared)

#calculate MAPE for testing data
mape <- mean(abs((testdat$Y1 - pred1)/testdat$Y1)*100)

#calculate MAPE for testing data
mape <- mean(abs((testdat$Y2 - pred2)/testdat$Y2)*100)

par(mfrow=c(1,1))
plot(testdat$Y1, pred1)
plot(traindat$Y1, pred2)


# optional: create a neural network for Y1 and compare linear regression to NN

# scale the data to [0, 1]
# makes all values z-scores, helps to normalize the data
maxs <- apply(traindat, 2, max) 
mins <- apply(traindat, 2, min)
traindat <- as.data.frame(scale(traindat, center = mins, scale = maxs - mins))

# scale the testing data to [0, 1] using the same scale as the training data
testdat <- as.data.frame(scale(testdat, center = mins, scale = maxs - mins))

# create the formula string
n <- names(traindat)
f <- as.formula(paste("Y1 ~ X1 + X3 + X5 + X7 +x24"))

# train the neural network
nn <- neuralnet(f, data=traindat, hidden=c(8, 16), lifesign = 'full', linear.output=T)

# plot the neural network
plot(nn)

# predict using the testing dataset
pr.nn <- compute(nn, testdat[ , 1:8])

# since results are scaled, descale them for comparison
pr.nn_ <- pr.nn$net.result*(maxs[['Y1']]-mins[['Y1']]) + mins[['Y1']]
test.r <- testdat$Y1*(maxs[['Y1']]-mins[['Y1']]) + mins[['Y1']]

# calculate RMSE
rmse.nn <- sqrt(mean((test.r - pr.nn_)^2))

# compare the two RMSEs
print("Comparing the RMSE of the linear model and the neural network")
print(paste(rmse.lm, rmse.nn))

# plot predictions
par(mfrow=c(1, 2))

plot(testdat$Y1, pred1, col='blue', main='Actual vs. Predicted (LM)', pch=18, cex=0.7)

legend('bottomright', legend='LM', pch=18, col='blue', bty='n')

plot(testdat$Y1, pr.nn_, col='red', main='Actual vs. Predicted (NN)', pch=18, cex=0.7)

legend('bottomright', legend='NN', pch=18, col='red', bty='n')


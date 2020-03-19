###Startup Success and Evaluation 
###Team A7
###Baron Wilton, Kaavash Bahl, Meghana Harish, Shiyu Qian, Shiyue Ma 


### source funtions
source("DataAnalyticsFunctions.R")
source("PerformanceCurves.R")

### data cleaning: remove redundant and nonimformative columns
rawdata <- read.csv("CAX_Startup_Train.csv")
summary(rawdata)
### Finding correlation 
cor(rawdata$Company_1st_investment_time, rawdata$Company_avg_investment_time)
drop <- c('CAX_ID', 'Founders_skills_score', 'Founders_experience', 'Company_avg_investment_time', 
          'Company_raising_fund')
data.clean <- rawdata[,!names(rawdata) %in% drop]

########## k-means: which makes no sense since computing mean value on binary or categorical is not meaningful :)
x.test <- model.matrix(~., data=data.clean[,2:46])[,-1]
x.scaled <- scale(x.test)
kfit <- lapply(1:50, function(k) kmeans(x.scaled,k))

###Selecting the number of clusters
source("kIC.R") 
kaic <- sapply(kfit,kIC)
kbic  <- sapply(kfit,kIC,"B")
kHDic  <- sapply(kfit,kIC,"C")
par(mar=c(1,1,1,1))
par(mai=c(1,1,1,1))
plot(kaic, xlab="k (# of clusters)", ylab="IC (Deviance + Penalty)", 
     ylim=range(c(kaic,kbic,kHDic)), # get them on same page
     type="l", lwd=2)
# Vertical line where AIC is minimized
abline(v=which.min(kaic))
# Next we plot BIC
lines(kbic, col=4, lwd=2)
# Vertical line where BIC is minimized
abline(v=which.min(kbic),col=4)
# Next we plot HDIC
lines(kHDic, col=3, lwd=2)
# Vertical line where HDIC is minimized
abline(v=which.min(kHDic),col=3)
# Insert labels
text(c(which.min(kaic),which.min(kbic),which.min(kHDic)),c(mean(kaic),mean(kbic),mean(kHDic)),c("AIC","BIC","HDIC"))
#Below function returns k=1 which proves that it is not meaningful
k= which.min(kHDic)

########## PCA: having too many variables in dataset makes it too complicated
pca <- prcomp(x.test, scale=TRUE)
summary(pca)
### Lets plot the variance that each component explains
plot(pca, main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors", line=1, font=2)



######## variables selection via modeling
###
### Models to compare:
### model.null : null model
### model.logistic : logistic regression using forward selection
### model.logistic.interaction : logistic regression with interaction using forward selection
### model.tree: classification tree
### model.l: Lasso and min choice of lambda
### model.pl: Post Lasso associated with Lasso and min choice of lambda

installpkg("e1071")
installpkg("rpart")
library(e1071)
library(rpart)
library(tree)

install.packages("rpart.plot")
library(rpart.plot)

installpkg("glmnet")
library(glmnet)

###Null Model predictions
sum(data.clean$Dependent==1)/nrow(data.clean)

###Logistic regression using forward selection
model.null <- glm(Dependent~1, data=data.clean, family='binomial')
model.full1 <- glm(Dependent~., data=data.clean, family='binomial')
model.full2 <- glm(Dependent~.^2, data=data.clean, family='binomial')
system.time(model.logistic <- step(model.null, scope=formula(model.full1), family="binominal", dir="forward"))
summary(model.logistic)
system.time(model.logistic.interaction <- step(model.null, scope=formula(model.full2), family="binominal", dir="forward"))
summary(model.logistic.interaction)
churntree=rpart(Dependent~., data=data.clean, method="class")
rpart.plot(churntree)
summary(churntree)

###LASSO
Mx <- model.matrix(Dependent ~ ., data=data.clean)[,-1]
My <- data.clean$Dependent == 1
lasso <- glmnet(Mx,My, family="binomial")

###Using lambda = min rule
lassoCV <- cv.glmnet(Mx,My, family="binomial")
lassoCV$lambda[which.min(lassoCV$cvm)]
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(lassoCV, main="Fitting Graph for CV Lasso \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
colnames(Mx[,features.min])

###Using lambda = 1se rule
features.1se <- support(lasso$beta[,which.min( (lassoCV$lambda-lassoCV$lambda.1se)^2)])
length(features.1se) 

###Ussing lambda theory
num.features <- ncol(Mx)
num.n <- nrow(Mx)
num.success <- sum(My)
w <- (num.success/num.n)*(1-(num.success/num.n))
lambda.theory <- sqrt(w*log(num.features/0.05)/num.n)
lassoTheory <- glmnet(Mx,My, family="binomial",lambda = lambda.theory)
features.theory <- support(lassoTheory$beta)
length(features.theory)

data.min <- data.frame(Mx[,features.min],My)
#data.1se <- data.frame(Mx[,features.1se],My)
#data.theory <- data.frame(Mx[,features.theory],My)

###
### prediction is a probability score
### we convert to 1 or 0 via prediction > threshold
PerformanceMeasure <- function(actual, pred, threshold=.5) {
  #1-mean( abs( (prediction>threshold) - actual ) )  
  #R2(y=actual, pred=prediction, family="binomial")
  1-mean( abs( (pred- actual) ) )  
}
########## 5-Fold Cross Validation using 1-mean( abs( (pred- actual) ) )  as measure metric
set.seed(7)
### create a vector of fold memberships (random order)
n <- nrow(data.clean)
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

OOS.R2 <- data.frame(PL.min=rep(NA,nfold), L.min=rep(NA,nfold),
                     logistic=rep(NA,nfold), logistic.interaction=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 
### null model prediction
sum(data.clean$Dependent[train]==1)/length(train)

for(k in 1:nfold){ 
  train <- which(foldid!=k)
  
  ## fit the two regressions and null model
  model.null <- glm(Dependent~1, data=data.clean, subset=train, family='binomial')
  model.full1 <- glm(Dependent~., data=data.clean, subset=train, family='binomial')
  model.full2 <- glm(Dependent~.^2, data=data.clean, subset=train, family='binomial')
  system.time(model.logistic <- step(model.null, scope=formula(model.full1), family="binominal", dir="forward"))
  system.time(model.logistic.interaction <- step(model.null, scope=formula(model.full2), family="binominal", dir="forward"))
  model.tree <- tree(Dependent~., data=data.clean, subset=train)
  
  ## get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=data.clean[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=data.clean[-train,], type="response")
  pred.tree                 <- predict(model.tree, newdata=data.clean[-train,], type="vector")
  pred.null                 <- predict(model.null, newdata=data.clean[-train,], type="response")
  
  ### calculate 
  OOS.R2$logistic.interaction[k] <- PerformanceMeasure(actual=data.clean$Dependent[-train], pred=pred.logistic.interaction)
  # Logistic
  OOS.R2$logistic[k] <-  PerformanceMeasure(actual=data.clean$Dependent[-train], pred=pred.logistic)
  # Tree
  OOS.R2$tree[k] <-  PerformanceMeasure(actual=data.clean$Dependent[-train], pred=pred.tree)
  #Null
  OOS.R2$null[k] <-  PerformanceMeasure(actual=data.clean$Dependent[-train], pred=pred.null)

  ### This is the CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  OOS.R2$PL.min[k] <- PerformanceMeasure(actual=My[-train], pred=predmin)
  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  OOS.R2$L.min[k] <- PerformanceMeasure(actual=My[-train], pred=predlassomin)

  print(paste("Iteration",k,"of",nfold,"completed"))
}
  
colMeans(OOS.R2)
names(OOS.R2)[4] <-"logistic\ninteraction"
boxplot(OOS.R2, ylim=c(0.3,0.8))

########## 5-Fold Cross Validation using accuracy
set.seed(7)
### create a vector of fold memberships (random order)
n <- nrow(data.clean)
nfold <- 5
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

OOS <- data.frame(logistic=rep(NA,nfold), logistic.interaction=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 
PL.OOS <- data.frame(PL.min=rep(NA,nfold))
L.OOS <- data.frame(L.min=rep(NA,nfold))

OOS.TPR <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 
OOS.FPR <- data.frame(logistic=rep(NA,nfold), tree=rep(NA,nfold), null=rep(NA,nfold)) 

PL.OOS.TPR <- data.frame(PL.min=rep(NA,nfold)) 
L.OOS.TPR <- data.frame(L.min=rep(NA,nfold)) 
PL.OOS.FPR <- data.frame(PL.min=rep(NA,nfold)) 
L.OOS.FPR <- data.frame(L.min=rep(NA,nfold)) 

val <- .3 

  
for(k in 1:nfold){ 
  train <- which(foldid!=k)
  
  ## fit the two regressions and null model
  model.null <- glm(Dependent~1, data=data.clean, subset=train, family='binomial')
  model.full1 <- glm(Dependent~., data=data.clean, subset=train, family='binomial')
  model.full2 <- glm(Dependent~.^2, data=data.clean, subset=train, family='binomial')
  system.time(model.logistic <- step(model.null, scope=formula(model.full1), family="binominal", dir="forward"))
  system.time(model.logistic.interaction <- step(model.null, scope=formula(model.full2), family="binominal", dir="forward"))
  model.tree <- tree(Dependent~., data=data.clean, subset=train)
  
  ## get predictions: type=response so we have probabilities
  pred.logistic.interaction <- predict(model.logistic.interaction, newdata=data.clean[-train,], type="response")
  pred.logistic             <- predict(model.logistic, newdata=data.clean[-train,], type="response")
  pred.tree                 <- predict(model.tree, newdata=data.clean[-train,], type="vector")
  pred.null                 <- predict(model.null, newdata=data.clean[-train,], type="response")
  
  ## calculate TPR,FPR,ACC
  # Logistic Interaction
  values <- FPR_TPR( (pred.logistic.interaction >= val) , My[-train] )
  
  OOS$logistic.interaction[k] <- values$ACC
  OOS.TPR$logistic.interaction[k] <- values$TPR
  OOS.FPR$logistic.interaction[k] <- values$FPR
  # Logistic
  values <- FPR_TPR( (pred.logistic >= val) , My[-train] )
  OOS$logistic[k] <- values$ACC
  OOS.TPR$logistic[k] <- values$TPR
  OOS.FPR$logistic[k] <- values$FPR
  # Tree
  values <- FPR_TPR( (pred.tree >= val) , My[-train] )
  OOS$tree[k] <- values$ACC
  OOS.TPR$tree[k] <- values$TPR
  OOS.FPR$tree[k] <- values$FPR
  #Null
  values <- FPR_TPR( (pred.null >= val) , My[-train] )
  OOS$null[k] <- values$ACC
  OOS.TPR$null[k] <- values$TPR
  OOS.FPR$null[k] <- values$FPR

  ### This is the CV for the Post Lasso Estimates
  rmin <- glm(My~., data=data.min, subset=train, family="binomial")
  predmin <- predict(rmin, newdata=data.min[-train,], type="response")
  
  values <- FPR_TPR( (predmin >= val) , My[-train] )
  PL.OOS$PL.min[k] <- values$ACC
  PL.OOS.TPR$PL.min[k] <- values$TPR
  PL.OOS.FPR$PL.min[k] <- values$FPR

  ### This is the CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train], family="binomial",lambda = lassoCV$lambda.min)
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  
  values <- FPR_TPR( (predlassomin >= val) , My[-train] )
  L.OOS$L.min[k] <- values$ACC
  L.OOS.TPR$L.min[k] <- values$TPR
  L.OOS.FPR$L.min[k] <- values$FPR

  print(paste("Iteration",k,"of",nfold,"completed"))
}

###Accuracy plot
par(mar=c(1,1,1,1))
par(mai=c(1,1,1,1))
names(OOS)[1] <-"logistic"
names(OOS)[2] <-"logistic\ninteraction"
ACCperformance <- cbind(PL.OOS,L.OOS,OOS)
boxplot(ACCperformance, ylim=c(0.3,0.8), las=2)
boxplot(OOS.R2, ylim=c(0.3,0.8), las=2)

###TPR & FPR graph with 0.3 threshold
plot( c( 0, 1 ), c(0, 1), type="n", xlim=c(0,1), ylim=c(0,1), bty="n", xlab = "False positive rate", ylab="True positive rate")
lines(c(0,1),c(0,1), lty=2)
text( colMeans(PL.OOS.FPR), colMeans(PL.OOS.TPR), labels=c("PL.min"))
points(colMeans(PL.OOS.FPR), colMeans(PL.OOS.TPR))
text( colMeans(L.OOS.FPR), colMeans(L.OOS.TPR), labels=c("L.min"))
points(colMeans(L.OOS.FPR), colMeans(L.OOS.TPR))
text( mean(OOS.FPR$logistic), mean(OOS.TPR$logistic), labels=c("logistic"))
points(mean(OOS.FPR$logistic), mean(OOS.TPR$logistic))
text( mean(OOS.FPR$logistic.interaction), mean(OOS.TPR$logistic.interaction), labels=c("logistic\ninteraction"))
points(mean(OOS.FPR$logistic.interaction), mean(OOS.TPR$logistic.interaction))
text( mean(OOS.FPR$tree), mean(OOS.TPR$tree), labels=c("tree"))
points(mean(OOS.FPR$tree), mean(OOS.TPR$tree))

###Results from Post Lasso
summary(rmin)
predmin.full <- predict(rmin, newdata=data.min, type="response")
par(mar=c(5,5,3,5))
roccurve <-  roc(p=predmin.full, y=My, bty="n", xlim=c(0,1), ylim=c(0,1))

hist(predmin.full, main="Predictions for Post Lasso", col="grey", xlab="")


############### clustering:PCA
x <- model.matrix(~., data=data.min[,1:8])[,-1]

pca <- prcomp(x, scale=TRUE)
summary(pca)
### Lets plot the variance that each component explains
plot(pca, main="PCA: Variance Explained by Factors")
mtext(side=1, "Factors",  line=1, font=2)
pc <- predict(pca) 
pc <- pc[,1:6]
pc.max <- as.data.frame(colnames(pc)[apply(pc,1,which.max)])
data.clean$pc2 <- pc.max
tapply(data.clean[,1], data.clean$pc2, mean)
tapply(data.clean[,16], data.clean$pc2, count) #model
tapply(data.clean[,9], data.clean$pc2, mean) #senior team count
tapply(data.clean[,18], data.clean$pc2, count) #industry exposure
tapply(data.clean[,19], data.clean$pc2, count) #education
tapply(data.clean[,41], data.clean$pc2, mean) #analytics
tapply(data.clean[,39], data.clean$pc2, count) #crowdfounding
tapply(data.clean[,24], data.clean$pc2, count) #publication
tapply(data.clean[,36], data.clean$pc2, mean) #competitor
tapply(data.clean[,17], data.clean$pc, count) #global exposure


names(data.clean)
library(plyr)
names(data.clean)
count(data.clean[,48])

######### Double selection here did not yield meaningful results thus we decided not to include them in our model
resSimple1 <- glm(Dependent ~ Company_business_model, data = data.clean )
summary(resSimple1)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Company_business_model, data = data.clean )
d <- data.clean$Company_business_model
source("causal_source.R")
cl <- CausalLogistic(y,d,x)

resSimple2 <- glm(Dependent ~ Company_senior_team_count, data = data.clean )
summary(resSimple2)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Company_senior_team_count, data = data.clean )
d <- data.clean$Company_senior_team_count
c2 <- CausalLinear(y,d,x)

resSimple3 <- glm(Dependent ~ Company_analytics_score, data = data.clean )
summary(resSimple3)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Company_analytics_score, data = data.clean )
d <- data.clean$Company_analytics_score
c3 <- CausalLinear(y,d,x)

resSimple4 <- glm(Dependent ~ Company_competitor_count, data = data.clean )
summary(resSimple4)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Company_competitor_count, data = data.clean )
d <- data.clean$Company_competitor_count
c4 <- CausalLinear(y,d,x)

resSimple5 <- glm(Dependent ~ Founders_Industry_exposure, data = data.clean )
summary(resSimple5)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Founders_Industry_exposure, data = data.clean )
d <- data.clean$Founders_Industry_exposure
c5 <- CausalLogistic(y,d,x)

resSimple6 <- glm(Dependent ~ Founder_education, data = data.clean )
summary(resSimple6)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Founder_education, data = data.clean )
d <- data.clean$Founder_education
c6 <- CausalLogistic(y,d,x)

resSimple7 <- glm(Dependent ~ Founders_publications, data = data.clean )
summary(resSimple7)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Founders_publications, data = data.clean )
d <- data.clean$Founders_publications
c7 <- CausalLogistic(y,d,x)

resSimple8 <- glm(Dependent ~ Company_crowdfunding, data = data.clean )
summary(resSimple8)
y <- data.clean$Dependent
x <- model.matrix(Dependent~.-Company_crowdfunding, data = data.clean )
d <- data.clean$Company_crowdfunding
c8 <- CausalLogistic(y,d,x)

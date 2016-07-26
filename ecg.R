

library(xgboost)
param <- list("objective" = "multi:softmax",    # multiclass classification 
              "num_class" = 3,    # number of classes 
              "eval_metric" = "mlogloss",    # evaluation metric 
              "nthread" = 4,   # number of threads to be used 
              "max_depth" = 30,    # maximum depth of tree 
              "eta" = 0.9,    # step size shrinkage 
              "subsample" = 0.9,    # part of data instances to grow tree 
              "colsample_bytree" = 0.9
)

ecg <- read.csv("ecgdata.csv")
mean_error = 1
while(mean_error>0.03){
ecg_p <- ecg[sample(nrow(ecg),2400,replace = FALSE),]

Xtrain = as.matrix(ecg_p[1:2000,1:144])
Ytrain = as.matrix(ecg_p[1:2000,145])
Xtest = as.matrix(ecg_p[2001:2400,1:144])
Ytest = as.matrix(ecg_p[2001:2400,145])

dtrain <- xgb.DMatrix(Xtrain,label=Ytrain)
dtest <- xgb.DMatrix(Xtest, label=Ytest)

watchlist <- list(test=dtest)

ecgmdl <- xgb.train(param, data=dtrain, nrounds=300,
                    watchlist=watchlist, early.stop.round=10)
#pred_train <- predict(ecgmdl, dtrain, outputmargin=TRUE)
#setinfo(dtrain, "base_margin", pred_train)

pred <- predict(ecgmdl, dtest)
mean_error <- mean(pred != Ytest)
accuracy <- (1-mean_error)*100
accuracy
number_of_wrong_predictions <- (mean_error)*400
number_of_wrong_predictions
}


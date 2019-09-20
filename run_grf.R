library(grf)



get_forest <- function(X_CT, Y_CT, T_CT){
  tau.forest = causal_forest(X_CT, Y_CT, T_CT)
  return(tau.forest)
}
train_subset_cross <-function(x,y,t,ind) {
  hflash <- get_forest(x[ind,], y[ind], t[ind])
  tau.hat = predict(hflash, x[-ind,], estimate.variance = FALSE)
  return(tau.hat)
}

sample_and_train <- function(subset, X_OS, T_OS, Y_OS_hf, n = 5){
  for (ind in seq(n)) {
    sampleOS <- sample.int(n = nrow(subset), size = floor(.5*nrow(subset)), replace = F)
    antisampleOS <- seq(1,length(Y_OS_hf))[-sampleOS]
    hf_fold1 <- train_subset_cross(X_OS, Y_OS_hf, T_OS, sampleOS)
    hf_fold2 <- train_subset_cross(X_OS, Y_OS_hf, T_OS, antisampleOS)
    print(length(hf_fold1$predictions))
    write.csv(hf_fold1$predictions, paste('pred-anti-sample', toString(ind), '.csv',sep=""), row.names = FALSE )
    write.csv(hf_fold2$predictions, paste('pred-sample', toString(ind), '.csv',sep=""), row.names = FALSE )
    write.csv(sampleOS, paste('sample-', toString(ind), '.csv',sep=""), row.names = FALSE )
    write.csv(antisampleOS, paste('antisample-', toString(ind), '.csv',sep=""), row.names = FALSE )
  }
}

sample_and_train(df, X_, T_, Y, 50)

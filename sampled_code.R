# Directory path ----
setwd("D:\\Bárbara Mena Velásqu\\Mis documentos\\Educación\\Maestrias\\Escocia\\06_Courses\\13_Project\\ceg_github")
load("sampled_code.RData")

# Packages ----
library(ceg)
library(caret)
library(reshape2)
library(Hmisc)
library(bnclassify)

# User-Defined functions ---- THIS IS WHAT YOU ARE LOOKING FOR!
ml.f <- function(ceg){
  # insumos
  ct <- ceg@staged.tree@contingency.table
  clus <- lapply(1:ceg@staged.tree@event.tree@num.variable, function(i) ceg@staged.tree@stage.structure[[i]]@cluster)
  ct.p <- list() 
  
  prop1.ct <- lapply(1:length(ct), function(i) prop.table(ct[[i]], margin = 1))
  
  # funcion a aplicar a staged cluster
  helper.f <- function(y){
    d <- ct[[y]] # copia de tabla a procesar
    nulo <- which(sapply(clus[[y]], function(c) any(is.na(c)))) # encuentra filas nulas
    index <- sapply(which(sapply(clus[[y]], function(c) any(is.na(c)))), function(i) which(sapply(sapply(clus[[y]], function(t) match(x = i, table = t, nomatch = 0)), any))) # encuentra nulos en otros clusters 
    ref <- cbind(index, nulo) # indice para for loop
    for(i in 1:nrow(ref)){
      d[ref[i, 1], ] <- d[ref[i, 1], ] + d[ref[i, 2], ]
    } # combina filas
    for(i in 1:nrow(ref)){
      d[ref[i, 2], ] <- d[ref[i, 1], ]
    } # sustituye valores en filas
    ct.p[[y]] <<- prop.table(d, margin = 1) # calcula probabilidades marginales de la tabla de contigencia
  }
  
  # transforma tablas de contigencias
  ifelse(!sapply(sapply(clus, is.na), any), # evaluacion
         ct.p <- lapply(which(!sapply(sapply(clus, is.na), any)), function(i) prop.table(ct[[i]], margin = 1)),
         lapply(which(sapply(sapply(clus, is.na), any)), helper.f)
  )
  
  melt <- lapply(ct.p, melt) 
  melt <- lapply(1:length(melt), function(i) melt[[i]][order(melt[[i]][,"Var1"]),])
  for(i in 1:length(melt)){
    melt[[i]][,"Var2"] <- 1:nrow(melt[[i]])
  }
  
  # melt tablas de contigencia para prediccion
  table <- merge(melt[[1]], melt[[2]], by.x = "Var2", by.y = "Var1")[, c(-2:-1)]
  for(i in 3:length(melt)){
    table <- merge(table, melt[[i]], by.x = "Var2", by.y = "Var1")[, -1]
  }
  table <- table[,-length(melt)]
  colnames(table) <- NULL
  colnames(table) <- colnames(table, do.NULL = FALSE, prefix = "var")
  table$prob <- apply(table, 1, prod) # tabla de probabilidades reorganizada
  
  # clasificacion
  mx <- matrix(table[,"prob"], nrow = nrow(table)/ceg@staged.tree@event.tree@num.category[[1]], 
               ncol = ceg@staged.tree@event.tree@num.category[[1]],
               byrow = FALSE)
  pred<- rep(apply(mx, 1, which.max), ceg@staged.tree@event.tree@num.category[[1]])
  
  # resultados
  output <- list(prop1.ct = prop1.ct, prop2.ct = ct.p, meltprop.ct = table, pred = pred)
  return(output)
}
predict.f <- function(ceg, mlout, data) {
  table <- do.call(expand.grid, ceg@staged.tree@event.tree@label.category)
  table <- table[with(table, order(Var1, Var2, Var3, Var4, Var5)), ] # Manually adjusted (pending to automate)
  table <- cbind(table, class.pred = mlout$pred)
  
  x <- apply(data, 2, as.numeric)
  y <- apply(table[, -ncol(table)], 2, as.numeric)
  matchIndex <- find.matches(x, y, maxmatch = 1, tol = rep(0, (ncol(table)-1)))
  output <- table[matchIndex$matches, ncol(table)]
  
  return(output)
  rm(table, x, y, matchIndex)
}

# Common settings ----
iss <- 2
score <- "loglik"
ite <- 100
trainset.prop <- 0.75

# Cross validation  ----

# ceg ----
ceg.output <- as.data.frame(matrix(data = 0, nrow = ite, ncol = 9))
colnames(ceg.output) <- c("set", "sst", "ceg", "MLoutput", "ConfusionMx", "Accuracy", "BalancedAccuracy", "F1Score", "ExecTime")

for(i in 1:ite){
  trainset <- data[sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  testset <- data[-sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  
  set <- Stratified.event.tree(trainset)
  sst <- Stratified.staged.tree(trainset, y =iss)
  ceg <- Stratified.ceg.model(sst)
  mlout <- ml.f(ceg)
  cmx <- confusionMatrix(predict.f(ceg, mlout, testset), unclass(testset[,"class"])) 
  
  ceg.output$set[i] <- list(serialize(set, connection = NULL))
  ceg.output$sst[i] <- list(serialize(sst, connection = NULL))
  ceg.output$ceg[i] <- list(serialize(ceg, connection = NULL))
  ceg.output$MLoutput[[i]] <- list(mlout)
  ceg.output$ConfusionMx[[i]] <- list(cmx)
  ceg.output[i,6] <- cmx$overall["Accuracy"]
  ceg.output[i,7] <- cmx$byClass["Balanced Accuracy"]
  ceg.output[i,8] <- cmx$byClass["F1"]
  ceg.output[i,9] <- system.time(Stratified.ceg.model(Stratified.staged.tree(trainset, y =iss)))[3] #elapsed
  
  rm(trainset, testset, set, sst, ceg, mlout, cmx)
  print(i)
}

# BNCs ----

# NB ----
nb.output <- as.data.frame(matrix(data = 0, nrow = ite, ncol = 6))
colnames(nb.output) <- c("learning", "ConfusionMx", "Accuracy", "BalancedAccuracy", "F1Score", "ExecTime")

for(i in 1:ite){
  trainset <- data[sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  testset <- data[-sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  
  nb <- bnc("nb", "class", trainset, smooth = iss)
  cmx <- confusionMatrix(predict(nb, testset), testset[,"class"])
  
  nb.output$learning[i] <- list(serialize(nb, connection = NULL))
  nb.output$ConfusionMx[i] <- list(cmx)
  nb.output[i,3] <- cmx$overall["Accuracy"]
  nb.output[i,4] <- cmx$byClass["Balanced Accuracy"]
  nb.output[i,5] <- cmx$byClass["F1"]
  nb.output[i,6] <- system.time(bnc("nb", "class", trainset, smooth = iss))[3] #elapsed
  
  rm(trainset, testset, nb, cmx)
}

# TAN ----
tan.output <- as.data.frame(matrix(data = 0, nrow = ite, ncol = 6))
colnames(tan.output) <- c("learning", "ConfusionMx", "Accuracy", "BalancedAccuracy", "F1Score", "ExecTime")

for(i in 1:ite){
  trainset <- data[sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  testset <- data[-sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  
  tan <- bnc("tan_cl", "class", trainset, smooth = iss, dag_args = list(score = score))
  cmx <- confusionMatrix(predict(tan, testset), testset[,"class"])
  
  tan.output$learning[i] <- list(serialize(tan, connection = NULL))
  tan.output$ConfusionMx[i] <- list(cmx)
  tan.output[i,3] <- cmx$overall["Accuracy"]
  tan.output[i,4] <- cmx$byClass["Balanced Accuracy"]
  tan.output[i,5] <- cmx$byClass["F1"]
  tan.output[i,6] <- system.time(bnc("tan_cl", "class", trainset, smooth = iss, dag_args = list(score = score)))[3] #elapsed
  
  rm(trainset, testset, tan, cmx)
}

# HCSPTAN ----
hcsptan.output <- as.data.frame(matrix(data = 0, nrow = ite, ncol = 6))
colnames(hcsptan.output) <- c("learning", "ConfusionMx", "Accuracy", "BalancedAccuracy", "F1Score", "ExecTime")

for(i in 1:ite){
  trainset <- data[sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  testset <- data[-sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  
  hcsptan <- bnc("tan_hcsp", "class", trainset, smooth = iss, dag_args = list(k = 2, epsilon = 0.01, smooth = iss))
  cmx <- confusionMatrix(predict(hcsptan, testset), testset[,"class"])
  
  hcsptan.output$learning[i] <- list(serialize(hcsptan, connection = NULL))
  hcsptan.output$ConfusionMx[i] <- list(cmx)
  hcsptan.output[i,3] <- cmx$overall["Accuracy"]
  hcsptan.output[i,4] <- cmx$byClass["Balanced Accuracy"]
  hcsptan.output[i,5] <- cmx$byClass["F1"]
  hcsptan.output[i,6] <- system.time(bnc("tan_hcsp", "class", trainset, smooth = iss, dag_args = list(k = 2, epsilon = 0.01, smooth = iss)))[3] #elapsed
  
  rm(trainset, testset, hcsptan, cmx)
}

summary(hcsptan.output[,3:6])

# SNB ----
snb.output <- as.data.frame(matrix(data = 0, nrow = ite, ncol = 6))
colnames(snb.output) <- c("learning", "ConfusionMx", "Accuracy", "BalancedAccuracy", "F1Score", "ExecTime")

for(i in 1:ite){
  trainset <- data[sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  testset <- data[-sample(1:nrow(data), ceiling(trainset.prop*nrow(data)), replace = FALSE), ]
  
  snb <- bnc("fssj", "class", trainset, smooth = iss, dag_args = list(k = 2, epsilon = 0.01, smooth = iss))
  cmx <- confusionMatrix(predict(snb, testset), testset[,"class"])
  
  snb.output$learning[i] <- list(serialize(snb, connection = NULL))
  snb.output$ConfusionMx[i] <- list(cmx)
  snb.output[i,3] <- cmx$overall["Accuracy"]
  snb.output[i,4] <- cmx$byClass["Balanced Accuracy"]
  snb.output[i,5] <- cmx$byClass["F1"]
  snb.output[i,6] <- system.time(bnc("tan_hcsp", "class", trainset, smooth = iss, dag_args = list(k = 2, epsilon = 0.01, smooth = iss)))[3] #elapsed
  
  rm(trainset, testset, snb, cmx)
}
summary(snb.output[,3:6])

# Metrics consolidation ----
union.accu <- as.data.frame(list(indicador = rep("accuracy", ite), 
                                 classifier = c(rep("nb", ite), rep("tan", ite), rep("hcsptan", ite), rep("snb", ite), rep("ceg", ite)),
                                 accuracy = c(nb.output$Accuracy, tan.output$Accuracy, hcsptan.output$Accuracy, snb.output$Accuracy, ceg.output$Accuracy)
)
)

union.balanaccu <- as.data.frame(list(indicador = rep("balanceAccuracy", ite), 
                                      classifier = c(rep("nb", ite), rep("tan", ite), rep("hcsptan", ite), rep("snb", ite), rep("ceg", ite)),
                                      balanceAccuracy = c(nb.output$BalancedAccuracy, tan.output$BalancedAccuracy, hcsptan.output$BalancedAccuracy, snb.output$BalancedAccuracy, ceg.output$BalancedAccuracy)
)
)

union.F1Score <- as.data.frame(list(indicador = rep("F1Score", ite), 
                                    classifier = c(rep("nb", ite), rep("tan", ite), rep("hcsptan", ite), rep("snb", ite), rep("ceg", ite)),
                                    F1Score = c(nb.output$F1Score, tan.output$F1Score, hcsptan.output$F1Score, snb.output$F1Score, ceg.output$F1Score)
)
)

union.time <- as.data.frame(list(indicador = rep("time", ite), 
                                 classifier = c(rep("nb", ite), rep("tan", ite), rep("hcsptan", ite), rep("snb", ite), rep("ceg", ite)),
                                 time = c(nb.output$ExecTime, tan.output$ExecTime, hcsptan.output$ExecTime, snb.output$ExecTime, ceg.output$ExecTime)
)
)

cols <- rainbow(5, s = 0.50)
boxplot(union.accu$accuracy ~ union.accu$classifier, main = "Accuracy metric", notch = TRUE, col = cols)
boxplot(union.balanaccu$balanceAccuracy ~ union.balanaccu$classifier, main = "Balanced Accuracy metric", notch = TRUE, col = cols)
boxplot(union.F1Score$F1Score ~ union.F1Score$classifier, main = "FScore metric", notch = TRUE, col = cols)
boxplot(union.time$time ~ union.time$classifier, main = "Training Time", notch = TRUE, col = cols)

by(union.accu[,3], union.accu[,2], summary)
by(union.balanaccu[,3], union.balanaccu[,2], summary)
by(union.F1Score[,3], union.F1Score[,2], summary)
by(union.time[,3], union.time[,2], summary)

# a simulation review
which.max(ceg.output$Accuracy)
ceg.output$Accuracy[[58]]

sst <- unserialize(ceg.output$sst[[58]])
plot(sst)
ceg <- unserialize(ceg.output$ceg[[58]])
plot(ceg)

# Saving ----
save.image("sampled_code.RData")

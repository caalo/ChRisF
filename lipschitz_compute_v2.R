#usage:
#Rscript lipschitz_compute.R [folder name] [number of epochs] [number of resampling]

setwd('/xchip/cle/analysis/clo/ChRisF-master-NIPS')
args<-commandArgs(TRUE)
folder = as.character(args[[1]])
nSentences = 7936;
 hashSize = 2000000;


#sparse vector subtraction: vec.j - vec.i
subtract.sparse.fn = function(vec.j, vec.i, sparse_default) {
  #make it not sparse, then subtract, not very optimal, but works.
  long.vec.j = rep(x = sparse_default, hashSize)
  long.vec.i = rep(x = sparse_default, hashSize)
  
  long.vec.j[vec.j[, 1]] = vec.j[, 2]
  long.vec.i[vec.i[, 1]] = vec.i[, 2]
  
  return(long.vec.j - long.vec.i)

}

sparseToLong.fn = function(vec, sparse_default) {
  long.vec = rep(sparse_default, hashSize)
  long.vec[vec[, 1]] = vec[, 2]
  return(long.vec)
}


#generate nPairs pairs for sampling from weights and gradient files. 
nEpochs = as.numeric(args[[2]]) - 1 # number of epochs.
nPairs = as.numeric(args[[3]])
sample_pairs = cbind(round(runif(n = nPairs, min = 1, max = nEpochs)), 
                     round(runif(n = nPairs, min = 1, max = nEpochs)))

lipschitz = mapply(sample_pairs[, 1], sample_pairs[, 2], FUN = function(x, y) {
  cat(x, y, "\n")

  weight.i = sparseToLong.fn(read.delim(paste(folder, "/", x, ".weight", sep = "") , sep = " "), 1)

  weight.j = sparseToLong.fn(read.delim(paste(folder, "/", y, ".weight", sep = "") , sep = " "), 1)

  weight_subtract_normed = norm(weight.i - weight.j, type = "2")

  grad.i = sparseToLong.fn(read.delim(paste(folder, "/", x, ".weight", sep = "") , sep = " "), 1) - 
           sparseToLong.fn(read.delim(paste(folder, "/", (x - 1), ".weight", sep = "") , sep = " "), 1) 

  grad.j = sparseToLong.fn(read.delim(paste(folder, "/", y, ".weight", sep = "") , sep = " "), 1) - 
           sparseToLong.fn(read.delim(paste(folder, "/", (y - 1), ".weight", sep = "") , sep = " "), 1) 

  grad_subtract_normed = norm(grad.i - grad.j, type = "2")
  
  #rm(grad.i, grad.j, weight.i, weight.j)
  return(grad_subtract_normed / weight_subtract_normed)
})

write(lipschitz, paste(folder, "/lipschitz.txt", sep=""), sep="\n")

max(lipschitz, na.rm = TRUE) #our final result.


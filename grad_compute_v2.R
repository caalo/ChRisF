#usage:
#Rscript grad_compute.R [folder name] [number of epochs]

#We first compute grad_i = weight_i - weight_i-1, starting at i = 1. 
#Then, compute the gradient variance, and the gradient of the last epoch.

setwd('/xchip/cle/analysis/clo/ChRisF-master-NIPS')
args<-commandArgs(TRUE)
folder = as.character(args[[1]])
epochs = as.numeric(args[2])
hashSize = 2000000;

sparseToLong.fn = function(vec, sparse_default) {
	long.vec = rep(sparse_default, hashSize)
	long.vec[vec[, 1]] = vec[, 2]
	return(long.vec)
}

#first pass to get the average gradient.
grad_avg = rep(0, hashSize) #vector of avg grad.

prevWeight = sparseToLong.fn(read.delim(paste(folder, "/0.weight", sep = ""), sep = " "), 1)
for(i in 1:(epochs - 1)) {
	cat(i, "\t")
	weight_i = sparseToLong.fn(read.delim(paste(folder, "/", i, ".weight", sep = "") , sep = " "), 1)
	grad_i = weight_i - prevWeight
	prevWeight = weight_i
	grad_avg = grad_avg + grad_i
}

grad_avg = grad_avg / epochs

#second pass for results.
gradVar = rep(0, epochs)
prevWeight = sparseToLong.fn(read.delim(paste(folder, "/0.weight", sep = ""), sep = " "), 1)

for(i in 1:(epochs - 1)) {
	cat(i, "\t");
	weight_i = sparseToLong.fn(read.delim(paste(folder, "/", i, ".weight", sep = "") , sep = " "), 1)
	grad_i = weight_i - prevWeight
	prevWeight = weight_i

	diff = grad_i  - grad_avg
	gradVar[i] = diff %*% diff

	if(i == epochs - 1) {
		cat("\n", "grad norm at last epoch: ", grad_i %*% grad_i, "\n")
	}
}

write(gradVar, paste(folder, "/gradVar.txt", sep=""), sep="\n")

cat("Mean of gradVars: ", mean(gradVar, na.rm = TRUE), "\n")
cat("Max of gradVars: ", max(gradVar, na.rm = TRUE), "\n")

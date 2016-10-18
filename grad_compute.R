#usage:
#Rscript grad_compute.R [folder name] [number of epochs]

setwd('/xchip/cle/analysis/clo/ChRisF-master-NIPS')
args<-commandArgs(TRUE)
folder = as.character(args[[1]])
epochs = as.numeric(args[2])
hashSize = 2000000;
nSentences = 7936;

sparseToLong.fn = function(vec, sparse_default) {
	long.vec = rep(sparse_default, hashSize)
	long.vec[vec[, 1]] = vec[, 2]
	return(long.vec)
}

#first pass to get the average gradient.
grad_avg = rep(0, hashSize)

for(i in 0:(epochs - 1)) {
	cat(i, "\t");
	if (!is.na(file.info(paste(folder, "/", i, ".grad", sep = ""))$size) && file.info(paste(folder, "/", i, ".grad", sep = ""))$size != 0) {
		grad_i = read.delim(paste(folder, "/", i, ".grad", sep = "") , sep = " ")
		grad_i = sparseToLong.fn(grad_i, 0)
		grad_avg = grad_avg + (grad_i / nSentences)
   
  	}
	
}
grad_avg = grad_avg / epochs

#second pass for results.
gradVar = rep(0, epochs)

for(i in 0:(epochs - 1)) {
	cat(i, "\t");
	if (!is.na(file.info(paste(folder, "/", i, ".grad", sep = ""))$size) && file.info(paste(folder, "/", i, ".grad", sep = ""))$size != 0) {
		grad_i = read.delim(paste(folder, "/", i, ".grad", sep = "") , sep = " ")
		grad_i = sparseToLong.fn(grad_i, 0)
		diff = (grad_i / nSentences) - grad_avg
		gradVar[i] = diff %*% diff

		if(i == epochs - 1) {
			cat("\n", "grad norm at last epoch: ", grad_i %*% grad_i, "\n")
		}
	}
}

write(gradVar, paste(folder, "/gradVar.txt", sep=""), sep="\n")

cat("Mean of gradVars: ", mean(gradVar, na.rm = TRUE), "\n")
cat("Max of gradVars: ", max(gradVar, na.rm = TRUE), "\n")

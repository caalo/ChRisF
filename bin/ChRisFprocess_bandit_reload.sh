#!/bin/bash
clear

#params to adjust!
TRAINFILE="train_no_dev.txt"
DEVFILE="dev.txt"
EPOCHS_PER_DECODE=50

#don't touch here
prevModel=$1
epoch=$2 #epoch to START/next epoch in filename

#the learning rate
#the regularization rate
#the loss function: bayes, crossentropy, pair, probit
l=$2
r=$3
loss=$4

FOLDER="$loss$l$r/"
DATE=`date +%Y-%m-%d:%H:%M`

while [  $epoch -lt 10000 ]; do
	#okay, this is kinda a stupid way to write the model filename, but..
	fname=$FOLDER
	fname+="model_lrate"
	fname+=$l
	fname+="_reg"
	fname+=$r
	fname+="_epoch"
	fname+=$epoch
	fname+=$COMMENT
	fname+=".crf"
	echo "model file: " 
	echo $fname
	fname2=$FOLDER
	fname2+="error"$loss$l

	./ChRisF_learner -mode bandit -loadmodel $prevModel -trainf $TRAINFILE -lrate $l -lambda $r -nsentence all -epoch $EPOCHS_PER_DECODE -loss $loss -savemodel $fname 2> $fname2

		
	#decode on dev
	f1=$FOLDER
	f1+="epoch"
	f1+=$epoch
	f1+="_decode_dev.txt"
	./ChRisF_decoder -testf $DEVFILE -nsentence all -loadmodel $fname -print yes > $f1
	#decode on train
	f2=$FOLDER
	f2+="epoch"
	f2+=$epoch
	f2+="_decode_train.txt"
	./ChRisF_decoder -testf $TRAINFILE -nsentence all -loadmodel $fname -print yes > $f2


	prevModel=$fname
	epoch=epoch+EPOCHS_PER_DECODE
done




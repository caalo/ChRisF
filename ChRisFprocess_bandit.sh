#!/bin/bash
clear

#params to adjust!
TRAINFILE="train_no_dev.txt"
DEVFILE="dev.txt"
EPOCHS_PER_DECODE=50

#don't touch here
firstRun=1
prevModel=""

#first arg is the learning rate
#second arg is regularization rate
#third arg is loss function: bayes, crossentropy, pair, probit
l=$1
r=$2
loss=$3

FOLDER="$loss$l$r/"
DATE=`date +%Y-%m-%d:%H:%M`

mkdir $FOLDER


#model filename
for i in $(seq 1 1 200) #to 10,0000 epochs
do   
	#okay, this is kinda a stupid way to write the model filename, but..
	epoch=$(($i * $EPOCHS_PER_DECODE))
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

	#if the first run, we don't need to load the model file. 
	#otherwise, we need to load the model file so that it can continue where we left off from prev decoding
	if [ $firstRun -eq 1 ]; then
		./ChRisF_learner -mode bandit -loadmodel none -trainf $TRAINFILE -lrate $l -lambda $r -nsentence all -epoch $EPOCHS_PER_DECODE -loss $loss -savemodel $fname 2> $fname2
		firstRun=0
	else
		./ChRisF_learner -mode bandit -loadmodel $prevModel -trainf $TRAINFILE -lrate $l -lambda $r -nsentence all -epoch $EPOCHS_PER_DECODE -loss $loss -savemodel $fname 2> $fname2
	fi
		
	#decode on dev
	f1=$FOLDER
	f1+="epoch"
	f1+=$epoch
	f1+="_decode_dev.txt"
	./ChRisF_decoder -testf $DEVFILE -nsentence all -loadmodel $fname -print yes > $f1
	#decode on train, append to text file
	f2=$FOLDER
	f2+="epoch"
	f2+=$epoch
	f2+="_decode_train.txt"
	./ChRisF_decoder -testf $TRAINFILE -nsentence all -loadmodel $fname -print yes > $f2


	prevModel=$fname

done




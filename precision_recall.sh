#!/bin/bash
clear

$FOLDER=$1


for i in $(seq 1 1 200) #to 10,0000 epochs
do   
	#okay, this is kinda a stupid way to write the model filename, but..
	epoch=$(($i * 50))
	
	#decode on dev
	f1=$FOLDER
	f1+="epoch"
	f1+=$epoch
	f1+="_decode_dev.txt"
	grep 'Precision:' $f1 | sed 's/^.*: //' >> precision.txt
	f2=$FOLDER
	f2+="epoch"
	f2+=$epoch
	f2+="_decode_train.txt"
	grep 'Recall:' $f1 | sed 's/^.*: //' >> recall.txt

done




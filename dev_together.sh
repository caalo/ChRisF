for i in $(seq 1 1 200) #to 10,0000 epochs
do   
	epoch=$(($i * 50))
	fname="epoch"
	fname+=$epoch
	fname+="_decode_dev.txt"
	tail $fname -n 4 | head -n 1 | cut -d ' ' -f 2 >> dev_overall.txt
done


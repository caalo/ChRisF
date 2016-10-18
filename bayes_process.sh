
EPOCH=400
FOLDER="oct12/bayes"

MODEL=$FOLDER
MODEL+="/"
MODEL+=$EPOCH
MODEL+=".weight"

./ChRisF_decoder -testf test.txt -nsentence all -loadmodel $MODEL -print yes
Rscript lipschitz_compute.R $FOLDER $EPOCH 300
Rscript grad_compute.R $FOLDER $EPOCH


EPOCH=400
FOLDER="oct12/pairwise"

MODEL=$FOLDER
MODEL+="/"
MODEL+=$EPOCH
MODEL+=".weight"

./ChRisF_decoder -testf test.txt -nsentence all -loadmodel $MODEL -print yes
Rscript lipschitz_compute.R $FOLDER $EPOCH 500
Rscript grad_compute.R $FOLDER $EPOCH

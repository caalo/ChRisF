
EPOCH=399
FOLDER="oct20/bayes"

MODEL=$FOLDER
MODEL+="/"
MODEL+=$EPOCH
MODEL+=".weight"

./ChRisF_decoder -testf test.txt -nsentence all -loadmodel $MODEL -print yes
Rscript lipschitz_compute_v2.R $FOLDER $EPOCH 300
Rscript grad_compute_v2.R $FOLDER $EPOCH

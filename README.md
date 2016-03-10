# ChRisF
Conditional Random Fields (CRFs) implementation for partial feedback learning environments

See sec-chris-long.pdf for a detailed description of CRFs and our implentation on the text chunking task. 
For details of the partial feedback learning environment, our conference paper, "Learning Structured Predictors from
Partial Information", has been submitted to the Association of Computational Linguistics 2016 conference and the paper
will be available for the public soon.

To run:

ChRisF_learner trains our CRF under partial feedback or full information environments on the chunking dataset (http://www.cnts.ua.ac.be/conll2000/chunking/)
ChRisF_decoder decodes the model (in .crf extension) via standard Viterbi decoding.

Arguments for the learner:

-mode : full, bandit. 
-loadmodel: none, "yourmodel.crf"
-trainf: 
-lrate:
-lambda:
-clip:
-nsentence: full, 
-epoch 
-loss: (bayes, crossentropy, crossentropymomentum, crossentropyadadelta pairwise, probit, pairwisecrossentropy) 
-savemodel

Arguments for the decoder:
-testf 
-nsentence (all) 
-loadmodel 
-print (yes)


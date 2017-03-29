# ChRisF
Conditional Random Fields (CRFs) implementation for partial feedback learning environments.

See *sec-chris-long.pdf* for a detailed description of CRFs and our implentation on the text chunking task. 
For details of the partial feedback learning environment, our conference paper, "Learning Structured Predictors from Bandit Feedback
for Interactive NLP", was accecpted to the Association of Computational Linguistics 2016 conference and the paper can be found at *ACL2016.pdf*. Additionally, the model was benchmarked for "Stochastic Structured Prediction under Bandit Feedback", accepted for NIPS 2016 conference. 

To run:

ChRisF_learner trains our CRF under partial feedback or full information environments on the chunking dataset (http://www.cnts.ua.ac.be/conll2000/chunking/)

ChRisF_decoder decodes the model (in .crf extension) via standard Viterbi decoding.

Arguments for the learner:

**-mode : full, bandit**. full refers to full information learning, bandit refers to partial feedback learning environments, and you have to pick from the -loss argument. 

**-loadmodel: none, "yourmodel.crf"**. does the learner want to train on an existing model? Specify "none" or your own model filename. 

**-trainf:** the training dataset. It's train.txt in the directory.

**-lrate:** the learning rate.

**-lambda:** the l_2 regularization rate.

**-clip:** clipping threshold for sampled model probability. this is used for only crossentropy loss function.

**-nsentence: full, n** number of sentences to train in -trainf file. If 'full', all sentences will be used.

**-epoch:** number of epochs to run the training

**-loss: (bayes, crossentropy, crossentropymomentum, crossentropyadadelta pairwise, probit, pairwisecrossentropy)** type of loss function to use when the -mode = bandit. See details described in sec-chris-long.pdf and the paper.

**-savemodel** path to save the model

Arguments for the decoder:

**-testf** filename to test the model on

**-nsentence (all)** number of sentences to train in -testf file. If 'full', all sentences will be used.

**-loadmodel** path to load the model

**-print yes, no** whether to print detailed output, such as sample sentences and sample predicted chunk tags.


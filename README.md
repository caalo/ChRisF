# ChRisF
Conditional Random Fields (CRFs) implementation for partial feedback learning environments.

### Overview

Structured prediction from bandit feedback describes a learning scenario where instead of having access to a gold standard structure, a learner only receives partial feedback in form of the loss value of a predicted structure. On each of a sequence of rounds, the learning algorithm makes a prediction, and receives partial information in terms of feedback on the predicted point.

We tested this framework on problems in Natural Language Processing (NLP), focusing on machine translation and text chunking. In particular, I implemented Conditional Random Fields (CRFs) used for predicting the sequential parts of speech in text processing, and reformulated the learning process from a supervised environment into a bandit/partial learning environment. I compared several variations of bandit/partial feedback learning focused on different criteria of loss functions, such as expected loss minimization, pairwise preference learning, and cross-entropy minimization, and showed that all algorithms have high precision and recall with pairwise preference learning having the fastest convergence speed. 

### Analysis 

Analysis writeup can be found under `writeup/`, featuring two conference papers: "Learning Structured Predictors from Bandit Feedback for Interactive NLP", was accecpted to the Association of Computational Linguistics 2016 conference. "Stochastic Structured Prediction under Bandit Feedback", accepted for Neural Information Processing Systems 2016 conference.
### Running the software 

Under `src/`, 

#### Using existing scripts

*ChRisFprocess_bandit.sh* will train and decode given the learning rate, epochs, and mode of bandit learning.

*ChRisFprocess_bandit_reload.sh* will train and decode given the learning rate, epochs, and mode of bandit learning given existing trained model.

*ChRisFprocess_fullinfo.sh* will train and decode in a supervised setting.

#### Directly from the software

*ChRisF_learner* trains our CRF under partial feedback or full information environments on the chunking dataset (https://www.clips.uantwerpen.be/conll2000/chunking/). The data is already in the folder. Outputs a model file in `.crf` format. 

*ChRisF_decoder* decodes the `.crf` model via standard Viterbi decoding.

Arguments for the learner:

**-mode : full, bandit**. full refers to full information learning, bandit refers to partial feedback learning environments, and you have to pick from the -loss argument. 

**-loadmodel: none, "yourmodel.crf"**. does the learner want to train on an existing model? Specify "none" or your own model filename. 

**-trainf:** the training dataset filename.

**-lrate:** the learning rate.

**-lambda:** the l_2 regularization rate.

**-clip:** clipping threshold for sampled model probability. this is used for only crossentropy loss function.

**-nsentence: full, n** number of sentences to train in -trainf file. If 'full', all sentences will be used.

**-epoch:** number of epochs to run the training.

**-loss: (bayes, crossentropy, crossentropymomentum, crossentropyadadelta pairwise, probit, pairwisecrossentropy)** type of loss function to use when the -mode = bandit. See details described in sec-chris-long.pdf and the paper.

**-savemodel** path to save the model

Arguments for the decoder:

**-testf** filename to test the model on.

**-nsentence (all)** number of sentences to train in -testf file. If 'full', all sentences will be used.

**-loadmodel** path to load the model.

**-print yes, no** whether to print detailed output, such as sample sentences and sample predicted chunk tags.


# Transition based Neural Dependency Parser

Transition operator Used :

LEFTARC(0) -: Assert a head-dependent relation between the word at the top of stack and the word directly beneath it; remove the lower word from the stack.
RIGHTARC(1) -: Assert a head-dependent relation between the second word on the stack and the word at the top; remove the word at the top of the stack.
SHIFT(2) -: Remove the word from the front of the input buffer and push it onto the stack.

Designing Oracle:

Choose LEFTARC if it produces a correct head-dependent relation given the reference parse and the current configuration.
Otherwise, choose RIGHTARC if 
(1) it produces a correct head-dependent re-lation given the reference parse and 
(2) all of the dependents of the word at the top of the stack have already been assigned
Otherwise, choose SHIFT

We then create the configuration vector X and label Y using this oracle and feed it into the Neural Network model created using Tensorflow, for prediction.
 
Input dimension of Neural network is 36 and output classes is 3.
We have used xavier_initializer to initialize the input weights. With this we have seen a significant increase of 5% in accuracy. Currently my model gives an accuracy of 81.5% on Dev dataset.

This code is the reproduced version of the work which discussed on the below paper:
Pasquale, Giulia, et al. "Teaching iCub to recognize objects using deep Convolutional Neural Networks." MLIS@ ICML. 2015.
The code has been rerproduced in python. 

The goal of this paper relies on to answer: how many objects can the iCub can be recognized for each day. 

The code is running on CPU, not GPU. If one intersted to run on GPU, it can be easily done by modifiying and re-compiling the configuration file on caffe library.

Note that at the original paper, they  relied on to use the GURLS maching learning library to perfom RLS, but this code has been evaluated by using SVM library, instead. The achieved results by SVM library is not that very different. Also please make sure that you set correctly the path of the directories for caffe, model, etc. within the source-code.  

If you found any bugs and have any questions about the code, please contact me without any hesitate in bahram_lavi@yahoo.com


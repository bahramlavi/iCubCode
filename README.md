The released code provided the implimention code for the below paper: 
Pasquale, Giulia, et al. "Teaching iCub to recognize objects using deep Convolutional Neural Networks." MLIS@ ICML. 2015.

The goal of this paper relies on to answer: how many objects can the iCub can be recognized for each day. 

The code is running on CPU, not GPU. If one intersted to use GPU, it can be easily done by modifiying and re-compiling the configuration file on caffe library.

Note that  at the original paper, they  relied on the GURLS maching learning library to perfom RLS, but this code has been done by using SVM library, instead. The achieved results by SVM library is not that very different. 


If you found any bugs and have any question about the code, please contact me without any hesitate by bahram_lavi@yahoo.com


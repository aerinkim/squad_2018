This is a packge of SAN (Stochastic Answer Networks for Machine Reading Comprehension) https://arxiv.org/abs/1712.03556. 
# pls cite our paper if you use this package
 
######### SETUP ENV ###########
1. python3.6
2. install requirements:
   >pip install -r requirements.txt
3. download data/word2vec 
   >sh download.sh

Or,

Use our pre-build docker:
>docker pull allenlao/pytorchv2
###############################

######### Train SAN Model #####
1. preproces data
   >python prepro.py
2. train a model
   >python train.py
################################

by
xiaodl@microsoft.com

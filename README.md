# Explainable Recommendation System for Solving Review Loss

This is our implementation for the paper:

[Explainable Recommendation System for Solving Review Loss](http://etds.lib.ntnu.edu.tw/cgi-bin/gs32/gsweb.cgi/ccd=mpCGdV/record?r1=1&h1=0)

Author: Sean Chen (n60512@gmail.com)

# ABSTRACT
We proposed a review-base recommender system named HANN-Plus, a hierarchical attention neural network to model user’s preference and product’s preference. HANN-Plus composed of two sub-models. The first one is rating prediction model named HANN-RPM, we adjust the calculation method of attention mechanism to improve the reviews’ extraction performance of model. The second one is review generation model named HANN-RGM, which is based on encoder-decoder architecture and can be used to generate the representation for making user aware of why such products are recommended.

# Training flow
![hann-train](https://i.imgur.com/VSbKBhH.png)

# Rating Prediction Model
![hann-rpm](https://i.imgur.com/Fs3w5Dp.png)

# Review Generation Model
![RGM](https://i.imgur.com/n0zCEbE.png)

# Environments

-   Python 3
-   Pytoch
-   tqdm
-   gensim
-   numpy
-   pymysql

# Dataset

In this experiments, we use the datasets from Amazon prodct data.
(http://jmcauley.ucsd.edu/data/amazon/)

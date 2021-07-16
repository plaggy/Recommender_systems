# Recommender_systems

This repo is implementation of a simple LightFM model and a LightFM + LightGBM two-level model for restaurants recommendation.

The dataset used is a public Yelp academic dataset. **yelp_process.py** has everything to get the data ready.

In the LightFM implimentation (https://github.com/lyst/lightfm) only its collaborative-filtering-based component is used, no user / item features are supplied to it.

The LightGBM tree based model (https://github.com/microsoft/LightGBM) is used as a second level model in two-level pipeline. The idea is to use LightFM to perform fast inference on a big pool of options and select a fraction of samples with high scores. The LightGBM model is then used to perform (slower) prediction on this refined pool of candidates and select those most relevant to a user.

For training the two-level model the dataset is split into two training datasets and one test set. The first training set is used to train the LightFM model (without features). The training set for LightGBM is formed by combining the second training set with the examples from the first one which LightFM got wrong during prediction after being trained. In this way we supply to the LightGBM model the most challenging examples. The final score is calculated with the LightGBM on the test set.

The code for training two different models is run from **main.py**. There either the LightFM without features or the two-level model can be chosen by providing "lfm" or "two_level" correspondingly as a **model** variable.

The metrics being used to appraise performance are AUC and MAPK. AUC gives a quantitative estimate of how likely positive examples (liked by a user) will be ranked higher than negative (disliked).
Two-level score: 0.6785927512802481, lfm mapk: (0.18067336212728313, 0.20145442997777074)

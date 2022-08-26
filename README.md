# Recommender system

This repo is implementation of a simple LightFM model and a LightFM + LightGBM two-level model for restaurants recommendation.

The dataset used is a public Yelp academic dataset. **yelp_process.py** has everything to get the data ready.

In the LightFM implementation (https://github.com/lyst/lightfm) only its collaborative-filtering-based component is used, no user / item features are supplied to it.

The LightGBM tree based model (https://github.com/microsoft/LightGBM) is used as a second level model in the two-level pipeline. The idea is to use LightFM to perform fast inference on a big pool of options and select a fraction of samples with high scores. The LightGBM model is then used to perform (slower) prediction on this refined pool of candidates and select those most relevant to a user.

For training the two-level model the dataset is split into two training datasets and one test set. The first training set is used to train the LightFM model (without features). The training set for LightGBM is formed by combining the examples from the second training set with the highest LightFM scores with the examples from the first one which LightFM got wrong after being trained. From the 2nd set top 10 items for each user are chosen and it's top 5 for the 1st set. The final score is calculated with the LightGBM on the test set.

The code for training two different models is run from **main.py**. There either the LightFM without features or the two-level model can be chosen by providing "lfm" or "two_level" correspondingly as a **model** variable.

The metrics being used to appraise performance are AUC and MAP@K. AUC gives a quantitative estimate of how likely positive examples (liked by a user) will be ranked higher than negative (disliked); MAR@K shows how many of of the top k (ranked by scores) predicted examples are actually positive.<br />
LightFM scores: AUC: 0.402, MAP@K: 0.0367 <br />
Two-level pipeline scores: AUC: 0.628, MAP@K: 0.181

Two-level pipeline results in a significant improvement of the performance; however, it is more time consuming.

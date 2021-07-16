import pandas as pd
import numpy as np
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from utils import lfm_mapk, lfm_auc


def train_test_split(datafolder, split):
    stars_df = pd.read_csv(datafolder + 'stars_df_AZ.csv')
    stars_df = stars_df.drop(stars_df.filter(regex="Unname"), axis=1)
    stars_df = stars_df.groupby(['user_id', 'business_id'], as_index=False).stars.mean()

    # if a review has less than 4 stars, label as 0 otherwise 1
    stars_df.stars = stars_df.stars.apply(lambda x: 0 if x < 4 else 1)
    user_counts = stars_df.user_id.value_counts()
    business_counts = stars_df.business_id.value_counts()
    valid_df = stars_df[stars_df.user_id.isin(user_counts[user_counts > 20].index) &
                        stars_df.business_id.isin(business_counts[business_counts > 20].index)]

    # test examples are sampled from a subset of businesses & users which have > 20
    # corresponding entries
    test_df = valid_df.sample(frac=split, random_state=11)
    train_df = stars_df.loc[~stars_df.index.isin(test_df.index)]

    id_cols = ['user_id', 'business_id']
    transformed_train = dict()
    transformed_test = dict()
    for k in id_cols:
        cate_enc = preprocessing.LabelEncoder()
        transformed_train[k] = cate_enc.fit_transform(train_df[k].values)
        transformed_test[k] = cate_enc.transform(test_df[k].values)
        test_df[k] = cate_enc.transform(test_df[k].values)

    n_users = len(np.unique(transformed_train['user_id']))
    n_items = len(np.unique(transformed_train['business_id']))
    train = coo_matrix((train_df.stars, (transformed_train['user_id'], transformed_train['business_id'])),
                       shape=(n_users, n_items))
    test = coo_matrix((test_df.stars, (transformed_test['user_id'], transformed_test['business_id'])),
                      shape=(n_users, n_items))

    return train, test, test_df


def run_lightfm(datafolder, n_comp, epochs):
    train, test, test_df = train_test_split(datafolder, 0.15)

    model = LightFM(no_components=n_comp, learning_rate=0.005, loss='warp')
    model.fit(train, epochs=epochs, num_threads=4, verbose=2)

    auc_score = lfm_auc(model, test_df)
    mapk_score = lfm_mapk(model, test_df, 10)

    print(f"auc scorec: {auc_score}")
    print(f"mapk score: {mapk_score}")

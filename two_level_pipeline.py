import pandas as pd
from sklearn import preprocessing
from lightfm import LightFM
import pickle
import lightgbm as lgb
import os
import time
from utils import *
import copy
from os.path import abspath, dirname


def extract_features(train1, train2, test, data_df, users_df, business_df, datadir):
    users_df = users_df.drop(users_df.filter(regex="Unname"), axis=1)
    business_df = business_df.drop(business_df.filter(regex="Unname"), axis=1)
    business_df = business_df.drop(columns=['latitude', 'longitude', 'attributes_BusinessAcceptsCreditCards',
                           'attributes_RestaurantsReservations', 'touristy'])
    business_df = pd.get_dummies(business_df, columns=['attributes_Alcohol', 'attributes_NoiseLevel'])
    business_df = business_df.rename(columns={'stars': 'business_stars'})
    users_df = users_df.drop(columns=['name', 'yelping_since', 'elite'])

    users_df.friends = users_df.friends.apply(lambda x: x.split(', '))

    train1_users = pd.merge(train1, users_df, how='left', on=['user_id'])
    train2_users = pd.merge(train2, users_df, how='left', on=['user_id'])
    test_users = pd.merge(test, users_df, how='left', on=['user_id'])
    users_by_business = data_df.groupby('business_id')['user_id'].apply(list)

    train1_users.friends = train1_users.apply(lambda r: friend_visited(r.friends, users_by_business.loc[r.business_id]), axis=1)
    train2_users.friends = train2_users.apply(lambda r: friend_visited(r.friends, users_by_business.loc[r.business_id]), axis=1)
    test_users.friends = test_users.apply(lambda r: friend_visited(r.friends, users_by_business.loc[r.business_id]), axis=1)
    train1_users = train1_users.drop(columns=['business_id', 'stars'])
    train2_users = train2_users.drop(columns=['business_id', 'stars'])
    test_users = test_users.drop(columns=['business_id', 'stars'])

    train1_users.to_csv(datadir + 'train1_users_features.csv', index=False)
    train2_users.to_csv(datadir + 'train2_users_features.csv', index=False)
    test_users.to_csv(datadir + 'test_users_features.csv', index=False)

    train1_items = pd.merge(train1, business_df, how='left', on=['business_id'])
    train2_items = pd.merge(train2, business_df, how='left', on=['business_id'])
    test_items = pd.merge(test, business_df, how='left', on=['business_id'])
    train1_items = train1_items.drop(columns=['user_id', 'stars'])
    train2_items = train2_items.drop(columns=['user_id', 'stars'])
    test_items = test_items.drop(columns=['user_id', 'stars'])

    train1_items.to_csv(datadir + 'train1_items_features.csv', index=False)
    train2_items.to_csv(datadir + 'train2_items_features.csv', index=False)
    test_items.to_csv(datadir + 'test_items_features.csv', index=False)


def target_encode(col, target):
    for val in col.unique():
        avg = target[col == val].mean()
        col.loc[col == val] = avg

    return col


def sample(chunk, frac):
    out = chunk.sample(frac=frac, replace=False, random_state=1)
    return out


def data_split(data_df, users_df, output_dir):
    data_df = data_df.drop(data_df.filter(regex="Unname"), axis=1)
    data_df = data_df.groupby(['user_id', 'business_id'], as_index=False).stars.mean()

    data_df.stars = data_df.stars.apply(lambda x: 0 if x < 4 else 1)

    # for the refined_df only users and businesses having
    # more than 5 corresponding data samples were chosen
    user_counts = data_df.user_id.value_counts()
    business_counts = data_df.business_id.value_counts()
    refined_df = data_df[data_df.user_id.isin(user_counts[user_counts > 5].index) &
                        data_df.business_id.isin(business_counts[business_counts > 5].index)]

    # the part 2 of the training set is obtained by sampling 0.35 data points from the refined_df
    train2 = refined_df.groupby('user_id', group_keys=False).apply(lambda x: sample(x, 0.35))
    # the test set is obtained by sampling 0.15 data points from the refined_df not present in train2
    test = refined_df[~refined_df.index.isin(train2.index)].groupby('user_id', group_keys=False).apply(lambda x: sample(x, 0.15))
    train1 = data_df[~(data_df.index.isin(train2.index) | data_df.index.isin(test.index))]

    train1 = train1[train1.user_id.isin(users_df.user_id)]
    train2 = train2[train2.user_id.isin(users_df.user_id)]
    test = test[test.user_id.isin(users_df.user_id)]

    train1.to_csv(output_dir + 'train_set_1_stars.csv', index=False)
    train2.to_csv(output_dir + 'train_set_2_stars.csv', index=False)
    test.to_csv(output_dir + 'test_set_stars.csv', index=False)

    return train1, train2, test


def encode_set(set1, set2, set3, if_pickle, output_dir):
    id_cols = ['user_id', 'business_id']
    set1_transformed = {}
    set2_transformed = copy.copy(set2)
    set2_transformed = set2_transformed[set2_transformed.user_id.isin(set1.user_id) &
                                        set2_transformed.business_id.isin(set1.business_id)]
    set3_transformed = copy.copy(set3)
    set3_transformed = set3_transformed[set3_transformed.user_id.isin(set1.user_id) &
                                        set3_transformed.business_id.isin(set1.business_id)]
    print(f"lfm set2 set len: {len(set2_transformed)}")
    print(f"lfm set3 set len: {len(set3_transformed)}")
    for k in id_cols:
        cate_enc = preprocessing.LabelEncoder()
        set1_transformed[k] = cate_enc.fit_transform(set1[k].values)
        set2_transformed[k] = cate_enc.transform(set2_transformed[k].values)
        set3_transformed[k] = cate_enc.transform(set3_transformed[k].values)
        if if_pickle:
            with open(output_dir + f'{k}_encoder', 'wb') as file:
                pickle.dump(cate_enc, file, protocol=pickle.HIGHEST_PROTOCOL)

    return set1_transformed, set2_transformed, set2_transformed


def train_lfm(train1, train2, test, users_df, n_comp, epochs, output_dir, datadir):
    train1 = train1[train1.user_id.isin(users_df.user_id)]
    train2 = train2[train2.user_id.isin(train1.user_id) & train2.business_id.isin(train1.business_id)]

    train1_transformed, train2_transformed, test_transformed = \
        encode_set(train1, train2, test, if_pickle=True, output_dir=output_dir)

    # form a sparse matrix to train a LightFM model
    n_users = len(np.unique(train1_transformed['user_id']))
    n_items = len(np.unique(train1_transformed['business_id']))
    train_data = coo_matrix((train1.stars, (train1_transformed['user_id'], train1_transformed['business_id'])),
                           shape=(n_users, n_items))

    model_lfm = LightFM(no_components=n_comp, learning_rate=0.005, loss='warp')
    model_lfm.fit(train_data, epochs=epochs, num_threads=4, verbose=2)

    lfm_auc_score = lfm_auc(model_lfm, test_transformed)
    lfm_mapk_score = lfm_mapk(model_lfm, test_transformed, 10)
    print(f"lfm auc: {lfm_auc_score}, lfm mapk: {lfm_mapk_score}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_dir + 'lfm_model', 'wb') as file:
        pickle.dump(model_lfm, file, protocol=pickle.HIGHEST_PROTOCOL)

    # calculate lfm predictions for the first part of the training set
    usr_rep = model_lfm.get_user_representations()
    item_rep = model_lfm.get_item_representations()
    users_to_predict1 = train1_transformed['user_id']
    items_to_predict1 = train1_transformed['business_id']
    user_test_emb1 = usr_rep[1][users_to_predict1] + usr_rep[0][users_to_predict1].reshape(-1, 1)
    item_test_emb1 = item_rep[1][items_to_predict1] + item_rep[0][items_to_predict1].reshape(-1, 1)
    lfm_pred1 = np.sum(np.multiply(user_test_emb1, item_test_emb1), axis=1)

    # save 5 items with the highest lfm scores from the set 1 for future use in lgb training
    lfm_output1 = copy.deepcopy(train1)
    lfm_output1['stars'] = lfm_pred1.tolist()
    lfm_output1 = lfm_output1.groupby('user_id', as_index=False).apply(lambda x: x.nlargest(5, 'stars'))
    lfm_output1.to_csv(datadir + 'lfm_pred1_for_lgb_stars.csv', index=False)

    users_to_predict2 = train2_transformed['user_id']
    items_to_predict2 = train2_transformed['business_id']
    user_test_emb2 = usr_rep[1][users_to_predict2] + usr_rep[0][users_to_predict2].reshape(-1, 1)
    item_test_emb2 = item_rep[1][items_to_predict2] + item_rep[0][items_to_predict2].reshape(-1, 1)
    lfm_pred2 = np.sum(np.multiply(user_test_emb2, item_test_emb2), axis=1)

    # save 10 items with the highest lfm scores from the set 2 for future use in lgb training
    lfm_output2 = copy.copy(train2)
    lfm_output2['stars'] = lfm_pred2.tolist()
    lfm_output2 = lfm_output2.groupby('user_id', as_index=False).apply(lambda x: x.nlargest(10, 'stars'))
    lfm_output2.to_csv(datadir + 'lfm_pred2_for_lgb_stars.csv', index=False)


def train_lgb_w_features(train1, train1_users, train1_items, train2, train2_users, train2_items,
                        test, test_users, test_items, users_df, lfm_output1, lfm_output2, lgb_params, output_dir):
    train1 = train1[train1.user_id.isin(users_df.user_id)]
    train2 = train2[train2.user_id.isin(users_df.user_id)]
    test = test[test.user_id.isin(users_df.user_id)]

    lfm_correct_in_set1 = lfm_output1.set_index(['user_id', 'business_id']).index.isin(train1[train1.stars == 1].set_index(
        ['user_id', 'business_id']).index)
    lfm_output1 = lfm_output1[~lfm_correct_in_set1]
    lfm_output1['stars'] = 0

    train2 = train2.sort_values(['user_id', 'business_id'])
    lfm_output2 = lfm_output2.sort_values(['user_id', 'business_id'])
    g = train2[train2.set_index(['user_id', 'business_id']).index.isin(lfm_output2.set_index(
        ['user_id', 'business_id']).index)]
    lfm_output2['stars'] = train2[train2.set_index(['user_id', 'business_id']).index.isin(lfm_output2.set_index(
        ['user_id', 'business_id']).index)]['stars']

    train_1_plus_2 = lfm_output1.append(lfm_output2)

    train1_in_lfm_output = train1.set_index(['user_id', 'business_id']).index.isin(lfm_output1.
                                                                         set_index(['user_id', 'business_id']).index)
    train2_in_lfm_output = train2.set_index(['user_id', 'business_id']).index.isin(lfm_output2.
                                                                         set_index(['user_id', 'business_id']).index)

    train_lgb_users = train1_users.iloc[train1_in_lfm_output].append(train2_users.iloc[train2_in_lfm_output])
    train_lgb_items = train1_items.iloc[train1_in_lfm_output].append(train2_items.iloc[train2_in_lfm_output])
    train_lgb_items = train_lgb_items.rename(columns={'review_count': 'review_count_business'})

    train_lgb = pd.concat([train_lgb_users, train_lgb_items], axis=1)
    train_lgb = train_lgb.sort_values('user_id')
    train_1_plus_2 = train_1_plus_2.sort_values('user_id')
    train_query_num = train_lgb.user_id.value_counts(sort=False).reindex(train_lgb.user_id.unique())
    train_lgb = train_lgb.drop(columns=['user_id', 'business_id'])

    test_items = test_items.rename(columns={'review_count': 'review_count_business'})
    test_lgb = pd.concat([test_users, test_items], axis=1)
    #test_lgb = test_lgb.drop(columns=['user_id', 'business_id'])

    train_data_lgb = lgb.Dataset(train_lgb, label=train_1_plus_2.stars, group=train_query_num.values)
    lgb_model = lgb.train(lgb_params, train_data_lgb)
    lgb_model.save_model(output_dir + 'lgb_model')

    lgb_auc_score = lgb_auc(lgb_model, test_lgb, test)
    lgb_mapk_score = lgb_mapk(lgb_model, test_lgb, test, 10)

    print(f"lgb auc: {lgb_auc_score}, lfm mapk: {lgb_mapk_score}")


def run_two_level(n_comp, epochs, homedir, datadir, task=None):
    output_dir = homedir + f'/two_level_{time.strftime("%m%d-%H%M")}_out/'
    data_df = pd.read_csv(datadir + 'stars_df_AZ.csv')
    users_df = pd.read_csv(datadir + 'users_df_AZ.csv')
    business_df = pd.read_csv(datadir + 'business_filtered_AZ.csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # data preparation stage
    if task == "data_prep":
        train1, train2, test = data_split(data_df, users_df, datadir)
        extract_features(train1, train2, test, data_df, users_df, business_df, datadir)

    # lfm training - first level of the model
    elif task == "train_lfm":
        # files created after "data_split" are loaded
        train1 = pd.read_csv(datadir + 'train_set_1_stars.csv')
        train2 = pd.read_csv(datadir + 'train_set_2_stars.csv')
        test = pd.read_csv(datadir + 'test_set_stars.csv')

        train_lfm(train1=train1,
                  train2=train2,
                  test=test,
                  users_df=users_df,
                  n_comp=n_comp,
                  epochs=epochs,
                  output_dir=output_dir,
                  datadir=datadir)

    # lgb training - second level of the model
    elif task == "train_lgb":
        train1 = pd.read_csv(datadir + 'train_set_1_stars.csv')
        train2 = pd.read_csv(datadir + 'train_set_2_stars.csv')

        test = pd.read_csv(datadir + 'test_set_stars.csv')
        train1_users = pd.read_csv(datadir + 'train1_users_features.csv')
        train2_users = pd.read_csv(datadir + 'train2_users_features.csv')
        test_users = pd.read_csv(datadir + 'test_users_features.csv')
        train1_items = pd.read_csv(datadir + 'train1_items_features.csv')
        train2_items = pd.read_csv(datadir + 'train2_items_features.csv')
        test_items = pd.read_csv(datadir + 'test_items_features.csv')

        print(train1.shape)
        print(train1_users.shape)
        print(train1_items.shape)

        # file created after "train_lfm" is run
        lfm_output1 = pd.read_csv(datadir + 'lfm_pred1_for_lgb_stars.csv')
        lfm_output2 = pd.read_csv(datadir + 'lfm_pred2_for_lgb_stars.csv')

        lgb_params = {
            "task": "train",
            "num_leaves": 50,
            "min_data_in_leaf": 20,
            "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 3, 5, 10],
            "learning_rate": .1,
            "num_threads": 2
        }

        train_lgb_w_features(train1=train1,
                             train1_users=train1_users,
                             train1_items=train1_items,
                             train2=train2,
                             train2_users=train2_users,
                             train2_items=train2_items,
                             test=test,
                             test_users=test_users,
                             test_items=test_items,
                             users_df=users_df,
                             lfm_output1=lfm_output1,
                             lfm_output2=lfm_output2,
                             lgb_params=lgb_params,
                             output_dir=output_dir)
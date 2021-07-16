import numpy as np
import ml_metrics
from sklearn.metrics import roc_auc_score
from scipy.sparse import coo_matrix


def lfm_auc(model, test_df):
    truth_items = test_df.groupby('user_id').business_id.apply(list).to_dict()
    truth_scores = test_df.groupby('user_id').stars.apply(list).to_dict()
    usr_rep = model.get_user_representations()
    item_rep = model.get_item_representations()
    aucs = []

    for (user, items), (_, scores) in zip(truth_items.items(), truth_scores.items()):
        user_test_emb = usr_rep[1][user] + usr_rep[0][user].reshape(-1, 1)
        item_test_emb = item_rep[1][items] + item_rep[0][items].reshape(-1, 1)
        predictions = np.dot(user_test_emb, item_test_emb.T)

        if len(np.unique(scores)) > 1:
            aucs.append(roc_auc_score((np.array(scores) - min(scores)) / (max(scores) - min(scores)), predictions.flatten()))

    return sum(aucs) / len(aucs)


def lfm_mapk(model, test_df, k):
    true_items = test_df.groupby('user_id').business_id.apply(list).to_dict()
    true_scores = test_df.groupby('user_id').stars.apply(list).to_dict()
    usr_rep = model.get_user_representations()
    item_rep = model.get_item_representations()
    all_user_emb = usr_rep[1] + usr_rep[0].reshape(-1, 1)
    all_item_emb = item_rep[1] + item_rep[0].reshape(-1, 1)
    pred = []
    true = []
    max_mapk = []

    for (user, items), (_, scores) in zip(true_items.items(), true_scores.items()):
        user_test_emb = all_user_emb[user]
        pred_ranks = np.argsort(np.dot(user_test_emb, all_item_emb.T))

        pred.append(pred_ranks)
        pos_items = np.array(items)[np.array(scores) == 1]
        true.append(pos_items)
        if len(pos_items) > 0:
            max_mapk.append(min(k, len(pos_items)) / k)

    pred_mapk = ml_metrics.mapk(true, pred, k)
    max_mapk = np.mean(max_mapk)

    return pred_mapk, max_mapk


def lgb_auc(lgb_model, test_lgb, test):
    truth_items = test.groupby('user_id').business_id.apply(list).to_dict()
    truth_scores = test.groupby('user_id').stars.apply(list).to_dict()
    aucs = []

    for (user, items), (_, scores) in zip(truth_items.items(), truth_scores.items()):
        chunk = test_lgb[(test_lgb.user_id == user) & test_lgb.business_id.isin(items)]
        chunk = chunk.drop(columns=['user_id', 'business_id'])
        predictions = lgb_model.predict(chunk)

        if len(np.unique(scores)) > 1:
            aucs.append(roc_auc_score((np.array(scores) - min(scores)) / (max(scores) - min(scores)), predictions.flatten()))

    return sum(aucs) / len(aucs)


def lgb_mapk(lgb_model, test_lgb, test, k):
    true_items = test.groupby('user_id').business_id.apply(list).to_dict()
    true_scores = test.groupby('user_id').stars.apply(list).to_dict()
    pred = []
    true = []
    max_mapk = []

    for (user, items), (_, scores) in zip(true_items.items(), true_scores.items()):
        chunk = test_lgb[(test_lgb.user_id == user) & test_lgb.business_id.isin(items)]
        chunk = chunk.drop(columns=['user_id', 'business_id'])
        pred_score = lgb_model.predict(chunk)
        pred_ranks = np.argsort(pred_score)

        pred.append(pred_ranks)
        pos_items = np.array(items)[np.array(scores) == 1]
        true.append(pos_items)
        if len(pos_items) > 0:
            max_mapk.append(min(k, len(pos_items)) / k)

    pred_mapk = ml_metrics.mapk(true, pred, k)
    max_mapk = np.mean(max_mapk)

    return pred_mapk, max_mapk


def create_sparse_features(user_df, item_df):
    user_features = coo_matrix((user_df, (user_df.index, user_df.columns)),
                       shape=(len(user_df), len(user_df.columns)))
    item_features = coo_matrix((item_df, (item_df.index, item_df.columns0)),
                       shape=(len(item_df), len(item_df.columns)))

    return user_features, item_features


def friend_visited(friends, rests):
    if (friends is not np.nan) and (friends[0] is not None):
        inters = list(set(friends) & set(rests))
        return int(len(inters) > 0)

    return 0



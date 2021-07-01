import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from lightgbm.sklearn import LGBMClassifier
from collections import defaultdict
import gc #gc的库是python用于垃圾回收的
import time

FEED_EMBEDDING_DIR="data/feed_embeddings_PCA.csv"
KEYWORDS_TAGS_DIR = "feedid_keywords_tags.csv"
FEED_EMBEDDINGS_ONEHOT_DIR = "feed_embeddings_100_k_means.csv"
USER_LATENT_DIR = "user_latent_features.csv"


#所有需要预测的项目
play_cols = ['is_finish', 'play_times', 'play', 'stay']
y_list = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']

pd.set_option('display.max_columns', None)

'''
这个reduce_mem是用来调整内存使用的
'''
def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

## 从官方baseline里面抽出来的评测函数
'''
输入参数为真实标签，预测值，用户列表
'''
def uAUC(labels, preds, user_id_list):
    """Calculate user AUC"""
    user_pred = defaultdict(lambda: [])
    user_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        pred = preds[idx]
        truth = labels[idx]
        user_pred[user_id].append(pred)
        user_truth[user_id].append(truth)

    user_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = user_truth[user_id]
        flag = False
        # 若全是正样本或全是负样本，则flag为False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        user_flag[user_id] = flag

    total_auc = 0.0
    size = 0.0
    for user_id in user_flag:
        if user_flag[user_id]:
            auc = roc_auc_score(np.asarray(user_truth[user_id]), np.asarray(user_pred[user_id]))
            total_auc += auc 
            size += 1.0
    user_auc = float(total_auc)/size
    return user_auc



def statis_feature(start_day=1, before_day=7, agg=['sum',]):
    """
    统计用户/feed 过去n天各类行为的次数
    :param start_day: Int. 起始日期
    :param before_day: Int. 时间范围（天数）
    :param agg: String. 统计方法
    """
    # 读取用户行为信息表
    history_data = pd.read_csv(USER_ACTION)[["userid", "date_", "feedid"] + FEA_COLUMN_LIST]
    feature_dir = os.path.join(ROOT_PATH, "feature")
    for dim in ["userid", "feedid"]:
        print(dim)
        user_data = history_data[[dim, "date_"] + FEA_COLUMN_LIST]
        res_arr = []
        for start in range(start_day, 14 - before_day + 1):
            # 开始日期为1，结束日期为15,before_day=7（时间范围）,start的取值范围比较明显是从1取到了8.
            # 这里统计了每7天的数据, 统计的内容是每一个用户每7天看了多少视频，读了多少次评论，点了多少次赞，转发了多少次等等
            # 对于feed来说也是类似的，统计了每7天的这些属性内容
            '''
            提取的特征包括：read_commentsum, likesum, click_avatarsum, forwardsum, commentsum, followsum, favoritesum, date_
            '''
            for a_id in agg:
                temp = user_data[((user_data["date_"]) >= start) & (user_data["date_"] < (start + before_day))]
                temp = temp.drop(columns=['date_'])
                # 根据userid进行归类
                temp = temp.groupby([dim]).agg([a_id]).reset_index()
                temp.columns = list(map(''.join, temp.columns.values))
                temp["date_"] = start + before_day
                res_arr.append(temp)
        # 实际上这一块儿统计的是用户在1-7,2-8,3-9,...,8-14每一个7天的统计量
        dim_feature = pd.concat(res_arr)
        feature_path = os.path.join(feature_dir, dim + "_feature.csv")
        print('Save to: %s' % feature_path)
        dim_feature.to_csv(feature_path, index=False)


def sliding_windows(df,n_days=[5,7], max_day=15):
    for n_day in n_days:
            ## 统计历史5天的曝光、转化、视频观看等情况（此处的转化率统计其实就是target encoding）
        for stat_cols in tqdm([
            ['userid'],
            ['feedid'],
            ['authorid'],
            #['userid','feedid'],
            ['userid', 'authorid'],
            ['feedid', 'authorid']
        ]):
            f = '_'.join(stat_cols)
            stat_df = pd.DataFrame()
            #从第二天到第14天
            for target_day in range(2, max_day + 1):
                #使用的滑动窗口的特征
                #left表示滑动窗口的左边，right表示滑动窗口的右边
                left, right = max(target_day - n_day, 1), target_day - 1
                tmp = df[((df['date_'] >= left) & (df['date_'] <= right))].reset_index(drop=True)
                tmp['date_'] = target_day
                tmp['{}_{}day_count'.format(f, n_day)] = tmp.groupby(stat_cols)['date_'].transform('count')
                g = tmp.groupby(stat_cols)
                tmp['{}_{}day_finish_rate'.format(f, n_day)] = g[play_cols[0]].transform('mean')
                feats = ['{}_{}day_count'.format(f, n_day), '{}_{}day_finish_rate'.format(f, n_day)]
                '''
                play_cols = ['is_finish', 'play_times', 'play', 'stay']
                '''
                for x in play_cols[1:]:
                    #for stat in ['min','max','mean','std','median']:
                    for stat in ['min','max','mean']:
                        tmp['{}_{}day_{}_{}'.format(f, n_day, x, stat)] = g[x].transform(stat)
                        feats.append('{}_{}day_{}_{}'.format(f, n_day, x, stat))
                #for y in y_list[:4]:
                for y in y_list:
                    tmp['{}_{}day_{}_sum'.format(f, n_day, y)] = g[y].transform('sum')
                    tmp['{}_{}day_{}_mean'.format(f, n_day, y)] = g[y].transform('mean')
                    feats.extend(['{}_{}day_{}_sum'.format(f, n_day, y), '{}_{}day_{}_mean'.format(f, n_day, y)])

                tmp = tmp[stat_cols + feats + ['date_']].drop_duplicates(stat_cols + ['date_']).reset_index(drop=True)
                stat_df = pd.concat([stat_df, tmp], axis=0, ignore_index=True)
                del g, tmp
            df = df.merge(stat_df, on=stat_cols + ['date_'], how='left')
            del stat_df
            gc.collect()
    return df

def global_features(df):
    ## 全局信息统计，包括曝光、偏好等，略有穿越，但问题不大，可以上分，只要注意不要对userid-feedid做组合统计就行
    for f in tqdm(['userid', 'feedid', 'authorid']):
        df[f + '_count'] = df[f].map(df[f].value_counts())
    for f1, f2 in tqdm([
        ['userid', 'feedid'],
        ['userid', 'authorid']
    ]):
        df['{}_in_{}_nunique'.format(f1, f2)] = df.groupby(f2)[f1].transform('nunique')
        df['{}_in_{}_nunique'.format(f2, f1)] = df.groupby(f1)[f2].transform('nunique')
    for f1, f2 in tqdm([
        ['userid', 'authorid']
    ]):
        df['{}_{}_count'.format(f1, f2)] = df.groupby([f1, f2])['date_'].transform('count')
        df['{}_in_{}_count_prop'.format(f1, f2)] = df['{}_{}_count'.format(f1, f2)] / (df[f2 + '_count'] + 1)
        df['{}_in_{}_count_prop'.format(f2, f1)] = df['{}_{}_count'.format(f1, f2)] / (df[f1 + '_count'] + 1)
    df['videoplayseconds_in_userid_mean'] = df.groupby('userid')['videoplayseconds'].transform('mean')
    df['videoplayseconds_in_authorid_mean'] = df.groupby('authorid')['videoplayseconds'].transform('mean')
    df['feedid_in_authorid_nunique'] = df.groupby('authorid')['feedid'].transform('nunique')
    # #将降维后的embedding加到里面去
    df_feed_embedding=pd.read_csv(FEED_EMBEDDING_DIR)
    df = df.merge(df_feed_embedding, on='feedid', how='left')
    #将keywords和tag的数据添加进去
    df_keywords_tags = pd.read_csv(KEYWORDS_TAGS_DIR)
    df = df.merge(df_keywords_tags,on='feedid',how='left')
    #将user的latent的向量信息加进去
    df_user_latent = pd.read_csv(USER_LATENT_DIR)
    df = df.merge(df_user_latent,on='userid',how='left')
    ## 内存够用的不需要做这一步
    df = reduce_mem(df, [f for f in df.columns if f not in ['date_'] + play_cols + y_list])
    train = df[~df['read_comment'].isna()].reset_index(drop=True)
    test = df[df['read_comment'].isna()].reset_index(drop=True)
    cols = [f for f in df.columns if f not in ['date_'] + play_cols + y_list]
    print(train[cols].shape)
    # print(train.columns)
    return train,test,cols

# columns_idxs = []
# for col_idx in train.columns:
#     columns_idxs.append(col_idx)
#将所有的columns保存到一个txt中
# def save_to_file(file_name, contents):
#     fh = open(file_name, 'w')
#     fh.write(contents)
#     fh.close() 
# save_to_file('lgb_features_columns.txt',';'.join(columns_idxs))

#按行写入
# def write_features_func(arr):
#     for i in arr:
#         f = open('features_all.txt','a', encoding='utf-8')
#         f.writelines(str(i)+'\n')
#         f.close()
# write_features_func(columns_idxs)

def offline_train(train,test,cols):
    trn_x = train[train['date_'] < 14].reset_index(drop=True)
    val_x = train[train['date_'] == 14].reset_index(drop=True)
    ##################### 线下验证 #####################
    uauc_list = []
    r_list = []
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        clf = LGBMClassifier(
            learning_rate=0.05,
            n_estimators=5000,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=2021,
            metric='None'
        )
        clf.fit(
            trn_x[cols], trn_x[y],
            eval_set=[(val_x[cols], val_x[y])],
            eval_metric='auc',
            early_stopping_rounds=100,
            verbose=50
        )
        val_x[y + '_score'] = clf.predict_proba(val_x[cols])[:, 1]
        val_uauc = uAUC(val_x[y], val_x[y + '_score'], val_x['userid'])
        uauc_list.append(val_uauc)
        print(val_uauc)
        r_list.append(clf.best_iteration_)
        print('runtime: {}\n'.format(time.time() - t))
    weighted_uauc = 0.4 * uauc_list[0] + 0.3 * uauc_list[1] + 0.2 * uauc_list[2] + 0.1 * uauc_list[3]
    print(uauc_list)
    print(weighted_uauc)
    return r_list,uauc_list,weighted_uauc
##################### 全量训练 #####################
'''
这里可以考虑开始做K折交叉检验的工作
'''
def total_train(train,test,y_list,r_list,u_auc_list,w_uauc,seeds=[1500,2021]):
    r_dict = dict(zip(y_list[:4], r_list))
    for y in y_list[:4]:
        print('=========', y, '=========')
        t = time.time()
        res = []
        for seed in seeds:
            clf = LGBMClassifier(
                learning_rate=0.05,
                n_estimators=r_dict[y],
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed
            )
            clf.fit(
                train[cols], train[y],
                eval_set=[(train[cols], train[y])],
                early_stopping_rounds=r_dict[y],
                verbose=100
            )
            res.append(clf.predict_proba(test[cols])[:, 1])
        res = np.array(res)
        test[y] = np.mean(res,axis=0)
        print('runtime: {}\n'.format(time.time() - t))
        test[['userid', 'feedid'] + y_list[:4]].to_csv('submit_%.6f_%.6f_%.6f_%.6f_%.6f.csv' % (w_uauc, u_auc_list[0], u_auc_list[1], u_auc_list[2], u_auc_list[3]),index=False)
        print("Finished!")  

        
if __name__ == "__main__":
    ## 读取训练集整个训练数据： 维度(7317882, 13)
    train = pd.read_csv('../data/wechat_algo_data1/user_action.csv')
    print(train.shape)
    #这个地方计算了每个预测模型对应的均值结果: 
    '''
    read_comment 0.03501586934580252
    like 0.02580487086290815
    click_avatar 0.007533327266004016
    forward 0.0038211876059220415
    favorite 0.0013424649372591687
    comment 0.00040462527272235326
    follow 0.0007211102884687126
    '''
    # for y in y_list:
    #     print(y, train[y].mean())
    ## 读取测试集
    test = pd.read_csv('../data/wechat_algo_data1/test_a.csv')
    test['date_'] = 15
    print(test.shape) #测试数据的维度：(421985, 4)
    ## 合并处理
    df = pd.concat([train, test], axis=0, ignore_index=True)
    print(df.head(3))
    ## 读取视频信息表
    feed_info = pd.read_csv('../data/wechat_algo_data1/feed_info.csv')
    feed_info = feed_info[[
        'feedid', 'authorid', 'videoplayseconds','bgm_song_id','bgm_singer_id'
    ]]
    df = df.merge(feed_info, on='feedid', how='left')
    ## 视频时长是秒，转换成毫秒，才能与play、stay做运算
    df['videoplayseconds'] *= 1000
    ## 是否观看完视频（其实不用严格按大于关系，也可以按比例，比如观看比例超过0.9就算看完）
    df['is_finish'] = (df['play'] >= df['videoplayseconds']).astype('int8')
    df['play_times'] = df['play'] / df['videoplayseconds']
    #添加滑动窗口的特征
    df = sliding_windows(df,n_days=[3,5,7], max_day=15)
    #添加全局特征
    train_data,test_data,cols = global_features(df)
    #线下训练的轮数
    r_list, u_auc_list,w_uauc = offline_train(train_data,test_data,cols)
    #全量训练结果
    total_train(train_data,test_data,y_list,r_list,u_auc_list,w_uauc,seeds=[2021])

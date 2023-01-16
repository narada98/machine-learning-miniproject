#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_dataset_ = pd.read_feather('../input/amexfeather/train_data.ftr')
# Keep the latest statement features for each customer
train_dataset = train_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()
# train_dataset = train_dataset.reset_index(drop = False)


# In[ ]:


train_dataset['target']


# In[ ]:


del train_dataset_
gc.collect()


# In[ ]:


percent_missing = train_dataset.isnull().sum() * 100 / len(train_dataset)
missing_value_df = pd.DataFrame({'column_name': train_dataset.columns,
                                 'percent_missing': percent_missing})


# In[ ]:


high_null_columns = missing_value_df[missing_value_df['percent_missing'] > 0].column_name.to_list()
len(high_null_columns)


# In[ ]:


missing_value_df[missing_value_df['percent_missing'] < 75].shape


# In[ ]:


missing_value_df[missing_value_df['percent_missing'] == 0].shape


# In[ ]:


#dropping columns with high null value percentages
train_dataset = train_dataset.drop(['S_2','D_66','D_42','D_49','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)


# In[ ]:


train_dataset.shape


# In[ ]:


test_dataset_ = pd.read_feather('../input/amexfeather/test_data.ftr')
# Keep the latest statement features for each customer
test_dataset = test_dataset_.groupby('customer_ID').tail(1).set_index('customer_ID', drop=True).sort_index()
# test_dataset = test_dataset.reset_index(drop = False)


# In[ ]:


del test_dataset_
gc.collect()


# In[ ]:


test_dataset = test_dataset.drop(['S_2','D_42','D_49','D_66','D_73','D_76','R_9','B_29','D_87','D_88','D_106','R_26','D_108','D_110','D_111','B_39','B_42','D_132','D_134','D_135','D_136','D_137','D_138','D_142'], axis=1)


# In[ ]:


categorical_columns = train_dataset.select_dtypes(exclude = 'number').columns.to_list()
print(categorical_columns)
print(len(categorical_columns))


# In[ ]:


#categorical columns
num_cols = [col for col in train_dataset.columns if col not in categorical_columns + ["target"]]

print(f'Total number of features: {(train_dataset.shape[1])-1}')
print(f'Total number of categorical features: {len(categorical_cols)}')
print(f'Total number of numerical features: {len(num_cols)}')


# In[ ]:


null_cols = train_dataset.columns[train_dataset.isna().any()].tolist()
num_null_columns = set(null_cols) - set(categorical_cols)
cat_null_cols = set(null_cols) - set(num_cols)
print(len(null_cols)),print(len(num_null_columns)), print(len(cat_null_cols))


# In[ ]:


for col in list(num_null_columns):
    train_dataset[col] = train_dataset[col].fillna(train_dataset[col].median())


# In[ ]:


for col2 in list(cat_null_cols):
    train_dataset[col2] = train_dataset[col2].astype('category').cat.add_categories('unknown')
    train_dataset[col2] = train_dataset[col2].fillna('unknown')


# In[ ]:


train_dataset.columns.isna().any()


# In[ ]:


null_cols = test_dataset.columns[test_dataset.isna().any()].tolist()
num_null_columns = set(null_cols) - set(categorical_cols)
cat_null_cols = set(null_cols) - set(num_cols)
print(len(null_cols)),print(len(num_null_columns)), print(len(cat_null_cols))


# In[ ]:


for column in list(num_null_columns):
    test_dataset[column] = test_dataset[column].fillna(train_dataset[col].median())


# In[ ]:


for column2 in list(cat_null_cols):
    test_dataset[column2] = test_dataset[column2].astype('category').cat.add_categories('unknown')
    test_dataset[column2] =  test_dataset[column2].fillna('unknown')


# In[ ]:


train_dataset.columns.isna().any()


# In[ ]:


print(test_dataset.shape)
print(train_dataset.shape)


# In[ ]:


# train_dataset_without_target = train_dataset.drop(["target"],axis=1)

# cor_matrix = train_dataset_without_target.corr()
# col_core = set()

# for i in range(len(cor_matrix.columns)):
#     for j in range(i):
#         if(cor_matrix.iloc[i, j] > 0.9):
#             col_name = cor_matrix.columns[i]
#             col_core.add(col_name)
# col_core


# In[ ]:


# train_dataset = train_dataset.drop(col_core, axis=1)
# test_dataset = test_dataset.drop(col_core, axis=1)


# In[ ]:


# print(test_dataset.shape)
# print(train_dataset.shape)


# In[ ]:


y = train_dataset['target']
train_dataset = train_dataset.drop(['target'],1)


# In[ ]:


print(f"X shape is = {train_dataset.shape}" )
print(f"Y shape is = {y.shape}" )


# In[ ]:


print(categorical_columns)


# In[ ]:


#encoding ctaegorical columns and reindex test data set's columns with train datas columns
train_dataset = pd.get_dummies(train_dataset, columns = categorical_columns)
test_dataset = pd.get_dummies(test_dataset, columns = categorical_columns)

test_dataset = test_dataset.reindex(columns = train_dataset.columns, fill_value=0)


# In[ ]:


# for col in categorical_cols:
#     test_dataset[col] =test_dataset[col].astype(str)


# In[ ]:


test_dataset.shape


# In[ ]:


train_dataset.dtypes.value_counts()


# In[ ]:


# import xgboost as xgb
# selector_model = xgb.XGBClassifier()
# selector_model.fit(train_dataset, y)


# In[ ]:


# k = 150
# top_k_features = selector_model.feature_importances_.argsort()[-k:]


# In[ ]:


# feature_importances = selector_model.feature_importances_
feat_imp_df = pd.read_csv('/kaggle/input/feature-importances/feature_importances.csv')


# In[ ]:


trueeee = pd.read_csv('/kaggle/input/amex-default-prediction/train_labels.csv')
trueeee.head()


# In[ ]:


feat_imp_df.head(5)


# In[ ]:


# feat_imp_df = pd.DataFrame(columns = ['feature','imp'])
# feat_imp_df['feature'] = (train_dataset.columns)
# feat_imp_df['imp'] = feature_importances
# feat_imp_df


# In[ ]:


# feat_imp_df.to_csv('feature_importances.csv', index=False)


# In[ ]:


# # for col,score in zip(train_dataset.columns,selector_model.feature_importances_):
# #     print(col,score)

# col_id_map = {}
# for col_id, col_name in enumerate(train_dataset.columns):
#     col_id_map[col_id] = col_name
# print(col_id_map)

# top_features_names= [col_id_map[col_id] for col_id in top_k_features[:100]]


# In[ ]:


# feat_imp_df = feat_imp_df.sort_values(by=['imp'], ascending=False)
# feat_imp_df.head()


# In[ ]:


number_of_features = 206


# In[ ]:


top_features_names = feat_imp_df['feature'].values[:number_of_features]


# In[ ]:


# ['R_8', 'R_16', 'R_15', 'B_33', 'R_4', 'S_22', 'D_78', 'D_117_-1.0', 'D_125', 'S_13', 'D_104', 'D_83', 'R_10', 'R_24', 'D_86', 'D_107', 'R_20', 'D_127', 'D_80', 'D_139', 'R_14', 'S_27', 'D_141', 'D_69', 'D_113', 'D_68_3.0', 'D_96', 'R_17', 'B_15', 'B_38_7.0', 'D_102', 'D_105', 'B_30_2.0', 'B_41', 'D_91', 'S_6', 'R_6', 'D_129', 'D_75', 'D_145', 'D_58', 'D_72', 'D_115', 'D_144', 'D_61', 'D_70', 'D_65', 'S_12', 'D_74', 'D_55', 'D_128', 'B_28', 'D_117_3.0', 'D_103', 'D_124', 'B_40', 'B_12', 'D_140', 'S_25', 'B_8', 'B_25', 'B_21', 'D_117_6.0', 'R_11', 'B_13', 'D_68_2.0', 'B_24', 'S_16', 'D_118', 'D_119', 'R_12', 'B_6', 'B_19', 'D_117_2.0', 'D_52', 'D_143', 'S_15', 'D_122', 'D_71', 'P_4', 'D_62', 'B_16', 'S_11', 'D_60', 'B_36', 'S_5', 'S_8', 'R_7', 'D_53', 'B_14', 'D_121', 'S_9', 'B_17', 'D_117_1.0', 'B_20', 'D_59', 'D_117_5.0', 'S_7', 'B_26', 'D_82']


# In[ ]:


# train_dataset = train_dataset[top_features_names]
# test_dataset = test_dataset[top_features_names]


# In[ ]:


print(test_dataset.shape)
print(train_dataset.shape)


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split( train_dataset, y, test_size=0.3, random_state=68)


# In[ ]:


test_dataset_v1 = test_dataset


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(test_dataset_v1.shape)


# In[ ]:


y_train.value_counts()


# In[ ]:


y_train.value_counts()[0]


# In[ ]:


scale_pos_weight = (y_train.value_counts()[0])/(y_train.value_counts()[1])
scale_pos_weight_full = (y.value_counts()[0])/(y.value_counts()[1])


# In[ ]:


scale_pos_weight


# In[ ]:


X_train.isna().any()


# In[ ]:


# from imblearn.over_sampling import SMOTE
# oversample = SMOTE()
# X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[ ]:


#Oversampling the minority class to solve class imbalance
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
train_dataset, y = oversample.fit_resample(train_dataset, y)


# ## Model Training

# In[ ]:


#SVM classifier - Linear
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1)).fit(train_dataset)
train_dataset = scaler.transform(train_dataset)
test_dataset_v1 = scaler.transform(test_dataset_v1)

from sklearn.svm import LinearSVC

clf = LinearSVC(random_state=0, tol=1e-5)

clf.fit(train_dataset, y.ravel())

predictions_svc = clf._predict_proba_lr(test_dataset_v1)

predictions_svc = predictions_svc[:,1]
sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': sample_dataset.customer_ID, 'prediction': predictions_svc})
output.to_csv('Submission SVM full.csv', index=False)


# In[ ]:


#feature selection to reduce dimensionality for knn
from sklearn.feature_selection import SelectKBest, mutual_info_classif

selector = SelectKBest(mutual_info_classif, k=20)
selector.fit(train_dataset, y)
mask = selector.get_support()
new_features = train_dataset.columns[mask]
new_features


# In[ ]:


train_dataset = selector.transform(train_dataset)
test_dataset_v1 = selector.transform(test_dataset_v1)


# In[ ]:


train_dataset.shape, test_dataset_v1.shape


# In[ ]:


#Knn classifier

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)

knn_clf.fit(train_dataset, y)
predictions_knn = knn_clf.predict_proba(test_dataset_v1)

predictions_knn = predictions_knn[:,1]
sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': sample_dataset.customer_ID, 'prediction': predictions_knn})
output.to_csv('Submission KNN full.csv', index=False)


# In[ ]:


import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


final_model = xgb.XGBClassifier(n_estimators = 500, 
                                objective = 'reg:logistic', 
                                seed = 69, 
#                                 scale_pos_weight = scale_pos_weight,
                                colsample_bytree=0.8,
#                                 min_child_weight=3,
                                max_depth = 5,
                                subsample = 0.8,
                               learning_rate = 0.1)


# In[ ]:


# final_model = KNeighborsClassifier(n_neighbors=5)


# In[ ]:


# final_model.fit(X_train, y_train)


# In[ ]:


final_model.fit(train_dataset, y)


# In[ ]:


# p = final_model.predict_proba(X_test)


# In[ ]:


## Model Evaluation


# In[ ]:


def amex_metric(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:

    def top_four_percent_captured(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        four_pct_cutoff = int(0.04 * df['weight'].sum())
        df['weight_cumsum'] = df['weight'].cumsum()
        df_cutoff = df.loc[df['weight_cumsum'] <= four_pct_cutoff]
        return (df_cutoff['target'] == 1).sum() / (df['target'] == 1).sum()
        
    def weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        df = (pd.concat([y_true, y_pred], axis='columns')
              .sort_values('prediction', ascending=False))
        df['weight'] = df['target'].apply(lambda x: 20 if x==0 else 1)
        df['random'] = (df['weight'] / df['weight'].sum()).cumsum()
        total_pos = (df['target'] * df['weight']).sum()
        df['cum_pos_found'] = (df['target'] * df['weight']).cumsum()
        df['lorentz'] = df['cum_pos_found'] / total_pos
        df['gini'] = (df['lorentz'] - df['random']) * df['weight']
        return df['gini'].sum()

    def normalized_weighted_gini(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        y_true_pred = y_true.rename(columns={'target': 'prediction'})
        return weighted_gini(y_true, y_pred) / weighted_gini(y_true, y_true_pred)

    g = normalized_weighted_gini(y_true, y_pred)
    d = top_four_percent_captured(y_true, y_pred)

    return 0.5 * (g + d)


# In[ ]:



p_v1 = p[:,1].reshape(-1,1)

y_pred = pd.DataFrame(p_v1, columns = ['prediction'])
# y_pred.head()

y_true = y_test.to_frame(name = 'target')
y_true = y_true.reset_index(drop=True)
# y_true.head()

from sklearn.metrics import roc_curve, roc_auc_score
roc_auc_score(y_test, p_v1)

amex_metric(y_true, y_pred)


# 0.7634980539964447
# 0.7637116591259403
# 0.7712345331426667 - best 150

# 0.9485239523284494
# 0.956661835314092

# In[ ]:


#plotting roc curve
def plot_roc(y_true, y_score, label_name, ax):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.set_title(
        f"{label_name}: AUC = {roc_auc_score(y_true, y_score):.4f}"
    )


fig, ax = plt.subplots(1, 1, figsize=(10, 8))
plot_roc(
    y_test, 
    p_v1, 
    'y',
    ax=ax
)


# In[ ]:


predictions = final_model.predict_proba(test_dataset_v1)
predictions

predictions = predictions[:,1]

sample_dataset = pd.read_csv('/kaggle/input/amex-default-prediction/sample_submission.csv')
output = pd.DataFrame({'customer_ID': sample_dataset.customer_ID, 'prediction': predictions})
output.to_csv('submission_v5.csv', index=False)


# In[ ]:





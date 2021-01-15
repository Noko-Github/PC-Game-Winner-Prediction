import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import optuna

data  = pd.read_csv('')

target = 'blueWins'
X = data[data.columns[data.columns != target]]
y = data.loc[:, target

# 目的関数の作成
def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    dtrain = lgb.Dataset(X_train, label=y_train)
    deval = lgb.Dataset(X_test, y_test, reference=dtrain)

    param = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('bagging_freq', 5, 100),
    }

    gbm = lgb.train(param, dtrain, valid_sets=deval, early_stopping_rounds=100)
    preds = gbm.predict(X_test)
    pred_labels = np.rint(preds)

    accuracy = accuracy_score(y_test, pred_labels)
    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
dtrain = lgb.Dataset(X, label=y)
model = lgb.train(study.best_params, dtrain)
test = pd.read_csv('/content/drive/MyDrive/削除可能/test/test.csv')
predicted = model.predict(test)
pred_labels = np.rint(predicted)

submission = pd.read_csv('', header=None)
submission.iloc[:, 1] =  list(map(int, pred_labels))
submission.to_csv('./submission.csv', header=False, index=False
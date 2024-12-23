import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("data/combined_train.csv")
test_data = pd.read_csv("data/test.csv")

train_data.drop(columns=['date'], inplace=True)
test_data.drop(columns=['date'], inplace=True)

categorical_columns = [
    'product_type', 'city', 'state', 'type.x', 'type.y',
    'locale', 'locale_name', 'description', 'transferred', 'day_of_week'
]

for col in categorical_columns:
    train_data = pd.get_dummies(train_data, columns=[col], drop_first=True)
    test_data = pd.get_dummies(test_data, columns=[col], drop_first=True)

train_columns = train_data.columns
test_data = test_data.reindex(columns=train_columns, fill_value=0)

if 'sales' in test_data.columns:
    test_data.drop(columns=['sales'], inplace=True)

selected_features = [
    'store_id',
    'onpromotion',
    'cluster',
    'oil_price',
    'year',
    'month',
]

selected_features = [col for col in selected_features if col in train_data.columns]

y = np.log1p(train_data["sales"])
X = train_data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

def rmsle(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.clip(preds, 0, None)
    return 'rmsle', np.sqrt(mean_squared_error(np.log1p(labels), np.log1p(preds)))

# Parameters
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'min_child_weight': 3,
    'learning_rate': 0.005,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,
    'alpha': 1,
    'random_state': 42
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    feval=rmsle,
    verbose_eval=False
)

y_pred = model.predict(dtest)
y_pred_original_scale = np.expm1(y_pred)

rmsle_score = np.sqrt(mean_squared_error(np.log1p(np.expm1(y_test)), np.log1p(y_pred_original_scale)))
print(f"RMSLE: {rmsle_score}")

test_dmatrix = xgb.DMatrix(test_data[selected_features])
test_predictions_log_scale = model.predict(test_dmatrix)
test_predictions = np.expm1(test_predictions_log_scale)

submission = pd.DataFrame({
    "id": test_data["id"],
    "sales": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

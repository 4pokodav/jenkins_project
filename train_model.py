import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import dump

df = pd.read_csv('data/processed.csv')
X = df.drop('price', axis=1)
y = df['price']

alphas = [0.01, 0.1, 1, 10, 100]

models = {
    'Ridge': Ridge(),
    'Lasso': Lasso(max_iter=10000)
}

best_model = None
best_score = float('inf')
best_name = None
best_estimator = None

mlflow.set_experiment("HousePriceRegression")

with mlflow.start_run():
    for name, model in models.items():
        grid = GridSearchCV(model, {'alpha': alphas}, cv=5, scoring='neg_mean_squared_error')
        grid.fit(X, y)

        mse = -grid.best_score_
        mlflow.log_param(f'{name}_best_alpha', grid.best_params_['alpha'])
        mlflow.log_metric(f'{name}_mse', mse)

        if mse < best_score:
            best_score = mse
            best_model = model
            best_estimator = grid.best_estimator_
            best_name = name

    mlflow.sklearn.log_model(best_estimator, "best_model")
    dump(best_estimator, 'model.pkl')

    mlflow.log_param("selected_model", best_name)
    mlflow.log_metric("best_mse", best_score)

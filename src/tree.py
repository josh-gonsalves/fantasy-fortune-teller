import polars as pl
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

"""
    train_tree_gridsearch():
        Train a DecisionTreeRegressor using GridSearchCV to find the best hyperparameters.
"""
def train_tree_gridsearch(df: pl.DataFrame):
    
    # Set target and drop non-useful features
    target = "fantasy_points_ppr"
    drop_cols = ["season", target]

    # Creates a list of columns including all features we care about
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Create X and y, use ravel() to avoid copying all the data (in the case of flatten())
    X = df.select(feature_cols).to_numpy()
    y = df.select(target).to_numpy().ravel()

    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set hyperparameter grid
    param_grid = {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_leaf_nodes': [5, 10, 15, None]
    }

    # GridSearchCV
    grid_search = GridSearchCV(
                DecisionTreeRegressor(random_state=42),
                param_grid,
                cv=3,
                verbose=1,
    )

    grid_search.fit(X_train, y_train)
    
    print("Best hyperparameters:", grid_search.best_params_)

    # Eval
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Decision Tree RMSE: {rmse:.3f}")
    print(f"Decision Tree R2:   {r2:.3f}")
    print(f"Decision Tree MAE:  {mae}")

    # Plot
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel("Actual Fantasy Points (PPR)")
    plt.ylabel("Predicted Fantasy Points (PPR)")
    plt.title("Predicted vs Actual Fantasy Points (Tree)")
    plt.show()

    idx = feature_cols.index("ppr_avg_8")
    tree_regplot(X_test, y_test, best_model, idx, "ppr_avg_8")
    
    return best_model, X_test, y_test, y_pred


def tree_regplot(X, y, model, feature_idx, feature_name, grid_points=200):
    # Grid over chosen feature
    x_grid = np.linspace(X[:, feature_idx].min(),X[:, feature_idx].max(),grid_points)

    # Hold other features fixed at mean
    X_curve = np.tile(X.mean(axis=0), (len(x_grid), 1))
    X_curve[:, feature_idx] = x_grid

    y_curve = model.predict(X_curve)

    # Plot singular feature vs predicted
    plt.figure(figsize=(7,5))
    plt.scatter(X[:, feature_idx], y, alpha=0.2)
    plt.plot(x_grid, y_curve, color='red', lw=2)

    plt.xlabel(feature_name)
    plt.ylabel("Fantasy Points (PPR)")
    plt.title(f"Decision Tree fit vs {feature_name}")
    plt.show()
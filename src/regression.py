import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

"""
    train_regression(): takes in a dataframe and trains four regressors
    - Linear regression
    - Logarithmic-ish regression
    - Quadratic Regression
    - Polynomial Regression
"""
def train_regression(df: pl.DataFrame):

    # Set target and drop non-useful features
    target = "fantasy_points_ppr"
    drop_cols = ["season", target]

    # Creates a list of columns including all features we care about
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Create X and y, use ravel() to avoid copying all the data (in the case of flatten())
    X = df.select(feature_cols).to_numpy()
    y = df.select(target).to_numpy().ravel()

    # split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test/train splitting taking time into account
    # Use earlier seasons for training, last one for testing
    """
    train_mask = df["season"] < df["season"].max()
    X_train = X[train_mask.to_numpy()]
    y_train = y[train_mask.to_numpy()]
    X_test  = X[~train_mask.to_numpy()]
    y_test  = y[~train_mask.to_numpy()]
    """

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # -----------------------------------------------------
    # Linear Regression
    # -----------------------------------------------------

    # Fit
    model = LinearRegression()
    model.fit(X_train, y_train)
    # Eval
    y_pred = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Linear RMSE: {rmse:.3f}")
    print(f"Linear R2:   {r2:.3f}")
    print(f"MAE:         {mae}")

    # -----------------------------------------------------
    # Logarithmic(ish) Regression
    # Here I tried to fit to a log function by taking a log of the data
    # -----------------------------------------------------

    # Shift y values because any <=0 will give NaN
    min_value = y_train.min()
    y_shift = y_train + abs(min_value) + 1
    # Fit
    log_model = LinearRegression()
    log_model.fit(X_train, np.log(y_shift))
    # Eval
    y_pred_log = log_model.predict(X_test)
    y_pred_log = np.exp(y_pred_log) - (abs(min_value) + 1)
    rmse = root_mean_squared_error(y_test, y_pred_log)
    r2 = r2_score(y_test, y_pred_log)
    mae = mean_absolute_error(y_test, y_pred_log)

    print(f"Logarithmic RMSE: {rmse:.3f}")
    print(f"Logarithmic R2:   {r2:.3f}")
    print(f"MAE:              {mae}")


    # -----------------------------------------------------
    # Polynomial Regression
    # -----------------------------------------------------

    # Reshape data for polynomial regressions
    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X_train)
    quad_test = quadratic.transform(X_test)
    X_cubic = cubic.fit_transform(X_train)
    cubic_test = cubic.transform(X_test)

    # Fit
    quad_model = LinearRegression().fit(X_quad, y_train)
    quad_pred = quad_model.predict(quad_test)
    cubic_model = LinearRegression().fit(X_cubic, y_train)
    cubic_pred = cubic_model.predict(cubic_test)

    # Eval
    rmse_quad = root_mean_squared_error(y_test, quad_pred)
    r2_quad = r2_score(y_test, quad_pred)
    mae_quad = mean_absolute_error(y_test, quad_pred)
    rmse_cubic = root_mean_squared_error(y_test, cubic_pred)
    r2_cubic = r2_score(y_test, cubic_pred)
    mae_cubic = mean_absolute_error(y_test, cubic_pred)

    print(f"Quadratic RMSE: {rmse_quad:.3f}")
    print(f"Quadratic R2:   {r2_quad:.3f}")
    print(f"MAE:            {mae_quad}")
    print(f"Cubic RMSE: {rmse_cubic:.3f}")
    print(f"Cubic R2:   {r2_cubic:.3f}")
    print(f"MAE:        {mae_cubic}")
    
    # Plotting
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.4)
    plt.xlabel("Actual Fantasy Points (PPR)")
    plt.ylabel("Predicted Fantasy Points (PPR)")
    plt.title("Predicted vs Actual Fantasy Points (Lin Reg)")
    plt.show()

    plt.figure()
    plt.scatter(y_test, y_pred_log, alpha=0.4)
    plt.xlabel("Actual Fantasy Points (PPR)")
    plt.ylabel("Predicted Fantasy Points (PPR)")
    plt.title("Predicted vs Actual Fantasy Points (Log Reg)")
    plt.show()

    plt.figure()
    plt.scatter(y_test, quad_pred, alpha=0.4)
    plt.xlabel("Actual Fantasy Points (PPR)")
    plt.ylabel("Predicted Fantasy Points (PPR)")
    plt.title("Predicted vs Actual Fantasy Points (Quad Reg)")
    plt.show()

    plt.figure()
    plt.scatter(y_test, cubic_pred, alpha=0.4)
    plt.xlabel("Actual Fantasy Points (PPR)")
    plt.ylabel("Predicted Fantasy Points (PPR)")
    plt.title("Predicted vs Actual Fantasy Points (Cubic Reg)")
    plt.show()

    # Build a plot showing the regression fit for a single parameter
    feature_name = "ppr_avg_8"
    idx = feature_cols.index(feature_name)

    # Create a grid of just ppr_avg_8 in order to feed into the model
    x_grid = np.linspace(X_train[:, idx].min(), X_train[:, idx].max(), 200)
    X_curve = np.tile(X_train.mean(axis=0), (len(x_grid), 1))
    X_curve[:, idx] = x_grid
    # Predict using artificial data
    y_curve_lin = model.predict(X_curve)
    y_curve_log = np.expm1(log_model.predict(X_curve))
    X_curve_quad = quadratic.transform(X_curve)
    y_curve_quad = quad_model.predict(X_curve_quad)
    X_curve_cubic = cubic.transform(X_curve)
    y_curve_cubic = cubic_model.predict(X_curve_cubic)

    plt.figure(figsize=(8,6))
    # Only plot ppr_avg_8 data
    plt.scatter(X_train[:, idx], y_train, alpha=0.2, label="Training data")

    # Plot each model's curve
    plt.plot(x_grid, y_curve_lin, label="Linear", lw=2)
    plt.plot(x_grid, y_curve_log, label="Log", lw=2)
    plt.plot(x_grid, y_curve_quad, label="Quadratic", lw=2)
    plt.plot(x_grid, y_curve_cubic, label="Cubic", lw=2)

    plt.xlabel(feature_name)
    plt.ylabel("Fantasy Points (PPR)")
    plt.title(f"Model Fits vs {feature_name}")
    plt.legend()
    plt.show()

    return model, scaler, feature_cols








import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gboost_module import xgbmodule
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class ForecastModule(torch.nn.Module):
    def __init__(
        self,
        n_features,
        xcols=None,
        params=None,
        use_batch_norm=True,
        datetime_feature_names=None,
    ):
        super(ForecastModule, self).__init__()
        if params is None:
            params = {}
        if xcols is None:
            xcols = []
        if datetime_feature_names is None:
            datetime_feature_names = [
                "split_1",
                "split_2",
                "split_3",
                "split_4",
                "minute",
                "hour",
                "month",
                "day",
                "weekday",
            ] + xcols
        self.seasonality = xgbmodule.XGBModule(
            n_features, len(datetime_feature_names), 1, params=params
        )
        self.xcols = xcols
        self.datetime_feature_names = datetime_feature_names
        self.use_batch_norm = use_batch_norm
        self.trend = torch.nn.Linear(1, 1)
        if use_batch_norm:
            self.bn = nn.LazyBatchNorm1d()
        else:
            self.bn = None
        self.initialized = False

    def initialize(self, df):
        X = df[["year"]].copy()
        y = df[["y"]].copy()

        if X["year"].std() == 0:
            with torch.no_grad():
                self.trend.weight.copy_(torch.Tensor([[0.0]]))
                self.trend.bias.copy_(torch.Tensor(y.mean().values))
            self.initialized = True
            return

        X["intercept"] = 1
        X["year"] = (X["year"] - X["year"].mean()) / X["year"].std()
        ests = lstsq(X[["intercept", "year"]], y)[0]

        with torch.no_grad():
            self.trend.weight.copy_(torch.Tensor(ests[1:, :]))
            self.trend.bias.copy_(torch.Tensor(ests[0]))

        self.initialized = True

    def forward(self, df):
        if not self.initialized:
            self.initialize(df)

        datetime_features = np.array(df[self.datetime_feature_names])
        trend = torch.Tensor(df["year"].values).reshape([-1, 1])

        self.minput = datetime_features
        if self.use_batch_norm and self.bn is not None:
            trend_output = self.trend(self.bn(trend))
        else:
            trend_output = self.trend(trend)
        output = trend_output + self.seasonality(datetime_features)
        return output

    def gb_step(self):
        self.seasonality.gb_step(self.minput)


def recursive_split(df, column, depth):
    """Adds columns of zeros and ones to substitute changepoints"""
    df = df.sort_values(by=column).reset_index(drop=True)
    binary_cols = pd.DataFrame(index=df.index)

    current_groups = [df.index]
    for level in range(1, depth + 1):
        next_groups = []
        for group in current_groups:
            mid = len(group) // 2
            binary_cols.loc[group[:mid], f"split_{level}"] = 0
            binary_cols.loc[group[mid:], f"split_{level}"] = 1
            next_groups.extend([group[:mid], group[mid:]])
        current_groups = next_groups

    binary_cols = binary_cols.astype("int64")
    return pd.concat([df, binary_cols], axis=1)


class Forecast(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        nrounds=3000,
        xcols=None,
        params=None,
        split_depth=4,
        date_features=None,
        use_batch_norm=True,
        learning_rate=0.1,
        datetime_feature_names=None,
    ):
        if params is None:
            params = {}
        if xcols is None:
            xcols = []
        if date_features is None:
            date_features = ["month", "year", "day", "weekday", "hour", "minute"]
        if datetime_feature_names is None:
            datetime_feature_names = (
                [f"split_{i}" for i in range(1, split_depth + 1)]
                + date_features
                + xcols
            )
        self.nrounds = nrounds
        self.xcols = xcols
        self.params = params
        self.split_depth = split_depth
        self.date_features = date_features
        self.use_batch_norm = use_batch_norm
        self.learning_rate = learning_rate
        self.datetime_feature_names = datetime_feature_names
        self.model_ = None
        self.losses_ = []

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y
        df = self._prepare_dataframe(df)
        df = recursive_split(df, "ds", self.split_depth)
        self.model_ = ForecastModule(
            df.shape[0],
            xcols=self.xcols,
            params=self.params,
            use_batch_norm=self.use_batch_norm,
            datetime_feature_names=self.datetime_feature_names,
        )
        self.model_.train()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        mse = torch.nn.MSELoss()

        for _ in range(self.nrounds):
            optimizer.zero_grad()
            preds = self.model_(df)
            loss = mse(preds.flatten(), torch.Tensor(df["y"].values).flatten())
            loss.backward(create_graph=True)
            self.losses_.append(loss.detach().item())
            self.model_.gb_step()
            optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        df = X.copy()
        df = self._prepare_dataframe(df)
        for j in range(1, self.split_depth + 1):
            df[f"split_{j}"] = 1.0
        preds = self.model_(df).detach().numpy()
        return preds.flatten()

    def _prepare_dataframe(self, df):
        df["ds"] = pd.to_datetime(df["ds"])
        for feature in self.date_features:
            if feature == "month":
                df["month"] = df["ds"].dt.month
            elif feature == "year":
                df["year"] = df["ds"].dt.year
            elif feature == "day":
                df["day"] = df["ds"].dt.day
            elif feature == "weekday":
                df["weekday"] = df["ds"].dt.weekday
            elif feature == "hour":
                df["hour"] = df["ds"].dt.hour
            elif feature == "minute":
                df["minute"] = df["ds"].dt.minute
        return df

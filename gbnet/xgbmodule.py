from typing import Union
import warnings
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch import nn


class XGBModule(nn.Module):
    def __init__(
        self, batch_size, input_dim, output_dim, params={}, min_hess=0, batch_mode=False
    ):
        super(XGBModule, self).__init__()
        self.batch_mode = batch_mode
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.params = params
        self.params["objective"] = "reg:squarederror"
        self.params["base_score"] = 0
        self.n_completed_boost_rounds = 0
        self.min_hess = min_hess

        init_matrix = np.zeros([batch_size, input_dim])
        self.bst = xgb.train(
            self.params,
            xgb.DMatrix(init_matrix, label=np.zeros(batch_size * output_dim)),
            num_boost_round=0,
        )
        self.n_completed_boost_rounds = 0
        self.dtrain = None
        self.training_n = None

        self.FX = nn.Parameter(
            torch.tensor(
                np.zeros([batch_size, output_dim]),
                dtype=torch.float,
            )
        )

    def _check_training_data(self):
        if self.dtrain.get_weight().shape[0] > 0:
            warnings.warn(
                "Weights will not work properly when defined as part of the input DMatrix. Weights should be defined in the loss."
            )

    def _input_checking_setting(
        self, input_data: Union[xgb.DMatrix, pd.DataFrame, np.ndarray]
    ):
        assert isinstance(input_data, (xgb.DMatrix, pd.DataFrame, np.ndarray))
        if self.training:
            if self.dtrain is None or self.batch_mode:
                if isinstance(input_data, xgb.DMatrix):
                    input_data.set_label(np.zeros(self.batch_size * self.output_dim))
                    self.dtrain = input_data
                    self.training_n = input_data.num_row()
                    self._check_training_data()
                else:
                    self.dtrain = xgb.DMatrix(
                        input_data, label=np.zeros(self.batch_size * self.output_dim)
                    )
                    self.training_n = input_data.shape[0]
            compare_n = (
                input_data.num_row()
                if isinstance(input_data, xgb.DMatrix)
                else input_data.shape[0]
            )
            assert (
                (compare_n == self.training_n) or self.batch_mode
            ), "Changing datasets while training is not currently supported. If trying to make predictions, set Module to eval mode via `Module.eval()`"
            return self.dtrain
        return (
            input_data
            if isinstance(input_data, xgb.DMatrix)
            else xgb.DMatrix(input_data)
        )

    def forward(self, input_data, return_tensor: bool = True):
        input_data = self._input_checking_setting(input_data)
        preds = self.bst.predict(input_data)

        if self.training:
            FX_detach = self.FX.detach()
            FX_detach.copy_(
                torch.tensor(
                    preds.reshape([self.batch_size, self.output_dim]), dtype=torch.float
                )
            )

        if return_tensor:
            if self.training:
                return self.FX
            else:
                return torch.tensor(
                    preds.reshape([-1, self.output_dim]), dtype=torch.float
                )
        return preds

    def gb_step(self):
        grad, hess = self._get_grad_hess_FX(self.FX)
        self._gb_step_grad_hess(grad, hess)

    def _gb_step_grad_hess(self, grad, hess):
        obj = XGBObj(grad, hess)
        g, h = obj(np.zeros([self.batch_size, self.output_dim]), None)

        if xgb.__version__ <= "2.0.3":
            self.bst.boost(
                self.dtrain,
                g,
                h,
            )
        else:
            self.bst.boost(
                self.dtrain,
                self.n_completed_boost_rounds + 1,
                g,
                h,
            )
        self.n_completed_boost_rounds = self.n_completed_boost_rounds + 1

    def _get_grad_hess_FX(self, FX):
        grad = FX.grad * self.batch_size

        # parameters are independent row by row, so we can
        # at least calculate hessians column by column by
        # considering the sum of the gradient columns
        hesses = []
        for i in range(self.output_dim):
            hesses.append(
                torch.autograd.grad(grad[:, i].sum(), FX, retain_graph=True)[0][
                    :, i : (i + 1)
                ]
            )
        hess = torch.maximum(torch.cat(hesses, axis=1), torch.Tensor([self.min_hess]))
        return grad, hess


class XGBObj:
    def __init__(self, grad, hess):
        self.grad = grad
        self.hess = hess

    def __call__(self, preds, dtrain):
        if len(preds.shape) == 2:
            M = preds.shape[0]
            N = preds.shape[1]
        else:
            M = preds.shape[0]
            N = 1

        if xgb.__version__ >= "2.1.0":
            g = self.grad.detach().numpy().reshape([M, N])
            h = self.hess.detach().numpy().reshape([M, N])
        else:
            g = self.grad.detach().numpy().reshape([M * N, 1])
            h = self.hess.detach().numpy().reshape([M * N, 1])

        return g, h

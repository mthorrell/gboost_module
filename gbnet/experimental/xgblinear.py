import torch
from gbnet.xgbmodule import XGBModule
import numpy as np
import json


class GBLinear(torch.nn.Module):
    def __init__(self, n, in_dim, out_dim, params={}, min_hess=0):
        super(GBLinear, self).__init__()
        # base_score being 0 is required because there seems to be an initial
        # mean not tracked in the json output.
        params.update({"booster": "gblinear", "base_score": 0})

        self.in_dim = in_dim
        self.out_dim = out_dim

        # random initialization for self.linear should be OK
        self.linear = torch.nn.Linear(self.in_dim, self.out_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.gblinear = XGBModule(
            n, self.in_dim, self.out_dim, params, min_hess=min_hess, batch_mode=True
        )

    def forward(self, X):
        self.gblinear(X.detach().numpy())
        lin_out = self.linear(X)

        self.FX = lin_out
        if self.training:
            self.FX.retain_grad()
        return self.FX

    def gb_step(self):
        grad, hess = self.gblinear._get_grad_hess_FX(self.FX)
        self.gblinear._gb_step_grad_hess(grad, hess)

        b = self.gblinear.bst
        config = json.loads(b.get_dump(dump_format="json")[0])
        coef = np.array(config["weight"]).reshape((-1, self.out_dim)).T
        intercept = np.array(config["bias"])

        with torch.no_grad():
            self.linear.weight.data = torch.Tensor(coef)
            self.linear.bias.data = torch.Tensor(intercept)
        return grad, hess

import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, user_size: int, job_size: int, behavior_size: int, dim: int, da_size: int, device: str):
        super(Model, self).__init__()
        self.user_size = user_size
        self.job_size = job_size
        self.behavior_size = behavior_size
        self.dim = dim
        self.device = device
        self.U_true = nn.Embedding(job_size, dim)
        self.U = nn.Parameter(torch.empty(user_size, behavior_size-1, dim))
        self.V = nn.Embedding(job_size, dim)
        self.da_size = da_size
        if da_size > 0:
            self.da = nn.Embedding(self.da_size+1, dim)
            nn.init.xavier_normal_(self.da.weight.data)
        self.linear_target = nn.Linear(dim, dim)
        nn.init.xavier_normal_(self.U_true.weight.data)
        nn.init.xavier_normal_(self.U.data)
        nn.init.xavier_normal_(self.V.weight.data)
        nn.init.xavier_normal_(self.linear_target.weight.data)
        self.loss = nn.TripletMarginLoss(reduction='none')

    def forward(self, phase, users, pos_job_ids, behavior_ids, das, neg_job_id_lists):
        if phase == 0:
            return self._auxiliary_loss(users, pos_job_ids, behavior_ids, neg_job_id_lists)
        if phase == 1:
            return self._target_loss(users, pos_job_ids, das, neg_job_id_lists)
        raise Exception

    def predict(self, user, candidates):
        u = self.U_true(torch.tensor([user]).to(self.device)) # dim
        i = self.V(candidates) # candidate_size x dim
        u = u.expand(i.shape) # candidate_size x dim
        preds = -torch.norm(u-i, dim=-1)
        return preds

    def _auxiliary_loss(self, users, pos_job_ids, behavior_ids, neg_job_id_lists):
        u = self.U[users, behavior_ids] # batch x dim
        u += self.U_true(users)
        i = self.V(pos_job_ids)
        j = self.V(neg_job_id_lists)

        return self._calc_loss(u, i, j)

    def _target_loss(self, users, pos_job_ids, das, neg_job_id_lists):
        u = self._target_users(users, das) # batch x dim
        i = self.V(pos_job_ids)
        j = self.V(neg_job_id_lists)

        return self._calc_loss(u, i, j)

    def _target_users(self, users, das):
        u_v = self.U_true(users) + self.U[users][:, 0]
        u_f = self.U_true(users) + self.U[users][:, 1]
        u = u_v + u_f
        u = self.linear_target(u)
        if self.da_size > 0:
            das = torch.clamp(das, min=0, max=self.da_size)
            u += self.da(das)
        return u

    def _calc_loss(self, u, i, j, weights=1.0):
        loss = 0.0
        for idx in range(j.shape[1]):
            loss += (weights * self.loss(u, i, j[:, idx, :])).mean()
        return loss

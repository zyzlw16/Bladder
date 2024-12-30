import torch

# def cox_loss(y_true, y_pred):

#     time_value = torch.squeeze(torch.index_select(y_true, 1, torch.tensor([0]).cuda()))
#     event = torch.squeeze(torch.index_select(y_true, 1, torch.tensor([1]).cuda())).bool()
#     score = torch.squeeze(y_pred, dim=1)

#     ix = torch.where(event)

#     sel_mat = (time_value[ix[0]].unsqueeze(-1) <= time_value).float()

#     p_lik = torch.gather(score, 0, ix) - torch.log(torch.sum(sel_mat * torch.exp(score.t()), dim=-1))

#     loss = -torch.mean(p_lik)

#     return loss


def cox_loss(y_true, y_pred):
    time_value = y_true[:, 0]
    event = y_true[:, 1].bool()
    score = y_pred.squeeze()

    if not torch.any(event):
        return torch.tensor(1e-8, requires_grad=True)

    ix = torch.where(event)
    sel_mat = torch.gather(time_value, 0, ix[0]).unsqueeze(0).t() <= time_value

    p_lik = score - torch.log(torch.sum((sel_mat * torch.exp(score))+ 1e-8, dim=0))
    loss = -torch.mean(p_lik )

    return loss


def concordance_index(y_true, y_pred):
    time_value = y_true[:, 0]
    event = y_true[:, 1].bool()

    ix = torch.where((torch.unsqueeze(time_value, dim=-1) < time_value) & torch.unsqueeze(event, dim=-1))

    s1 = torch.index_select(y_pred, 0, ix[0])
    s2 = torch.index_select(y_pred, 0, ix[1])
    ci = torch.mean(torch.tensor(s1 < s2, dtype=torch.float32))

    return ci

def bootstrap_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.05):
    ci_values = []
    n_samples = y_true.shape[0]

    for _ in range(n_bootstrap):

        indices = torch.randint(0, n_samples, size=(n_samples,))
        y_true_sampled = y_true[indices]
        y_pred_sampled = y_pred[indices]


        ci = concordance_index(y_true_sampled, y_pred_sampled)
        ci_values.append(ci.item())


    lower_bound = torch.quantile(torch.tensor(ci_values), alpha / 2)
    upper_bound = torch.quantile(torch.tensor(ci_values), 1 - alpha / 2)

    return lower_bound, upper_bound
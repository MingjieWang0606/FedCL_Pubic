import torch
from cvxopt import matrix, spdiag, solvers
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
import math
from scipy.special import lambertw
from scipy.special import binom


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.5, batch_size=32, view_num=1, p=2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.targets =  torch.cat([torch.arange(batch_size) for i in range(view_num)], dim=0)
        self.eps = 1e-7

    def forward(self, inputs,targets = None):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        if targets == None:
            targets = self.targets
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = []
        for i in range(n):
            dist.append(inputs[i] - inputs)
        dist = torch.stack(dist)
        dist = torch.linalg.norm(dist,ord=self.p,dim=2)

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)+self.eps

def compute_weights(lossgrad, lamb):

    device = lossgrad.get_device()
    lossgrad = lossgrad.data.cpu().numpy()

    # Compute Optimal sample Weights
    aux = -(lossgrad**2+lamb)
    sz = len(lossgrad)
    P = 2*matrix(lamb*np.identity(sz))
    q = matrix(aux.astype(np.double))
    A = spdiag(matrix(-1.0, (1,sz)))
    b = matrix(0.0, (sz,1))
    Aeq = matrix(1.0, (1,sz))
    beq = matrix(1.0*sz)
    solvers.options['show_progress'] = False
    solvers.options['maxiters'] = 20
    solvers.options['abstol'] = 1e-4
    solvers.options['reltol'] = 1e-4
    solvers.options['feastol'] = 1e-4
    sol = solvers.qp(P, q, A, b, Aeq, beq)
    w = np.array(sol['x'])
    
    return torch.squeeze(torch.tensor(w, dtype=torch.float))

class LOWLoss(torch.nn.Module):
    def __init__(self, lamb=0.1):
        super(LOWLoss, self).__init__()
        self.lamb = lamb # higher lamb means more smoothness -> weights closer to 1
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')  # replace this with any loss with "reduction='none'"
    def forward(self, logits, target):
        # Compute loss gradient norm
        output_d = logits.detach()
        loss_d = torch.mean(self.loss(output_d.requires_grad_(True), target), dim=0)
        loss_d.backward(torch.ones_like(loss_d))
        lossgrad = torch.norm(output_d.grad, 2, 1)
        
        # Computed weighted loss
        weights = compute_weights(lossgrad, self.lamb)
        loss = self.loss(logits, target)
        loss = torch.mean(torch.mul(loss, weights), dim=0)
        
        return loss, weights
    
#epoch,self.local_epochs
def ft_Cam_1(output, target, alpha, schedule):   
    
    if schedule[0] > int(schedule[1]/2):
        loss = F.cross_entropy(output, target, reduction='none')
        'Sort the loss in descending order'
        loss_sorted, indices = torch.sort(loss, descending=True)

        top_k = round(alpha * target.size(0))   # Select top_K values for determining the hardness in mini-batch (alpha x batch_size)

        # Calculate the adaptive hardness threshold (thres as in Eq. 1 in the paper)
        a = 0.7
        b = 0.2
        #print(schedule)
        thres = a*(1-(schedule[0]/len(range(schedule[1])))) + b
        # print('thres', thres)
        # print('current_batch', batch_idx)
        # print('max_iteration', len(train_loader))

        # Select the hardness in each mini-batch based on the threshold (thres)
        hard_samples = loss_sorted[0:top_k]
        total_sum_hard_samples = sum(hard_samples)

        # Check whether total sum exceeds the threshold and update the loss accordingly (Eq. 2 in the paper)
        if total_sum_hard_samples > (thres * sum(loss_sorted)):
            output = output[indices, :]
            target = target[indices]
            top_k_output = output[0:top_k]
            tok_k_target = target[0:top_k]
            loss = F.cross_entropy(top_k_output, tok_k_target, reduction='mean')
        else:
            loss = F.cross_entropy(output, target, reduction='mean')
        return loss

    else:
        loss = F.cross_entropy(output, target, reduction='none')

        'Sort the loss in descending order'
        loss_sorted, indices = torch.sort(loss, descending=True)

        top_k = round(alpha * target.size(0))   # Select top_K values for determining the hardness in mini-batch (alpha x batch_size)

        # Calculate the adaptive hardness threshold (thres as in Eq. 1 in the paper)
        a = 0.7
        b = 0.2
        thres = a*(1-(schedule[0]/len(range(schedule[1])))) + b
        top_k2 = round(thres * top_k)   # Select hardness level again within top_K values (i.e., top-K' as described in paper)

#         print('thres=', thres)
#         print('top_k=', top_k)
#         print('top_k2'=', top_k2)
#         print('current_batch=', batch_idx)
#         print('max_iteration=', len(train_loader))

        # Select the hardness in each mini-batch based on top_k values
        hard_samples = loss_sorted[0:top_k]
        total_sum_hard_samples = sum(hard_samples)

        # Select the hardness within top_k values (i.e., top-k')
        hard_samples_k = hard_samples[0:top_k2]
        total_sum_hard_samples_k = sum(hard_samples_k)

        # Select top_k and k output and target values
        output = output[indices, :]
        target = target[indices]

        top_k_output = output[0:top_k]
        tok_k_target = target[0:top_k]

        k_output = top_k_output[0:top_k2]
        k_target = tok_k_target[0:top_k2]

        # Check whether total sum exceeds the threshold and update the loss accordingly (Eq. 3 in the paper)
        if total_sum_hard_samples_k > (thres * total_sum_hard_samples):
            loss = F.cross_entropy(k_output, k_target, reduction='mean')
            #print('K_update done')
        else:
            loss = F.cross_entropy(top_k_output, tok_k_target, reduction='mean')
            #print('top_K_update done')
        return loss
    
class LabelSmoothingCrossEntropyWithSuperKLDivLoss(nn.Module):
    def __init__(self, eps=0.01, reduction='mean', classes=10, rank=None, lam=0.5):
        super(LabelSmoothingCrossEntropyWithSuperKLDivLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.super_loss = SuperLoss(C=classes, rank=rank, lam=lam)
        self.rank = rank

    def forward(self, output, target):
        B, c = output.size()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        super_loss = self.super_loss(FocalLoss(Superloss = True)(log_preds, target))
        #super_loss = self.super_loss(F.nll_loss(log_preds, target, reduction='none'))
        # l_i = (-log_preds.sum(dim=-1)) * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction='none')
        # return self.super_loss(l_i)
        loss_cls = loss * self.eps / c + (1 - self.eps) * super_loss
        return loss_cls, torch.exp(super_loss)   
    
class LabelSmoothingCrossEntropyWithSuperFlLoss(nn.Module):
    def __init__(self, eps=0.01, reduction='mean', classes=10, rank=None, lam=0.5):
        super(LabelSmoothingCrossEntropyWithSuperFlLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.super_loss = SuperLoss(C=classes, rank=rank, lam=lam)
        self.rank = rank

    def forward(self, output, target):
        B, c = output.size()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        super_loss = self.super_loss(FocalLoss(Superloss = True)(log_preds, target))
        #super_loss = self.super_loss(F.nll_loss(log_preds, target, reduction='none'))
        # l_i = (-log_preds.sum(dim=-1)) * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction='none')
        # return self.super_loss(l_i)
        loss_cls = loss * self.eps / c + (1 - self.eps) * super_loss
        return loss_cls, torch.exp(super_loss)   
    
class LabelSmoothingCrossEntropyWithSuperCELoss(nn.Module):
    def __init__(self, eps=0.01, reduction='mean', classes=10, rank=None, lam=0.5):
        super(LabelSmoothingCrossEntropyWithSuperCELoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.super_loss = SuperLoss(C=classes, rank=rank,lam=lam)
        self.rank = rank

    def forward(self, output, target):
        B, c = output.size()
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        celoss = F.nll_loss(log_preds, target, reduction='none')
        super_loss, score_list,tau = self.super_loss(celoss)

        
        # l_i = (-log_preds.sum(dim=-1)) * self.eps / c + (1 - self.eps) * F.nll_loss(log_preds, target, reduction='none')
        # return self.super_loss(l_i)
        loss_cls = loss * self.eps / c + (1 - self.eps) * super_loss
        return loss_cls, torch.exp(super_loss) , score_list, celoss, tau
    
class SuperLoss(nn.Module):
    def __init__(self, C=10, lam=0.5, rank=None):
        super(SuperLoss, self).__init__()
        self.tau = torch.log(torch.FloatTensor([C]).to(rank))
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.rank = rank

    def forward(self, l_i):
        l_i_detach = l_i.detach()
        
        # self.tau = 0.9 * self.tau + 0.1 * l_i_detach
        sigma,y,self.tau = self.sigma(l_i_detach)
        loss = (l_i - self.tau) * sigma + self.lam * torch.log(sigma)**2

        loss_mean = loss.mean()
        return loss_mean, y,self.tau

    def sigma(self, l_i):
        x = -2 / torch.exp(torch.ones_like(l_i)).to(self.rank)
        cl_score = l_i - self.tau
        y_ = 0.5 * torch.max(x, cl_score / self.lam)
        y = y_.cpu().numpy()
        sigma = np.exp(-lambertw(y))   
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).to(self.rank)
        
        return sigma,cl_score,self.tau
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, Superloss = False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.Superloss = Superloss
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.Superloss:
            return focal_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()
        
def CL_User_Score(model_result, Algorithms, loss_fun, y, local_epoch, schedule):
    """
    Get Curriculum Learning Score and batch
    :param model_result:
    :param y: label
    :param Algorithms: base
    :param loss_fun :
    
    :return:
    :Curriculum_Learning_Score:
    :Curriculum_Learning_Loss:
    :Base_Loss:
    """
    
    if Algorithms == 'base':
        Base_Loss = loss_fun(model_result['output'], y) 
        Curriculum_Learning_Score = None
        
    elif Algorithms == 'LOW':
        Base_Loss, weights = LOWLoss()(model_result['logit'], y) 
        Curriculum_Learning_Score = weights
        
    elif Algorithms == 'ft_Cam':
        Base_Loss = ft_Cam_1(model_result['logit'], y, 0.1,schedule)
        Curriculum_Learning_Score = None
        
    elif Algorithms == 'SuperLoss_ce':    
        Base_Loss, Curriculum_Learning_Score, score_list,celoss, tau = LabelSmoothingCrossEntropyWithSuperCELoss(classes=10, rank=None, lam=1)(model_result['logit'], y) 

    elif Algorithms == 'SuperLoss_fl':    
        Base_Loss, Curriculum_Learning_Score = LabelSmoothingCrossEntropyWithSuperFlLoss(classes=10, rank=None, lam=0.1)(model_result['logit'], y) 
            
    elif Algorithms == 'FocalLoss':       
        Base_Loss = FocalLoss()(model_result['logit'], y) 
        Curriculum_Learning_Score = None

    else:
        Exception('Algorithms None')   

    return {'Curriculum_Learning_Score':Curriculum_Learning_Score, 
            'Loss':Base_Loss,
           'score_list':(score_list-torch.min(score_list)) / (torch.max(score_list) - torch.min(score_list)),
           'celoss':celoss,
           'TripletLoss_':0.01, #TripletLoss__,
            'tau':tau,
            'score_list_base':score_list
           }
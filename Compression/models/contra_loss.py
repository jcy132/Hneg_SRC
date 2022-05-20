from packaging import version
import torch
from torch import nn
class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class SRC_Loss(nn.Module):
    def __init__(self,opt,batch_size=1):
        super().__init__()
        self.opt = opt
        self.batch_size = batch_size
        self.mask_dtype = torch.bool


    def forward(self, feat_q, feat_k, epoch=1,max_epoch=1,only_weight=False):
        
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = self.batch_size
        feat_k = Normalize()(feat_k)
        feat_q = Normalize()(feat_q)

        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)

        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1))
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))
        
        gamma = self.opt.gamma_start
        gamma = gamma + (self.opt.gamma_min - gamma)*(epoch)/(max_epoch)
        
        weight_seed = spatial_k.clone().detach()
        diagonal = torch.eye(self.opt.n_patch, device=feat_k_v.device, dtype=self.mask_dtype)[None, :, :]
        
        weight_seed.masked_fill_(diagonal, -10.0)                                         
        weight_out = nn.Softmax(dim=2)(weight_seed.clone() / gamma).detach()
        wmax_out, _ = torch.max(weight_out, dim=2, keepdim=True)
        weight_out /= wmax_out

        if only_weight:
            return 0, 0, weight_out

        spatial_q = nn.Softmax(dim=1)(spatial_q)                    # symmetry dim=1 okay
        spatial_k = nn.Softmax(dim=1)(spatial_k).detach()

        loss_SRC = self.opt.lambda_SRC*self.get_jsd(spatial_q, spatial_k)
        if self.opt.lambda_SRC == 0:
            weight_out = None
        
        loss = loss_SRC, weight_out

        return loss

    def get_jsd(self, p1, p2):

        m = 0.5 * (p1 + p2)
        out = 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p1))
                     + nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p2)))
        return out
    
    
class PatchDCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.nce_T = 0.07
    def forward(self, feat_q, feat_k,weight=None):
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        batch_dim_for_bmm=1
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        if weight is not None:
            l_neg_curbatch *= weight

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        logits = (l_neg-l_pos)/self.opt.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v-v.detach())

        # for exp
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        CELoss_dummy = self.cross_entropy_loss(out_dummy, torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean()-1+CELoss_dummy.detach()

        return self.opt.lambda_dce*loss

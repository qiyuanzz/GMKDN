import torch
import torch.nn as nn



class MMD_Loss(nn.Module):
    """ source: https://github.com/facebookresearch/DomainBed/blob/main/domainbed/algorithms.py
    """
    def __init__(self, kernel_type = 'gaussian'):
        super().__init__()
        
        self.kernel_type = kernel_type
        
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                           1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K
    
    def gaussian_rbf_kernel(self, x, y, sigma_sqr=2., **kwargs):
        r"""
        Gaussian radial basis function (RBF) kernel.
        .. math::
            k(x, y) = \exp (\frac{||x-y||^2}{\sigma^2})
        """
        pairwise_distance_matrix = torch.sum((x[:, None, :] - y[None, :, :]) ** 2, dim=-1)
        K = torch.exp(-pairwise_distance_matrix / (1. * sigma_sqr))
        return K

    def forward(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy

        
        elif self.kernel_type == "mean_cov": # this is coral loss
            mean_x = x.mean(0, keepdim=True) 
            mean_y = y.mean(0, keepdim=True) 
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1) 
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1) 

            mean_diff = (mean_x - mean_y).pow(2).mean() 
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff
        else:
            raise NotImplementedError()
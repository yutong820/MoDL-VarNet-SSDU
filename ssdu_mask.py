import numpy as np
import utils
import torch

class ssdu_masks():
    """

    Parameters
    ----------
    rho: split ratio for training and loss mask. \ rho = |\Lambda|/|\Omega|
    small_acs_block: keeps a small acs region fully-sampled for training masks
    if there is no acs region, the small acs block should be set to zero
    input_data: input k-space, nrow x ncol x ncoil
    input_mask: input mask, nrow x ncol

    Gaussian_selection:
    -divides acquired points into two disjoint sets based on Gaussian  distribution
    -Gaussian selection function has the parameter 'std_scale' for the standard deviation of the distribution. We recommend to keep it as 2<=std_scale<=4.

    Uniform_selection: divides acquired points into two disjoint sets based on uniform distribution

    Returns
    ----------
    trn_mask: used in data consistency units of the unrolled network
    loss_mask: used to define the loss in k-space

    """

    def __init__(self, rho=0.4, small_acs_block=(4, 4)):
        self.rho = rho # segmentation ratio, Lambda-sampling points in undersampl mask; Omega-all k space sampling points
        self.small_acs_block = small_acs_block

    # def Gaussian_selection(self, input_data, input_mask, std_scale=4, num_iter=1):

    #     nrow, ncol = input_data.shape[0], input_data.shape[1]
    #     center_kx = int(utils.find_center_ind(input_data, axes=(1, 2)))
    #     print('center x:', center_kx)
    #     center_ky = int(utils.find_center_ind(input_data, axes=(0, 2)))
    #     print('center y:', center_ky)

    #     if num_iter == 0:
    #         print(f'\n Gaussian selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

    #     # keep unchoose ACS area
    #     temp_mask = np.copy(input_mask)
    #     temp_mask[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
    #     center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 0

    #     loss_mask = np.zeros_like(input_mask)
    #     count = 0

    #     while count <= int(np.ceil(np.sum(input_mask[:]) * self.rho)):

    #         indx = int(np.round(np.random.normal(loc=center_kx, scale=(nrow - 1) / std_scale)))
    #         indy = int(np.round(np.random.normal(loc=center_ky, scale=(ncol - 1) / std_scale)))

    #         if (0 <= indx < nrow and 0 <= indy < ncol and temp_mask[indx, indy] == 1 and loss_mask[indx, indy] != 1):
    #             loss_mask[indx, indy] = 1
    #             count = count + 1

    #     trn_mask = input_mask - loss_mask

    #     return trn_mask, loss_mask
    

    def Gaussian_selection(self, input_data, input_mask, std_scale=4.5):
        
        if isinstance(input_data, np.ndarray):
            input_data = torch.from_numpy(input_data).float()
        
        if isinstance(input_mask, np.ndarray):
            input_mask = torch.from_numpy(input_mask).float()

        nrow, ncol = input_data.shape[-2], input_data.shape[-1]
        center_kx, center_ky = nrow // 2, ncol // 2
        # center_kx = int(utils.find_center_ind(input_data.cpu().numpy(), axes=(1, 2)))
        # center_ky = int(utils.find_center_ind(input_data.cpu().numpy(), axes=(0, 2)))

        temp_mask = input_mask.clone()
        temp_mask[center_kx - self.small_acs_block[0] // 2:center_kx + self.small_acs_block[0] // 2,
                center_ky - self.small_acs_block[1] // 2:center_ky + self.small_acs_block[1] // 2] = 0

        loss_mask = torch.zeros_like(input_mask)
        target_count = int(torch.ceil(torch.sum(input_mask).float() * self.rho))

        indx = torch.round(torch.normal(mean=center_kx, std=(nrow - 1) / std_scale, size=(target_count,))).clamp(0, nrow-1).int()
        indy = torch.round(torch.normal(mean=center_ky, std=(ncol - 1) / std_scale, size=(target_count,))).clamp(0, ncol-1).int()

        valid_indices = (temp_mask[indx, indy] == 1) & (loss_mask[indx, indy] != 1)
        valid_indx, valid_indy = indx[valid_indices], indy[valid_indices]
        loss_mask[valid_indx, valid_indy] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask.numpy(), loss_mask.numpy() 


    def uniform_selection(self, input_data, input_mask, num_iter=1):

        nrow, ncol = input_data.shape[0], input_data.shape[1]

        center_kx = int(utils.find_center_ind(input_data, axes=(1, 2)))
        center_ky = int(utils.find_center_ind(input_data, axes=(0, 2)))

        if num_iter == 0:
            print(f'\n Uniformly random selection is processing, rho = {self.rho:.2f}, center of kspace: center-kx: {center_kx}, center-ky: {center_ky}')

        temp_mask = np.copy(input_mask)
        temp_mask[center_kx - self.small_acs_block[0] // 2: center_kx + self.small_acs_block[0] // 2,
        center_ky - self.small_acs_block[1] // 2: center_ky + self.small_acs_block[1] // 2] = 0

        pr = np.ndarray.flatten(temp_mask)
        ind = np.random.choice(np.arange(nrow * ncol),
                               size=int(np.count_nonzero(pr) * self.rho), replace=False, p=pr / np.sum(pr))

        [ind_x, ind_y] = utils.index_flatten2nd(ind, (nrow, ncol))

        loss_mask = np.zeros_like(input_mask)
        loss_mask[ind_x, ind_y] = 1

        trn_mask = input_mask - loss_mask

        return trn_mask, loss_mask
    

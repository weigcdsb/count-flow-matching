"""Trajectory-window dataset utilities."""

import numpy as np
import torch
from torch.utils.data import Dataset

class ToyDsetDynamics(Dataset):
    """
    dataloader for toy datasets with dynamics. expects data in the form of that created by
    my toy data creation methods -- in other words, a list of np.arrays.
    flattens all arrays and creates a set of valid indices of that array to sample from.
    This set of valid indices is also based on nForward: the number of steps forward in time
    that we want our model to predict. 
    When sampling, will return samples of length nForward + 2 (last index is dt)
    
    Now also handles time-varying covariates if provided.
    """

    def __init__(self, data, dt, nForward=1, cov_dynamic_data=None, cov_static_data=None,
                image_lag_source=None, lag_k=0):
        self.maxForward = nForward
        exampleInd = np.random.choice(len(data), 1)[0]
        self.exampleTraj = data[exampleInd]
        
        lens = list(map(len, data))
        lens2 = [0] + list(np.cumsum([l for l in lens][:-1]))
        sets = [np.vstack([np.arange(ii, l+ ii - self.maxForward) for ii in range(self.maxForward + 1)]).T for l in lens]
        sumSets = [p+l for p,l in zip(sets,lens2)]
        validInds = np.vstack(sumSets)
        
        self.data = np.vstack(data)
        self.data_inds = validInds 
        self.dt = dt
        self.length = len(validInds)

        self.cov_dynamic_data = None
        if cov_dynamic_data is not None:
            cov_lens = list(map(len, cov_dynamic_data))
            assert lens == cov_lens, "Dynamic covariate trajectories must have same lengths as data trajectories"
            self.cov_dynamic_data = np.vstack(cov_dynamic_data)

        self.is_image = (self.exampleTraj.ndim == 4)  # (T,C,H,W)
        if self.is_image:
            _, self.C, self.H, self.W = self.exampleTraj.shape
        else:
            self.C = self.H = self.W = None
        

        # Handle dynamic covariates
        self.cov_static_data = None
        self.cov_static_list = None   # NEW: for images we keep list, not vstack
        if cov_static_data is not None:
            if isinstance(cov_static_data, np.ndarray) and len(cov_static_data.shape) == 2:
                expanded_static = []
                for i, l in enumerate(lens):
                    expanded_static.append(np.tile(cov_static_data[i:i+1], (l, 1)))
                self.cov_static_data = np.vstack(expanded_static)
            else:
                # list of arrays [T, dim_static] — for images we keep it as a list
                if self.is_image:
                    self.cov_static_list = [np.asarray(a) for a in cov_static_data]  # NEW
                else:
                    cov_lens = list(map(len, cov_static_data))
                    assert lens == cov_lens, "Static covariate trajectories must have same lengths as data trajectories"
                    self.cov_static_data = np.vstack(cov_static_data)

        self.image_lag_source = image_lag_source if self.is_image else None
        self.lag_k = int(lag_k) if self.is_image else 0

        # NEW: keep trial offsets to map flat indices → (trial_id, local_t)
        self.trial_offsets = np.array([0] + list(np.cumsum(lens)[:-1]))
        self.trial_lengths = np.array(lens)

    def _flat_index_to_trial_t(self, flat_idx: int):
        # Find trial j such that trial_offsets[j] <= flat_idx < trial_offsets[j] + trial_lengths[j]
        j = int(np.searchsorted(self.trial_offsets[1:], flat_idx, side='right'))
        t_local = flat_idx - int(self.trial_offsets[j])
        return j, t_local

    def __len__(self):
        return self.length 
    
    def __getitem__(self, index):
        single_index = False
        result = []
        try:
            iter(index)
        except TypeError:
            index = [index]
            single_index = True

        for ii in index:
            inds = self.data_inds[ii]
            
            # Get data samples
            samples = [self.transform(self.data[ind]) for ind in inds]
            samples.append(self.dt)

            # Add dynamic covariate samples if available
            if self.cov_dynamic_data is not None:
                cov_dynamic_samples = [self.transform(self.cov_dynamic_data[ind]) for ind in inds]
                samples.extend(cov_dynamic_samples)

            # Add static covariate samples if available
            if self.is_image:
                # Build ONE row at the left endpoint (repeat across window)
                j_trial, t_trunc = self._flat_index_to_trial_t(int(inds[0]))
                # t_original = t_trunc + lag_k  (because you truncated T→T-k before creating this dataset)
                t_original = t_trunc + self.lag_k
                # slice original frames [t-k .. t] → concat on channel
                Xorig = self.image_lag_source[j_trial]  # (T, C, H, W) original
                # safety: bounds within trial
                t0 = t_original - self.lag_k
                lag_blocks = [Xorig[t0 + s] for s in range(self.lag_k + 1)]  # [(C,H,W), ...]
                lag_stack = np.concatenate(lag_blocks, axis=0)               # ((k+1)·C, H, W)
                lag_row = lag_stack.reshape(-1)                               # ((k+1)·C·H·W,)

                # true static (time-varying) if provided
                if self.cov_static_list is not None:
                    static_row = self.cov_static_list[j_trial][t_trunc]       # (S,)
                    fused = np.concatenate([static_row, lag_row], axis=0)
                elif self.cov_static_data is not None:
                    # unlikely for images; kept for completeness
                    fused = self.cov_static_data[inds[0]]
                else:
                    fused = lag_row

                fused_t = self.transform(fused)
                samples.extend([fused_t for _ in inds])  # repeat across the window
            else:
                # vector path (unchanged)
                if self.cov_static_data is not None:
                    cov_static_samples = [self.transform(self.cov_static_data[ind]) for ind in inds[:1]]
                    samples.extend(cov_static_samples * len(inds))

            result.append(samples)

        if single_index:
            return result[0]
        return result
    
    def transform(self, data):
        return torch.from_numpy(data).type(torch.FloatTensor)

import h5py
import remfile
import torch
import os
from dandi.download import download
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from nlb_tools.nwb_interface import NWBDataset
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from scipy.io import loadmat
from skimage.transform import resize
from torch.utils.data.dataset import Dataset


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
            iterator = iter(index)
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
                t1 = t_original
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

def validate_metadata(data,metadata):
    """
    just checks metadata against data to ensure
    metadata has the number of trials per split as
    data
    """

    for split in data.keys():
        d = data[split]
        md = metadata[split]['behavior']
        for key in md.keys():
            print(f"checking {split}: {key}")
            assert len(md[key]) == len(d)
        md = metadata[split]['trial_info']
        for key in md.keys():
            print(f"checking {split}: {key}")
            assert len(md[key]) == len(d)

    return

def load_nwb_stream(id,filepath):

    with DandiAPIClient() as client:
        asset = client.get_dandiset(id, 'draft').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
    
        rem_file = remfile.File(s3_url)
        file = h5py.File(rem_file, "r")
        io_stream = NWBHDF5IO(file=file)
        nwbfile_stream = io_stream.read()

    return nwbfile_stream,io_stream


def make_dmfc_rsg_loaders(batch_size=128, nForward=1, num_workers=1, validate=False):

    dandiset_id = '000130'
    filepath = "sub-Haydn/sub-Haydn_desc-train_ecephys.nwb"
    
    nwbfile_stream, io_stream = load_nwb_stream(dandiset_id, filepath)

    # Use raw spikes only (no Gaussian smoothing)
    ds = NWBDataset(fpath=nwbfile_stream, split_heldout=True)
    print("done! using raw spike counts (no smoothing)")

    # Get trial info
    trials = nwbfile_stream.trials[:]
    start_times_s = trials['start_time'].to_numpy()
    end_times_s = trials['stop_time'].to_numpy()
    trial_labels = trials['split'].to_list()

    # Raw spike matrix over continuous time
    rates = ds.data.spikes
    time_s = rates.index.seconds + rates.index.microseconds / 1e6
    rates_vals = rates.to_numpy()

    data = {
        'train': [],
        'val': []
    }

    trial_info_names = [
        'start_time', 'stop_time', 'target_on_time', 'ready_time',
        'set_time', 'go_time', 'reward_time', 'is_eye', 'ts', 'tp'
    ]
    behavior_names = []

    metadata = {
        'train': {
            'trial_info': {name: [] for name in trial_info_names},
            'behavior': {name: [] for name in behavior_names},
        },
        'val': {
            'trial_info': {name: [] for name in trial_info_names},
            'behavior': {name: [] for name in behavior_names},
        },
    }

    # Slice spikes into trials
    for trial_ind, (label, onset, offset) in tqdm(
        enumerate(zip(trial_labels, start_times_s, end_times_s)),
        total=len(trial_labels),
        desc='separating into trials'
    ):
        inds = (time_s >= onset) & (time_s < offset)
        if label != 'none':
            data[label].append(rates_vals[inds, :])

            for name in behavior_names:
                metadata[label]['behavior'][name].append(
                    ds.data.loc[inds][name].to_numpy()
                )
            for name in trial_info_names:
                metadata[label]['trial_info'][name].append(
                    nwbfile_stream.trials[trial_ind][name].item()
                )

    # Optional metadata validation
    if validate:
        validate_metadata(data, metadata)

    # Build datasets and loaders
    train_dataset = ToyDsetDynamics(data['train'], dt=1/1000, nForward=nForward)
    val_dset = ToyDsetDynamics(data['val'], dt=1/1000, nForward=nForward)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    io_stream.close()

    return train_loader, val_loader, data['train'], data['val'], metadata


def loader_mc_maze_with_behavior(nForward=1, num_workers=1, batch_size=128):
    """
    Load MC Maze dataset with RAW spike counts (no Gaussian smoothing) and behavior.
    Note: dataset is large, loading may be memory intensive.
    """

    dandiset_id = '000128'
    filepath = "sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
        s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)

        rem_file = remfile.File(s3_url)
        file = h5py.File(rem_file, "r")
        io_stream = NWBHDF5IO(file=file)
        nwbfile_stream = io_stream.read()
        dataset = NWBDataset(nwbfile_stream, split_heldout=True)

    # Use RAW spikes only (no smoothing)
    spikes = dataset.data.spikes.copy()
    spikes_numpy = spikes.to_numpy()
    spikes['time_stamp'] = spikes.index.total_seconds()

    trial_start_times = dataset.trial_info.start_time.dt.total_seconds().to_numpy()
    trial_end_times = dataset.trial_info.end_time.dt.total_seconds().to_numpy()
    train_val_label = dataset.trial_info.split.to_numpy()

    splitted_data = {
        'train': [],
        'val': [],
    }

    data_timestamp_interval = (dataset.data.index[1] - dataset.data.index[0]).total_seconds()

    trial_info_names = [
        'trial_type', 'start_time', 'stop_time', 'trial_version', 'maze_id', 'success',
        'target_on_time', 'go_cue_time', 'move_onset_time',
        'rt', 'delay', 'num_targets', 'target_pos',
        'num_barriers', 'barrier_pos', 'active_target'
    ]
    behavior_names = ['cursor_pos', 'eye_pos', 'hand_pos', 'hand_vel']

    metadata = {
        'train': {
            'trial_info': {name: [] for name in trial_info_names},
            'behavior': {name: [] for name in behavior_names},
        },
        'val': {
            'trial_info': {name: [] for name in trial_info_names},
            'behavior': {name: [] for name in behavior_names},
        },
    }

    for trial_ind, (trial_start, trial_end, label) in enumerate(
        zip(trial_start_times, trial_end_times, train_val_label)
    ):
        in_trial_index = (spikes.time_stamp >= trial_start) & (spikes.time_stamp <= trial_end)
        in_trial_data = spikes_numpy[in_trial_index, :]

        # find leading/trailing NaNs (first channel) and trim
        isnan = np.isnan(in_trial_data[:, 0])
        lead_end = np.argmax(~isnan)
        tail_start_reverse = np.argmax(~isnan[::-1])
        tail_start = len(in_trial_data[:, 0]) - tail_start_reverse
        nan_index_single_trial = [lead_end, tail_start]

        excluded_nan_single_trial = in_trial_data[
            nan_index_single_trial[0]:nan_index_single_trial[1], :
        ]

        # keep only labeled train/val trials
        if label in splitted_data:
            splitted_data[label].append(excluded_nan_single_trial)

            for name in behavior_names:
                in_trial_behavior = dataset.data.loc[in_trial_index][name].to_numpy()
                metadata[label]['behavior'][name].append(
                    in_trial_behavior[
                        nan_index_single_trial[0]:nan_index_single_trial[1], :
                    ]
                )

            for name in trial_info_names:
                if name == 'start_time':
                    metadata[label]['trial_info'][name].append(
                        nwbfile_stream.trials[trial_ind][name].item()
                        + data_timestamp_interval * lead_end
                    )
                elif name == 'stop_time':
                    metadata[label]['trial_info'][name].append(
                        nwbfile_stream.trials[trial_ind][name].item()
                        - data_timestamp_interval * (tail_start_reverse - 1)
                    )
                else:
                    metadata[label]['trial_info'][name].append(
                        nwbfile_stream.trials[trial_ind][name].item()
                    )

    train_dataset = ToyDsetDynamics(splitted_data['train'], dt=1e-3, nForward=nForward)
    val_dataset = ToyDsetDynamics(splitted_data['val'], dt=1e-3, nForward=nForward)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    io_stream.close()

    return train_dataloader, val_dataloader, splitted_data['train'], splitted_data['val'], metadata


def save_chkpts(nets, path):
    unwrap = (lambda m: m.module if hasattr(m, "module") else m)
    if isinstance(nets, (list, tuple)):
        blob = {f"net_{i}": unwrap(n).state_dict() for i, n in enumerate(nets)}
    else:
        blob = {"net": unwrap(nets).state_dict()}
    torch.save(blob, path)

def load_chkpts(nets, path, map_location=None, strict=True):
    blob = torch.load(path, map_location=map_location)
    unwrap = (lambda m: m.module if hasattr(m, "module") else m)
    if isinstance(nets, (list, tuple)):
        for i, n in enumerate(nets): unwrap(n).load_state_dict(blob[f"net_{i}"], strict=strict)
    else:
        unwrap(nets).load_state_dict(blob["net"], strict=strict)

    
    

"""Optional DANDI/NWB loaders; install the nwb extra to use them."""
import h5py
import remfile
import numpy as np
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO
from nlb_tools.nwb_interface import NWBDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .datasets import ToyDsetDynamics

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

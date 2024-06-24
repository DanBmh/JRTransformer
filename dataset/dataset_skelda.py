import copy
import sys
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import tqdm

from .dataset_3dpw import normalize, rotate_Y

sys.path.append("/PoseForecasters/")
import utils_pipeline

# ==================================================================================================

datamode = "gt-gt"
# datamode = "pred-pred"

config = {
    "item_step": 2,
    "window_step": 2,
    # "item_step": 1,
    # "window_step": 1,
    "select_joints": [
        # "hip_middle",
        "hip_right",
        "hip_left",
        "knee_right",
        "knee_left",
        "ankle_right",
        "ankle_left",
        # "shoulder_middle",
        "nose",
        "shoulder_right",
        "shoulder_left",
        "elbow_right",
        "elbow_left",
        "wrist_right",
        "wrist_left",
    ],
}

# datasets_train = [
#     "/datasets/preprocessed/mocap/train_forecast_samples_4fps.json",
#     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_4fps.json",
#     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_4fps.json",
#     "/datasets/preprocessed/amass/kit_train_forecast_samples_4fps.json"
# ]

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_kppspose.json",
    # "/datasets/preprocessed/human36m/train_forecast_kppspose_10fps.json",
    # "/datasets/preprocessed/human36m/train_forecast_kppspose_4fps.json",
    # "/datasets/preprocessed/mocap/train_forecast_samples.json",
]

# datasets_train = [
#     "/datasets/preprocessed/mocap/train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/kit_train_forecast_samples_10fps.json"
# ]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
# dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_4fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_4fps.json"

# ==================================================================================================


class SkeldaDataset(Dataset):
    def __init__(self, dset_path, seq_len, N, J, split_name="train"):
        self.seq_len = seq_len

        config["input_n"] = seq_len // 2
        config["output_n"] = seq_len - (seq_len // 2)

        # Load preprocessed datasets
        print("Loading datasets ...")
        dataset = None
        if split_name == "train":
            dataset_train, dlen_train = [], 0
            for dp in datasets_train:
                cfg = copy.deepcopy(config)
                if "mocap" in dp:
                    cfg["select_joints"][
                        cfg["select_joints"].index("nose")
                    ] = "head_upper"

                ds, dlen = utils_pipeline.load_dataset(dp, "train", cfg)
                dataset_train.extend(ds["sequences"])
                dlen_train += dlen
            dataset = dataset_train
            dlen = dlen_train
        else:
            if split_name != "test":
                esplit = "test" if "mocap" in dataset_eval_test else "eval"
            else:
                esplit = "test"
            cfg = copy.deepcopy(config)
            if "mocap" in dataset_eval_test:
                cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
            dataset_eval, dlen_eval = utils_pipeline.load_dataset(
                dataset_eval_test, esplit, cfg
            )
            dataset_eval = dataset_eval["sequences"]
            dataset = dataset_eval
            dlen = dlen_eval

        self.data = []
        self.data_para = []

        agentsNum = N
        timeStepsNum = seq_len
        jointsNum = J
        coordsNum = 3  # x y z
        self.dim = 6

        label_gen = utils_pipeline.create_labels_generator(dataset, config)

        nbatch = 1
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen, batch_size=nbatch),
            total=int(dlen / nbatch),
        ):
            sequences_train = utils_pipeline.make_input_sequence(
                batch, "input", datamode, make_relative=False
            )
            sequences_gt = utils_pipeline.make_input_sequence(
                batch, "target", datamode, make_relative=False
            )

            # Convert to meters
            sequences_train = sequences_train / 1000.0
            sequences_gt = sequences_gt / 1000.0

            # Switch y and z axes
            sequences_train = sequences_train[:, :, :, [0, 2, 1]]
            sequences_gt = sequences_gt[:, :, :, [0, 2, 1]]

            # Reshape to [nbatch, npersons, nframes, njoints * 3]
            sequences_train = sequences_train.reshape(
                [nbatch, 1, sequences_train.shape[1], J, 3]
            )
            sequences_gt = sequences_gt.reshape(
                [nbatch, 1, sequences_gt.shape[1], J, 3]
            )

            temp_data = np.concatenate([sequences_train, sequences_gt], axis=2)
            temp_data = temp_data[0]

            # print(temp_data.shape)
            # print(temp_data[0, 0])
            # exit()

            # from . import vis_skelda
            # vis_skelda.visualize(temp_data)
            # exit()

            temp_ = temp_data.copy()
            curr_data, curr_data_para = normalize(temp_data)
            vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
            vel_data[:, 1:, :, :] = (np.roll(curr_data, -1, axis=1) - curr_data)[
                :, :-1, :, :
            ]
            data = np.concatenate((curr_data, vel_data), axis=3)
            self.data.append(data)
            self.data_para.append(curr_data_para)

            if split_name == "train":
                # rotate
                rotate_data = rotate_Y(temp_, 120)
                rotate_data, rotate_data_para = normalize(rotate_data)
                vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
                vel_data[:, 1:, :, :] = (
                    np.roll(rotate_data, -1, axis=1) - rotate_data
                )[:, :-1, :, :]
                data = np.concatenate((rotate_data, vel_data), axis=3)
                self.data.append(data)
                self.data_para.append(rotate_data_para)

                # reverse
                reverse_data = np.flip(temp_, axis=2)
                reverse_data, reverse_data_para = normalize(reverse_data)
                vel_data = np.zeros((agentsNum, timeStepsNum, jointsNum, coordsNum))
                vel_data[:, 1:, :, :] = (
                    np.roll(reverse_data, -1, axis=1) - reverse_data
                )[:, :-1, :, :]
                data = np.concatenate((reverse_data, vel_data), axis=3)
                self.data.append(data)
                self.data_para.append(reverse_data_para)

    def __getitem__(self, idx: int):
        data = self.data[idx].transpose((1, 0, 2, 3))  # [T, N, J, 3]
        para = self.data_para[idx]
        return data, para

    def __len__(self):
        return len(self.data)

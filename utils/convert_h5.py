import argparse
import os

import h5py
import numpy as np

import utils.common_utils as common_utils
import utils.data_utils as du
import utils.kits_data_utils as kutils
import utils.preprocessor as preprocessor


def convert_h5(data_dir, data_split, l, orientation=preprocessor.ORIENTATION['coronal'], remap_config='Neo', dwSample = 4):
    # Data splitting
    train_file = '..\\datasets\\train_data.json'
    val_file =  '..\\datasets\\val_data.json'
    skip_file = '..\\datasets\\dataskip.json'
    if data_split:
        train_file_paths, test_file_paths = du.apply_split(skip_file, train_file, val_file, data_split, data_dir)
    else:
        raise ValueError('You must either provide the split ratio or a train, train dataset list')

    print("Train dataset size: %d, Test dataset size: %d" % (len(train_file_paths), len(test_file_paths)))
    # loading,pre-processing and writing train data
    print("===Train data===")
    data_train, label_train, class_weights_train, weights_train, _ = du.load_dataset(train_file_paths,
                                                                                     orientation,
                                                                                     remap_config=remap_config,
                                                                                     return_weights=True,
                                                                                     reduce_slices=False,
                                                                                     remove_black=False,
                                                                                     downsample=dwSample);

    du.write_h5(data_train, label_train, class_weights_train, weights_train, l, mode='train')

    # loading,pre-processing and writing test data
    print("===Test data===")
    data_test, label_test, class_weights_test, weights_test, _ = du.load_dataset(test_file_paths,
                                                                                 orientation,
                                                                                 remap_config=remap_config,
                                                                                 return_weights=True,
                                                                                 reduce_slices=False,
                                                                                 remove_black=False,
                                                                                 downsample=dwSample)

    du.write_h5(data_test, label_test, class_weights_test, weights_test, l, mode='test')


if __name__ == "__main__":
    print("* Start *")
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-dd', required=True,
                        help='Base directory of the data folder. This folder should contain one folder per volume.')
    parser.add_argument('--data_split', '-ds', required=False,
                        help='Ratio to split data randomly into train and test. input e.g. 80,20')
    parser.add_argument('--remap_config', '-rc', required=False, help='Valid options are "FS" and "Neo"')
    parser.add_argument('--orientation', '-o', required=False, help='Valid options are COR, AXI, SAG')
    parser.add_argument('--destination_folder', '-df', required=True, help='Path where to generate the h5 files')

    args = parser.parse_args()

    common_utils.create_if_not(args.destination_folder)

    f = {
        'train': {
            "data": os.path.join(args.destination_folder, "Data_train.h5"),
            "label": os.path.join(args.destination_folder, "Label_train.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_train.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_train.h5"),
        },
        'test': {
            "data": os.path.join(args.destination_folder, "Data_test.h5"),
            "label": os.path.join(args.destination_folder, "Label_test.h5"),
            "weights": os.path.join(args.destination_folder, "Weight_test.h5"),
            "class_weights": os.path.join(args.destination_folder, "Class_Weight_test.h5")
        }
    }
    convert_h5(args.data_dir,  args.data_split, f, args.orientation)

    print("* Finish *")

import json
import numpy as np
from pathlib import Path
import os

DATA_SET_SIZE = 210
MAX_DATA_LOADED = 50


def get_test_data(a_list, ratio):
    frac = ratio.split(",")
    part = int(len(a_list) * float(frac[0])/100)
    return a_list[part:]


def write_to_file(filePath, jsonContent):
    try:
        with open(filePath, 'w') as tst:
            print("writting file ", filePath)
            json.dump(jsonContent, tst)
    except Exception as e:
        print(type(e))
        print(str(e))
        print("Failed to write to file")


def filter_file_list(data_skip, file_list):
    case_files = []
    data_flt = []

    data_skip = os.path.join(os.getcwd(), "datasets", "dataskip.json")
    with open(data_skip, 'r') as f:
        data_flt = json.load(f)['removal']

    ignore_cases = [get_full_case_id(case) for case in data_flt]
    for cid in file_list:
        if cid in ignore_cases:
            continue
        case_files.append(cid)

    return case_files


def load_file_paths(data_skip, data_path):
    case_paths=[]

    content = os.listdir(data_path)
    cases = [k for k in content if 'case_' in k]
    cases = filter_file_list(data_skip, cases)
    case_paths = [os.path.join(data_path, k) for k in cases]
    file_paths = [
        [os.path.join(vol, 'imaging.nii.gz'), os.path.join(vol, 'segmentation.nii.gz')]
        for
        vol in case_paths]
    return file_paths


def get_train_files():
    original_list = np.array(range(DATA_SET_SIZE));
    print(str(Path(__file__).parent.absolute()))

    original_list = filter_file_list(original_list)

    test_data = []
    try:
        with open('..\datasets\eval_volumes.json', 'r') as tst:
            test_data = json.load(tst)
    except IOError:
        print("No data split for testing provided")
    original_list = np.delete(original_list, test_data['test_cases'])

    st = min(MAX_DATA_LOADED, len(original_list))
    return original_list[:st]


def get_data_paths(data_dir, case_list):
    file_list = []
    for cid in case_list:
        data_path = get_case_path(Path(data_dir), cid)
        file_list.append(data_path)
    return file_list


def get_full_case_id(cid):
    try:
        cid = int(cid)
        case_id = "case_{:05d}".format(cid)
    except ValueError:
        case_id = cid

    return case_id


def get_case_path(data_path, cid):
    # Resolve location where data should be living
    if not data_path.exists():
        raise IOError(
            "Data path, {}, could not be resolved".format(str(data_path))
        )

    # Get case_id from provided cid
    case_id = get_full_case_id(cid)

    # Make sure that case_id exists under the data_path
    case_path = data_path / case_id
    if not case_path.exists():
        raise ValueError(
            "Case could not be found \"{}\"".format(case_path.name)
        )

    return case_path



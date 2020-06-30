import json
import os
import re


def write_to_file(filePath, jsonContent):
    try:
        with open(filePath, 'w') as tst:
            print("writting file ", filePath)
            json.dump(jsonContent, tst)
    except Exception as e:
        print(type(e))
        print(str(e))
        print("Failed to write to file")


def get_case_numbers_from_file(file_path):
    case_numbers = []
    try:
        with open(file_path, 'r') as tst:
            test_case_ids = json.load(tst)
            case_numbers = test_case_ids['cases']
    except IOError:
        print("Cannot read case numbers from file")
    return case_numbers


def filter_case_numbers(data_skip, test_data, data_path):

    content = os.listdir(data_path)
    cases = [k for k in content if 'case_' in k]
    case_numbers = [int(re.split('_', case)[1]) for case in cases]
    case_numbers = [x for x in case_numbers if x not in data_skip]
    case_numbers = [x for x in case_numbers if x not in test_data]

    return case_numbers


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



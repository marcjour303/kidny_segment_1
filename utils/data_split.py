import argparse
from utils.kits_data_utils import create_eval_data_file

if __name__ == '__main__':
    desc = "Split the data into training and testing. The training needs to be further split for validation." \
           "This testing data does not get included in any computations except for the evaluation step."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "-r", "--ratio", required=False, default="80,20",
        help="The identifier for the case you would like to visualize"
    )
    args = parser.parse_args()

    create_eval_data_file('..\datasets\eval_volumes.json', args.ratio)

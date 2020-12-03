import os, argparse

params = {
    "variance_threshold": [
        "0.001",
        "0.01",
        "0.05",
        "0.1",
    ],
    "kbest": [
        "0.9",
        "0.8",
        "0.7",
        "0.6",
    ],
    "linearSVC": [
        "0.1",
        "1",
        "10",
        "100"
    ],
    "tree": [
        "10",
        "50",
        "100",
        "200",
    ],
    "rfecv": [
        "3",
        "4",
        "5"
    ]
}

open_ml_problems_folder = 'OpenML_datasets'

def main():

    parser = argparse.ArgumentParser(description="Get features extraction from specific methods and params")

    parser.add_argument('--ntrain', type=int, help='number of training in order to keep mean of score', default=1)
    parser.add_argument('--output', type=str, help='output features selection results', required=True)

    args = parser.parse_args()

    p_ntrain    = args.ntrain
    p_output    = args.output

    open_ml_problems = os.listdir(open_ml_problems_folder)

    for ml_problem in open_ml_problems:

        ml_problem_name = ml_problem.replace('.csv', '')
        ml_problem_path = os.path.join(open_ml_problems_folder, ml_problem)

        for key, values in params.items():

            for param in values:

                print(f'Run features selection for OpenML `{ml_problem_name}` problem with {{method: {key}, params: {param}, ntrain: {p_ntrain}}}')
                command_str = f'python features_selection/run_method_openML.py ' \
                            f'--data {ml_problem_path} ' \
                            f'--method {key} ' \
                            f'--params {param} ' \
                            f'--ntrain {p_ntrain} ' \
                            f'--output {p_output}'
                             
                os.system(command_str)

if __name__ == "__main__":
    main()
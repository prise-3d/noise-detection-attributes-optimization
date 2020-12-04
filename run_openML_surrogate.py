import os, argparse

open_ml_problems_folder = 'OpenML_datasets'

def main():

    parser = argparse.ArgumentParser(description="Find best features for each OpenML problems")

    parser.add_argument('--ils', type=int, help='number of total iteration for ils algorithm', required=True)
    parser.add_argument('--ls', type=int, help='number of iteration for Local Search algorithm', required=True)

    args = parser.parse_args()

    p_ils = args.ils
    p_ls  = args.ls

    open_ml_problems = sorted(os.listdir(open_ml_problems_folder))

    for ml_problem in open_ml_problems:

        ml_problem_name = ml_problem.replace('.csv', '')
        ml_problem_path = os.path.join(open_ml_problems_folder, ml_problem)

        ml_surrogate_command = f"python find_best_attributes_surrogate_openML.py " \
                               f"--data {ml_problem_path} " \
                               f"--ils {p_ils} " \
                               f"--ls {p_ls} " \
                               f"--output {ml_problem_name}"
        print(f'Run surrogate features selection for {ml_problem_name} with [ils: {p_ils}, ls: {p_ls}]')
        print(ml_surrogate_command)
        os.system(ml_surrogate_command)
    

if __name__ == "__main__":
    main()
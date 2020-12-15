import os, argparse
import shutil

open_ml_problems_folder = 'OpenML_datasets'
surrogate_data_path = 'data/surrogate/data/'

# fixed test params as first part
k_params = [100] # 100, 150, 200
k_random = [0] # 0, 1
k_reinit = [0] # 0, 1
every_ls = 50

n_times = 5

def main():

    parser = argparse.ArgumentParser(description="Find best features for each OpenML problems")

    parser.add_argument('--ils', type=int, help='number of total iteration for ils algorithm', required=True)
    parser.add_argument('--ls', type=int, help='number of iteration for Local Search algorithm', required=True)

    args = parser.parse_args()

    p_ils = args.ils
    p_ls  = args.ls

    open_ml_problems = sorted(os.listdir(open_ml_problems_folder))

    for ml_problem in open_ml_problems:

        # for each problem prepare specific pre-computed real solution file
        ml_problem_name = ml_problem.replace('.csv', '')
        ml_problem_path = os.path.join(open_ml_problems_folder, ml_problem)

        # ml_surrogate_command = f"python find_best_attributes_surrogate_openML_multi_specific.py " \
        #                        f"--data {ml_problem_path} " \
        #                        f"--ils {p_ils} " \
        #                        f"--ls {p_ls} " \
        #                        f"--output {ml_problem_name} " \
        #                        f"--generate_only 1"
        # print(f'Running extraction real evaluations data for {ml_problem_name}')
        # os.system(ml_surrogate_command)

        # real_evaluation_data_file_path = os.path.join(surrogate_data_path, ml_problem_name)

        # for each multi param:
        # - copy precomputed real_evaluation_data_file
        # - run new instance using specific data
        for k in k_params:
            for k_r in k_random:
                for k_init in k_reinit:

                    # if not use of k_reinit and use of random, then run multiple times this instance to do mean later
                    if k_init == 0 and k_r == 1:
                        for i in range(n_times):

                            str_index = str(i)

                            while len(str_index) < 3:
                                str_index = "0" + str_index

                            output_problem_name = f'{ml_problem_name}_everyLS_{every_ls}_k{k}_random{k_r}_reinit{k_init}_{str_index}'

                            # copy pre-computed real evaluation data for this instance
                            current_output_real_eval_path = os.path.join(surrogate_data_path, output_problem_name)
                            # shutil.copy2(real_evaluation_data_file_path, current_output_real_eval_path)

                            ml_surrogate_multi_command = f"python find_best_attributes_surrogate_openML_multi_specific.py " \
                                            f"--data {ml_problem_path} " \
                                            f"--ils {p_ils} " \
                                            f"--ls {p_ls} " \
                                            f"--every_ls {every_ls} " \
                                            f"--k_division {k} " \
                                            f"--k_random {k_r} " \
                                            f"--output {output_problem_name}"
                                            f"--k_dynamic {k_init} " \
                            print(f'Running extraction data for {ml_problem_name} with [ils: {p_ils}, ls: {p_ls}, k: {k}, k_r: {k_r}, i: {i}]')
                            os.system(ml_surrogate_multi_command)

                    else:
                        output_problem_name = f'{ml_problem_name}_everyLS_{every_ls}_k{k}_random{k_r}_reinit{k_init}'

                        # copy pre-computed real evaluation data for this instance
                        current_output_real_eval_path = os.path.join(surrogate_data_path, output_problem_name)
                        # shutil.copy2(real_evaluation_data_file_path, current_output_real_eval_path)

                        ml_surrogate_multi_command = f"python find_best_attributes_surrogate_openML_multi_specific.py " \
                                        f"--data {ml_problem_path} " \
                                        f"--ils {p_ils} " \
                                        f"--ls {p_ls} " \
                                        f"--every_ls {every_ls} " \
                                        f"--k_division {k} " \
                                        f"--k_random {k_r} " \
                                        f"--output {output_problem_name}"
                                        f"--k_dynamic {k_init} " \
                        print(f'Running extraction data for {ml_problem_name} with [ils: {p_ils}, ls: {p_ls}, k: {k}, k_r: {k_r}]')
                        os.system(ml_surrogate_multi_command)



if __name__ == "__main__":
    main()
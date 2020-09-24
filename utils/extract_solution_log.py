import argparse
import os

def main():

    parser = argparse.ArgumentParser(description="Train and find best filters to use for model")

    parser.add_argument('--log', type=str, help='log file attribute', required=True)
    parser.add_argument('--output', type=str, help='output solution choice', required=True)

    args = parser.parse_args()

    p_log    = args.log
    p_output = args.output

    with open(p_log, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if 'Current Binary solution' in line:
                score = float(line.split('SCORE')[-1])
                
                solution = list(map(int, line.split('[')[-1].split(']')[0].split(' ')))
                
                with open(p_output, 'a') as f:
                    
                    line = ''

                    for index, v in enumerate(solution):
                        line += str(v)

                        if index < len(solution) - 1:
                            line += ','
                    line += ';' + str(score)

                    f.write(line + '\n')


if __name__ == "__main__":
    main()
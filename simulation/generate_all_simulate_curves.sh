for file in "threshold_map"/*; do

    echo ${file}

    python display_simulation_curves.py --folder ${file}
done
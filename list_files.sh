search_dir=$1
sentence=$2

for entry in "$search_dir"/*
do
    if [ -f $entry ]; then
        if grep -q "$sentence" "$entry"; then
            echo "$entry"
        fi
    fi
done


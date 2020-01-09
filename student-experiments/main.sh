search_dir="./data/datasets/normalised/stats"

for entry in "$search_dir"/*
do
  echo "$entry"
  python -m scoop main.py -m 50 -i 1000 -d $entry
done


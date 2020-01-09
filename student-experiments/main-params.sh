search_dir="./data/datasets/normalised/stats"

for entry in "$search_dir"/*
do
  echo "$entry"
  python main.py -i 1000 -p -d $entry
done

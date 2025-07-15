export WANDB_MODE=disabled

for split in 125k 250k 500k 1m 2m 4m 8m; do
    # test chunking on a small test set
    # printed hex bytes are magenta when there's a new token (a new insertion). total number of insertions (tokens) is printed at the end to the file
    echo $split
    hal --train bytes:data/bruk-test10k.txt --reset-step 0 --wd 0 --init exp/bruk$split.pt --save '' --bptt-len 1 --lr 0 | awk '/bpt:/ {print $8}' | jq -rs 'add/8'
done

# ran modal volume get for the checkpoints below
for ep in 2 4 8 16 24; do
    split=8m
    echo $split epoch $ep
    hal --rnn-size 1024 --train bytes:data/bruk-test10k.txt --reset-step 0 --wd 0 --init exp/bruk8m-w1024-lr=0.0001-epoch$ep.pt --save '' --bptt-len 1 --lr 0 | awk '/bpt:/ {print $8}' | jq -rs 'add/8'
done

mkdir -p data
wget -O data/bruk.txt --continue https://gist.githubusercontent.com/proger/5be9c7e095839d11563e1925147ba9f4/raw/dfb19c8a31c51863755dd6f2469430c90977e128/bruk.txt
# make train splits
head -c 125000 data/bruk.txt > data/bruk125k.txt
head -c 250000 data/bruk.txt > data/bruk250k.txt
head -c 500000 data/bruk.txt > data/bruk500k.txt
head -c 1000000 data/bruk.txt > data/bruk1m.txt
head -c 2000000 data/bruk.txt > data/bruk2m.txt
head -c 4000000 data/bruk.txt > data/bruk4m.txt
# make test splits
head -c 8000000 data/bruk.txt > data/bruk8m.txt
tail -c 1000000 data/bruk.txt > data/bruk-test1m.txt
tail -c 10000 data/bruk.txt > data/bruk-test10k.txt

export WANDB_MODE=disabled

# train for 1 epoch
hal --train bytes:data/bruk125k.txt --reset-step 0 --wd 0.001 --save exp/bruk125k.txt

# test chunking on a small test set
# printed hex bytes are magenta when there's a new token (a new insertion). total number of insertions (tokens) is printed at the end.
hal --train bytes:data/bruk-test10k.txt --reset-step 0 --wd 0  --init exp/bruk125k.txt --save '' --chunk --bptt-len 1 --lr 0

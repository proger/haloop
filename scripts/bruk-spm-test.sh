for split in 125k 250k 500k 1m 2m 4m 8m; do
    # test chunking on a small test set
    # printed hex bytes are magenta when there's a new token (a new insertion). total number of insertions (tokens) is printed at the end to the file
    for vocab in 512 1024 2048 4096 8192 16384 32768 65536 131072; do
        echo $split $vocab
    
        python -m ha.spm_train --vocab_size $vocab --model exp/bruk$split-$vocab.spm --test data/bruk-test10k.txt data/bruk$split.txt | grep bits=
    done
done


# haloop

[![PyPI Version](https://img.shields.io/pypi/v/haloop.svg)](https://pypi.python.org/pypi/haloop)

Haloop is a speech agent toolkit. Haloop provides `hac` program for acoustic model training, `hal` for RNN language model training and evaluation and `hat` for attention decoder LM. The package is available on PyPI:

```
pip install haloop
```

Currently, `hat` is a REPL for Ukrainian GPT-2 models from the paper [GPT-2 Metadata Pretraining Towards Instruction Finetuning for Ukrainian](https://github.com/proger/uk4b).

To use `hat`, install some additional dependencies and models:

```
pip install bitsandbytes sentencepiece

wget https://a.wilab.org.ua/gpt/wiki.model  # sentencepiece tokenizer
wget https://a.wilab.org.ua/gpt/ckpt10m.pt  # model checkpoint for GPT-2 Large
```

Now, kick start the REPL:

```
hat --spm wiki.model ckpt10m.pt
```

Please cite:

```
@inproceedings{kyrylov-chaplynskyi-2023-gpt,
    title = "{GPT}-2 Metadata Pretraining Towards Instruction Finetuning for {U}krainian",
    author = "Kyrylov, Volodymyr  and
      Chaplynskyi, Dmytro",
    booktitle = "Proceedings of the Second Ukrainian Natural Language Processing Workshop (UNLP)",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.unlp-1.4",
    pages = "32--39",
    abstract = "We explore pretraining unidirectional language models on 4B tokens from the largest curated corpus of Ukrainian, UberText 2.0. We enrich document text by surrounding it with weakly structured metadata, such as title, tags, and publication year, enabling metadata-conditioned text generation and text-conditioned metadata prediction at the same time. We pretrain GPT-2 Small, Medium and Large models each on single GPU, reporting training times, BPC on BrUK and BERTScore on titles for 1000 News from the Future. Next, we venture to formatting POS and NER datasets as instructions, and train low-rank attention adapters, performing these tasks as constrained text generation. We release our models for the community at https://github.com/proger/uk4b.",
}
```

See also [Speech Discrimination by Dynamic Programming, T. K. Vintsyuk (1968)](https://link.springer.com/article/10.1007/BF01074755)

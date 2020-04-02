# Subword tokenizer Malayalam
A subword tokenizer for Malayalam using encoder-double-decoder architecture.

## Tokenization
The `subword_tokenizer.py` script returns subwords seperated by '+'

## Usage
The following usage accepts the `word` as input and writes the subwords into the console.
```
python subword_tokenizer.py word
```

## Alternatively
You can import the `tokenize` function from `subword_tokenizer.py` and call it with the `word` as input.

```
from subword_tokenizer import tokenize
subwords = tokenize(word)
```

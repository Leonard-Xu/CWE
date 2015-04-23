#Train Model

##Requirements
- GCC/Clang
- Unix/Linux/OSX (Windows is not supported because of `wchar`)

##Usage
- use `make` to compile `cwe.c`
- run `./cwe -train corpus.txt -output-word word.txt -output-char char.txt`
- run `./cwe` for parameter details 

##Input
Segmented chinese corpus encoded in UTF-8. 

Example:

```
我 能 吞下 玻璃 而不 伤 身体
...
你好 世界
...

```

##Output
####word embeddings
```
N M
word#1 [x#1, x#2, ..., x#M]
...
word#N [x#1, x#2, ..., x#M]
```
These embeddings are the combinition of word and character vectors, i.e. x = mean(w + mean(ci)). 

####character embeddings
```
N M
character#1 pos [c#1, c#2, ..., c#M]
...
character#N pos [c#1, c#2, ..., c#M]
```

`pos` may be `{b, m, e, s}` in CWE+P and CEW+LP, which means `{begin, middle, end, single}` .

`pos` will be `a` in other cases, which means `all`.

If `character#i` and `character#j` are the same character and with the same pos, then they are two clusters of the character.

The output will be sorted by characters' Unicode.

#CWE

Most word embedding methods take a word as a basic unit and learn embeddings according to words’ external contexts, ignoring the internal structures of words. However, in some languages such as Chinese, a word is usually composed of several characters and contains rich internal information. The semantic meaning of a word is also related to the meanings of its composing characters. Hence, we take Chinese for example, and present a character-enhanced word embedding model (CWE). In order to address the issues of character ambiguity and non-compositional words, we propose multiple-prototype character embeddings and an effective word selection method. We evaluate the effectiveness of CWE on word relatedness computation and analogical reasoning. The results show that CWE outperforms other baseline methods which ignore internal character information.

The work is published in IJCAI 2015, entitled with "Joint Learning of Character and Word Embeddings". This project maintains the source codes and evaluation data for character-enhanced word embedding model (CWE). The analogical reasoning dataset on Chinese is available in data folder. Hope the codes and data are helpful for your research in NLP. If you use the codes or the data, please cite this paper:

Xinxiong Chen, Lei Xu, Zhiyuan Liu, Maosong Sun, Huanbo Luan. Joint Learning of Character and Word Embeddings. The 25th International Joint Conference on Artificial Intelligence (IJCAI 2015).

Download paper：![link](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf)


##Note
The CWE project (MIT license) is based on Google's word2vec project (Apache 2.0 License).


by Leonard Xu

leonard.xu.thu@gmail.com

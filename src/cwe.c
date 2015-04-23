//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <locale.h>
#include <wchar.h>


#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

//Unicode range of Chinese characters
#define MIN_CHINESE 0x4E00
#define MAX_CHINESE 0x9FA5

// Maximum 30 * 0.7 = 21M words in the vocabulary
const int vocab_hash_size = 30000000;

// Precision of float numbers
typedef float real;

struct vocab_word {
  long long cn;
  int *point, *character, character_size, *character_emb_select;
  char *word, *code, codelen;

  /*
   * character[i]   : Unicode(the i-th character in the word) - MIN_CHINESE
   * character_size : the length of the word
                      (not equal to the length of string due to UTF-8 encoding)
   * character_emb_select[i][j] : count character[i] select the j-th cluster
   */
};

char train_file[MAX_STRING], output_word[MAX_STRING], output_char[MAX_STRING];
char non_comp[MAX_STRING], char_init[MAX_STRING];
struct vocab_word *vocab;
int cbow = 1, debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;
int cwe_type = 2, multi_emb = 3, *embed_count, cwin = 5;
int nonpara_limit = 1000, *last_emb_cnt;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, dim = 100, character_size = 0;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3, nonpara = 1e-3, char_rate = 1;
real *syn0, *syn1, *syn1neg, *expTable, *charv;
clock_t start;

int hs = 0, negative = 5;
const int table_size = 1e8;
int *table;

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word, int is_non_comp) {
  unsigned int hash, length = strlen(word) + 1, len, i, pos;
  wchar_t wstr[MAX_STRING];
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  if (!cwe_type || is_non_comp) return vocab_size - 1;
  len = mbstowcs(wstr, word, MAX_STRING);
  for (i = 0; i < len; i++)
    if (wstr[i] < MIN_CHINESE || wstr[i] > MAX_CHINESE) {
      vocab[vocab_size - 1].character = 0;
      vocab[vocab_size - 1].character_size = 0;
      return vocab_size - 1;
    }
  vocab[vocab_size - 1].character = calloc(len, sizeof(int));
  vocab[vocab_size - 1].character_size = len;
  for (i = 0; i < len; i++) {
    if (cwe_type == 1 || cwe_type == 3 || cwe_type == 5) {
      //character
      vocab[vocab_size - 1].character[i] = wstr[i] - MIN_CHINESE;
    } else if (cwe_type == 2 || cwe_type == 4) {
      //character + positon
      if (len == 1)
        pos = 0;
      else if (i == 0)
        pos = 1;
      else if (i == len - 1)
        pos = 2;
      else
        pos = 3;
      vocab[vocab_size - 1].character[i] = (wstr[i] - MIN_CHINESE) * 4 + pos;
    }
  }
  if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
    vocab[vocab_size - 1].character_emb_select = (int *)calloc(len * multi_emb, sizeof(int));
  }
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      if (vocab[a].character != NULL) free(vocab[a].character);
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    memcpy(vocab + b, vocab + a, sizeof(struct vocab_word));
    b++;
  } else {
    if (vocab[a].character != NULL) free(vocab[a].character);
    free(vocab[a].word);
  }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnNonCompWord() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  fin = fopen(non_comp, "rb");
  if (fin == NULL) {
    printf("ERROR: non-compositional wordlist file not found!\n");
    exit(1);
  }
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word, 1);
      vocab[a].cn = 0;
    }
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>", 0);
  if (strlen(non_comp)) LearnNonCompWord();
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word, 0);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b, c, t1, t2;
  real *vec = calloc(dim, sizeof(real)), len;
  wchar_t buf[10];
  FILE *file;

  unsigned long long next_random = 1;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * dim * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * dim * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < dim; b++)
     syn1[a * dim + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * dim * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < dim; b++)
     syn1neg[a * dim + b] = 0;
  }
  for (a = 0; a < vocab_size; a++) for (b = 0; b < dim; b++) {
    next_random = next_random * (unsigned long long)25214903917 + 11;
    syn0[a * dim + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dim;
  }
  if (cwe_type) {
    a = posix_memalign((void **)&charv, 128, (long long)character_size * dim * sizeof(real));
    if (charv == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < (long long)character_size * dim; a++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      charv[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / dim;
    }
    if (cwe_type == 5) {
      a = posix_memalign((void **)&embed_count, 128, (MAX_CHINESE - MIN_CHINESE + 1) * sizeof(int));
      if (embed_count == NULL) {printf("Memory allocation failed\n"); exit(1);}
      a = posix_memalign((void **)&last_emb_cnt, 128, (MAX_CHINESE - MIN_CHINESE + 1) * sizeof(int));
      if (last_emb_cnt == NULL) {printf("Memory allocation failed\n"); exit(1);}
      for (a = 0; a < (MAX_CHINESE - MIN_CHINESE + 1); a++)
        embed_count[a] = 0;
    }
  }

  if (strlen(char_init)) {
    file = fopen(char_init, "r");
    fscanf(file, "%lld%lld", &a, &b);
    if (b != dim) {
      printf("Unable to load -char-init\n");
    } else {
      while (a--) {
        fscanf(file, "%ls", buf);
        if (wcslen(buf) != 1) continue;
        b = buf[0];
        if (b < MIN_CHINESE || b > MAX_CHINESE) continue;
        b -= MIN_CHINESE;
        t1 = b, t2 = b + 1;
        if (cwe_type == 2 || cwe_type == 4)
          t1 *= 4, t2 *= 4;
        if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5)
          t1 *= multi_emb, t2 *= multi_emb;
         len = 0;
         for (c = 0; c < dim; c++) {
           fscanf(file, "%f", vec + c);
           len += vec[c] * vec[c];
         }
         len = sqrt(len);
         for (c = 0; c < dim; c++) {
           for (b = t1; b < t2; b++) {
             charv[c + b * dim] = vec[c] / len;
           }
         }
      }
    }
    fclose(file);
  }


  CreateBinaryTree();
}

//select a character embedding given contect vector
int get_emb(real *vec, long long charv_id, int word_id, int char_id) {
  long long i, j, k, best = 0;
  real dotv, len1, len2, bestv = -1e8;
  int cnt = multi_emb;
  if (cwe_type == 5)
    cnt = embed_count[charv_id];
  for (i = 0; i < cnt; i++) {
    dotv = 0; len1 = 0; len2 = 0;
    for (j = 0; j < dim; j++) {
      dotv += vec[j] * charv[j + (charv_id * multi_emb + i) * dim];
      len1 += vec[j] * vec[j];
      len2 += charv[j + (charv_id * multi_emb + i) * dim] * charv[j + (charv_id * multi_emb + i) * dim];
    }
    if (len1 != 0 && len2 != 0) {
      dotv /= sqrt(len1) * sqrt(len2);
    } else {
      dotv = 0;
    }
    if (dotv > bestv) {
      bestv = dotv;
      best = i;
    }
  }
  if (cwe_type == 5 && bestv < nonpara) {
    if (cnt == 0 || (last_emb_cnt[charv_id] > nonpara_limit && cnt < multi_emb)) {
      best = cnt;
      embed_count[charv_id]++;
      last_emb_cnt[charv_id] = 0;
    } else if (last_emb_cnt[charv_id] < nonpara_limit) {
      best = cnt - 1;
    }
  }
  if (cwe_type == 5 && best == embed_count[charv_id] - 1) last_emb_cnt[charv_id]++;
  vocab[word_id].character_emb_select[char_id * multi_emb + best]++;
  return best + charv_id * multi_emb;
}

//get the most frequently used character vector among multi-embeddings
int get_res_emb(int word_id, int char_id, int charv_id) {
  int i, maxn = 0, maxni = 0;
  for (i = 0; i < multi_emb; i++) {
    if (vocab[word_id].character_emb_select[char_id * multi_emb + i] > maxn) {
      maxn = vocab[word_id].character_emb_select[char_id * multi_emb + i];
      maxni = i;
    }
  }
  return charv_id * multi_emb + maxni;
}

//get context vector for selecting character embedding
void get_base(real *base, long long *sen, long long sentence_length, long long index) {
  long long i, j, k, c, charv_id;
  for (i = 0; i < dim; i++) base[i] = 0;
  for (i = index - cwin;  i <= index + cwin; i++) {
    if (i < 0 || i >= sentence_length) continue;
    c = sen[i];
    if (c == -1) continue;
    for (j = 0; j < dim; j++) base[j] += syn0[c * dim + j];
    for (k = 0; k < vocab[c].character_size; k++) {
      charv_id = get_res_emb(c, k, vocab[c].character[k]);
      for (j = 0; j < dim; j++) base[j] += charv[charv_id * dim + j] / vocab[c].character_size;
    }
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, cw, t1, t2, word, last_word, sentence_length = 0, sentence_position = 0, charv_id;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter, index;
  long long *charv_id_list = calloc(MAX_SENTENCE_LENGTH, sizeof(long long));
  int char_list_cnt;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(dim, sizeof(real));
  real *neu1char = calloc(dim, sizeof(real));
  real *neu1e = (real *)calloc(dim, sizeof(real));
  real *base = (real *)calloc(dim, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < dim; c++) neu1[c] = 0;
    for (c = 0; c < dim; c++) neu1e[c] = 0;
    for (c = 0; c < dim; c++) base[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;


    if (cbow) {  //train the cbow architecture

      // in -> hidden
      cw = 0;
      char_list_cnt = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        index = c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < dim; c++) neu1char[c] = 0;
        for (c = 0; c < dim; c++) neu1char[c] = syn0[c + last_word * dim];
        if (cwe_type && vocab[last_word].character_size) {
          if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
            get_base(base, sen, sentence_length, index);
          }
          for (c = 0; c < vocab[last_word].character_size; c++) {
            charv_id = vocab[last_word].character[c];
            if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
              charv_id = get_emb(base, charv_id, last_word, c);
            }
            for (d = 0; d < dim; d++)
              neu1char[d] += charv[d + charv_id * dim] / vocab[last_word].character_size;
            charv_id_list[char_list_cnt] = charv_id;
            char_list_cnt++;
          }
          for (d = 0; d < dim; d++) neu1char[d] /= 2;
        }
        for (c = 0; c < dim; c++) neu1[c] += neu1char[c];
        cw++;
      }
      if (cw) {
        for (c = 0; c < dim; c++) neu1[c] /= cw;
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * dim;
          // Propagate hidden -> output
          for (c = 0; c < dim; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < dim; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * dim;
          f = 0;
          for (c = 0; c < dim; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < dim; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
          c = sentence_position - window + a;
          if (c < 0) continue;
          if (c >= sentence_length) continue;
          last_word = sen[c];
          if (last_word == -1) continue;
          for (c = 0; c < dim; c++) syn0[c + last_word * dim] += neu1e[c];
        }
        for (a = 0; a < char_list_cnt; a++) {
          charv_id = charv_id_list[a];
          for (c = 0; c < dim; c++) charv[c + charv_id * dim] += neu1e[c] * char_rate;
        }
      }



    } else { //train the skip-gram architecture

      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {

        index = c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * dim;
        for (c = 0; c < dim; c++) neu1[c] = 0;
        for (c = 0; c < dim; c++) neu1[c] = syn0[c + l1];
        char_list_cnt = 0;
        if (cwe_type && vocab[last_word].character_size) {
          if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
            get_base(base, sen, sentence_length, index);
          }
          for (c = 0; c < vocab[last_word].character_size; c++) {
            charv_id = vocab[last_word].character[c];
            if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
              charv_id = get_emb(base, charv_id, last_word, c);
            }
            for (d = 0; d < dim; d++)
              neu1[d] += charv[d + charv_id * dim] / vocab[last_word].character_size;
            charv_id_list[char_list_cnt] = charv_id;
            char_list_cnt++;
          }
          for (d = 0; d < dim; d++) neu1[d] /= 2;
        }

        for (c = 0; c < dim; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * dim;
          // Propagate hidden -> output
          for (c = 0; c < dim; c++) f += neu1[c] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < dim; c++) syn1[c + l2] += g * neu1[c];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * dim;
          f = 0;
          for (c = 0; c < dim; c++) f += neu1[c] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < dim; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < dim; c++) syn1neg[c + l2] += g * neu1[c];
        }
        // Learn weights input -> hidden
        for (c = 0; c < dim; c++) syn0[c + l1] += neu1e[c];
        for (c = 0; c < char_list_cnt; c++) {
          charv_id = charv_id_list[c];
          for (d = 0; d < dim; d++) charv[d + charv_id * dim] += neu1e[d] * char_rate;
        }
      }



    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d, charv_id;
  long long tot;
  wchar_t ch[10];
  char buf[10], pos;
  real *vec = calloc(dim, sizeof(real));
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  LearnVocabFromTrainFile();
  if (output_word[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_word, "wb");
    // Save the word vectors
    fprintf(fo, "%lld\t%lld\n", vocab_size, dim);
    for (a = 0; a < vocab_size; a++) {
      for (b = 0; vocab[a].word[b] != 0; b++) fputc(vocab[a].word[b], fo);
      fputc('\t', fo);
      for (b = 0; b < dim; b++) vec[b] = 0;
      for (b = 0; b < dim; b++) vec[b] = syn0[b + a * dim];
      if (cwe_type && vocab[a].character_size) {
        for (b = 0; b < vocab[a].character_size; b++) {
          charv_id = vocab[a].character[b];
          if (cwe_type == 3 || cwe_type == 4 || cwe_type == 5) {
            charv_id = get_res_emb(a, b, charv_id);
          }
          for (c = 0; c < dim; c++) vec[c] += charv[c + charv_id * dim] / vocab[a].character_size;
        }
      }
      for (b = 0; b < dim; b++) fprintf(fo, "%lf\t", vec[b]);
      fprintf(fo, "\n");
    }
  fclose(fo);
  if (strlen(output_char)) {
    fo = fopen(output_char, "wb");
    if (cwe_type == 5) {
      tot = 0;
      for (a = 0; a <= MAX_CHINESE - MIN_CHINESE; a++)
        tot += embed_count[a];
    } else
      tot = character_size;
    fprintf(fo, "%lld\t%lld\n", tot, dim);
    for (a = 0; a < character_size; a++) {
      if (cwe_type == 1) {
        ch[0] = MIN_CHINESE + a;
        ch[1] = 0;
        fprintf(fo, "%ls\ta\t", ch);
      } else if (cwe_type == 2 || cwe_type == 4) {
        if (cwe_type == 2)
          ch[0] = MIN_CHINESE + a / 4;
        else
          ch[0] = MIN_CHINESE + a / 4 / multi_emb;
        ch[1] = 0;
        if (cwe_type == 2)
          c = a % 4;
        else
          c = a / multi_emb % 4;
        if (c == 0)
          pos = 's';
        else if (c == 1)
          pos = 'b';
        else if (c == 2)
          pos = 'e';
        else
          pos = 'm';
        fprintf(fo, "%ls\t%c\t", ch, pos);
      } else if (cwe_type == 3 || cwe_type == 5) {
        ch[0] = MIN_CHINESE + a / multi_emb;
        ch[1] = 0;
        c = a % multi_emb;
        if (cwe_type == 5 && c >= embed_count[a / multi_emb])
          continue;
        fprintf(fo, "%ls\ta\t", ch);
      }
      for (b = 0; b < dim; b++) fprintf(fo, "%lf\t", charv[b + dim * a]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  setlocale(LC_ALL, "en_US.UTF-8");
  if (argc == 1) {
    printf("Joint Learning of Character and Word Embeddings DEMO\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text corpus <file> to train the model\n");
    printf("\t-non-comp <file>\n");
    printf("\t\tUse wordlist from <file> to learn non-compositional words\n");
    printf("\t-char-init <file>\n");
    printf("\t\tUse pre-trained character vectors from <file>\n");
    printf("\t-output-word <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors\n");
    printf("\t-output-char <file>\n");
    printf("\t\tUse <file> to save the resulting character vectors\n");

    printf("\t-size <int>\n");
    printf("\t\tSet size of word and character vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse cbow; default is 1 (0 for skip gram)\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.05 (for CBOW) and 0.025 (for skip-gram)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");

    printf("\t-cwe-type <int>\n");
    printf("\t\tSet cwe type; default is 2(CWE+P), 1(CWE), 0(word2vec), 3(CWE+L), 4(CWE+LP), 5(CWE+N)\n");
    printf("\t-multi-emb <int>\n");
    printf("\t\tSet cluster number when +P and +LP; Set maximum cluster number when +N; default is 3\n");
    printf("\t-nonparametric-lambda <float>\n");
    printf("\t\tSet the boundary similarity to add a new embedding. Default is 1e-3\n");
    printf("\t-nonparametric-limit <int>\n");
    printf("\t\tSet the minimum training iters on a character vector before creating a new one, default is 1000\n");
    printf("\t-char-rate <float>\n");
    printf("\t\tSet the factor <float> of learning rate for characters, default is 1.0 (0.0 for fixed character vectors)\n");
    printf("\t-cwin <int>\n");
    printf("\t\tSet the window size used to predict character embeddings, default is 5\n");

    printf("\nExamples:\n");
    printf("./cwe -train data.txt -output-word vec.txt -output-char chr.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -iter 3\n\n");
    return 0;
  }

  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-non-comp", argc, argv)) > 0) strcpy(non_comp, argv[i + 1]);
  if ((i = ArgPos((char *)"-char-init", argc, argv)) > 0) strcpy(char_init, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-word", argc, argv)) > 0) strcpy(output_word, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-char", argc, argv)) > 0) strcpy(output_char, argv[i + 1]);

  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) dim = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if (cbow) alpha = 0.05;
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);

  if ((i = ArgPos((char *)"-cwe-type", argc, argv)) > 0) cwe_type = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-multi-emb", argc, argv)) > 0) multi_emb = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-nonparametric-lambda", argc, argv)) > 0) nonpara = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-nonparametric-limit", argc, argv)) > 0) nonpara_limit = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-char-rate", argc, argv)) > 0) char_rate = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-cwin", argc, argv)) > 0) cwin = atoi(argv[i + 1]);


  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (cwe_type == 1)
    character_size = (MAX_CHINESE - MIN_CHINESE + 1);
  else if (cwe_type == 2)
    character_size = (MAX_CHINESE - MIN_CHINESE + 1) * 4;
  else if (cwe_type == 3 || cwe_type == 5)
    character_size = (MAX_CHINESE - MIN_CHINESE + 1) * multi_emb;
  else if (cwe_type == 4)
    character_size = (MAX_CHINESE - MIN_CHINESE + 1) * 4 * multi_emb;
  printf("CWE-Type: %d\n", cwe_type);
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}

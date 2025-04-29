#  Song Lyrics Genre Classification

This project explores the use of deep learning techniques to classify song lyrics into one of three genres: **Hip-Hop**, **Pop**, or **Rock**. It demonstrates how Natural Language Processing (NLP) models can interpret text data and predict categorical outcomes using progressively more advanced architectures.

---

##  Dataset

The dataset consists of nearly **90,000 song lyrics** labeled by genre, split into training, validation, and test sets:

- `lyric_genre_train.csv`  
- `lyric_genre_val.csv`  
- `lyric_genre_test.csv`

---

##  Project Workflow

### 1. Data Preparation
- Lyrics are loaded and inspected.
- Genre labels are one-hot encoded for compatibility with neural networks.

### 2. Model 1 – Bag of Words
- Tokenizes lyrics into binary presence/absence of the top 5,000 most frequent words using `TextVectorization`.
- A shallow neural network predicts the genre from these multi-hot vectors.

### 3. Improved Model 2 – Pretrained Word Embeddings (GloVe)
- Lyrics are tokenized into padded sequences of fixed length (300).
- Integrates GloVe (100-dimensional) pretrained embeddings via an `Embedding` layer.
- Uses `GlobalAveragePooling1D` to aggregate embeddings before passing to a deeper dense network.

### 4. Fine-Tuned Embeddings
- The embedding layer is made trainable to adapt GloVe vectors to this task.
- Fine-tuning improves performance by tailoring representations to the dataset.

### 5. Prediction Interface
- Includes a helper function `lyric_predict()` for real-time genre prediction from custom input lyrics.

---

##  Key Takeaways

- Baseline unigram models outperform naive classifiers.
- Pretrained embeddings bring semantic understanding, though impact varies with data size.
- Fine-tuning embeddings generally yields better accuracy when ample data is available.

---

##  Performance

- All models are evaluated on the test set.
- Compared against a majority-class baseline (~58% accuracy).
- Performance improves as models evolve from basic to more advanced architectures.

---

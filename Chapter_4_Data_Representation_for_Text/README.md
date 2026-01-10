# Chapter 4: Data Representation for Text
Before text data can be processed by machine learning models, it needs to be converted into a numerical format. This process is called text representation or embedding. Effective text representations capture the semantic and syntactic meaning of words and sentences.

Common methods for text representation include:

*   **One-Hot Encoding:** Each word in the vocabulary is assigned a unique integer, and then represented as a binary vector where a '1' indicates the presence of the word and '0' otherwise. This method suffers from high dimensionality and doesn't capture semantic relationships.
*   **Bag-of-Words (BoW):** Represents a document as a collection of its words, disregarding grammar and even word order, but keeping track of word frequencies. It's an improvement over one-hot encoding for documents but still lacks semantic understanding.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** A statistical measure that evaluates how relevant a word is to a document in a collection of documents. It increases with the number of times a word appears in the document but is offset by the frequency of the word in the corpus.
*   **Word Embeddings (e.g., Word2Vec, GloVe, FastText):** These models learn dense vector representations (embeddings) for words where words with similar meanings are located closer to each other in the vector space. They capture semantic relationships and are much lower dimensional than one-hot encoding.
*   **Contextual Embeddings (e.g., ELMo, BERT, GPT):** More advanced models that generate word embeddings based on the context in which the word appears. This means the same word can have different embeddings depending on its usage in a sentence, capturing nuances of meaning. These models are often based on Transformer architectures.
*   **Sentence/Document Embeddings:** Methods to represent entire sentences or documents as single vectors, often by averaging word embeddings or using models specifically trained for sentence-level representations (e.g., Sentence-BERT).

The choice of text representation significantly impacts the performance of multimodal models, especially when integrating text with other modalities.

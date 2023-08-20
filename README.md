# Information-Retrieval
Developed a search engine for Persian news based on a positional inverted index and TF-IDF scoring. Also, create an index on Elasticsearch to enhance result retrieval speed.

## Phase 1
- Data Preprocessing (Normalization, Tokenization, Stemming, Removing Stopwords) using [hazm](https://github.com/roshan-research/hazm), [parsivar](https://github.com/ICTRC/Parsivar), Persian NLP toolkits
- Created a positional inverted index
- Analyzing Zipf's Law and Heaps' Law
- Searching by boolean queries (AND, NOT) and phrase queries

## Phase 2
- Computed document vectors based on TF-IDF scores
- Calculated cosine similarity between users' queries and documents
- Utilized champion list to enhance processing speed
- Ranking the results

## Phase 3
- Bulk inserting data to the ElasticSearch
- Used spelling correction, similarity modulation, and KNN classification to enhance search results

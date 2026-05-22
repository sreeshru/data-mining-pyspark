# Data Mining with PySpark

A collection of data mining projects implemented with Python and Apache Spark RDDs. This repository focuses on scalable data processing, frequent itemset mining, similarity search, and recommendation systems using Yelp-style datasets.

## Overview

This project demonstrates core data mining and big data concepts through four PySpark-based modules:

| Module | Focus Area | Key Concepts |
|---|---|---|
| `spark-data-exploration` | Distributed data exploration | Spark RDDs, MapReduce, partitioning, text preprocessing |
| `frequent-itemset-mining` | Market basket analysis | SON algorithm, A-Priori, PCY, frequent itemsets |
| `minhash-lsh-similarity` | Similarity search | MinHash, LSH, Jaccard similarity, candidate generation |
| `recommendation-systems` | Recommender systems | TF-IDF, content-based filtering, item-based collaborative filtering, Pearson similarity |

## Technologies Used

- Python 3.9
- Apache Spark / PySpark
- Spark RDD API
- JSON and CSV data processing
- Yelp-style review and business datasets

## Repository Structure

```text
data-mining-pyspark/
├── README.md
├── spark-data-exploration/
│   ├── README.md
│   ├── task1.py
│   ├── task2.py
│   ├── task3_default.py
│   └── task3_customized.py
├── frequent-itemset-mining/
│   ├── README.md
│   ├── task1.py
│   └── task2.py
├── minhash-lsh-similarity/
│   ├── README.md
│   └── task.py
└── recommendation-systems/
    ├── README.md
    ├── task1_build.py
    ├── task1_predict.py
    ├── task2_build.py
    └── task2_predict.py
```

## Project Highlights

- Built distributed data processing pipelines using Spark RDD transformations and actions.
- Implemented frequent itemset mining with SON, A-Priori, and PCY algorithms.
- Used MinHash and Locality Sensitive Hashing to reduce expensive all-pairs similarity comparisons.
- Built recommendation systems using both content-based and item-based collaborative filtering approaches.
- Applied TF-IDF, Jaccard similarity, Pearson correlation, and partition optimization techniques.

## Setup

Install Python and PySpark in a virtual environment:

```bash
conda create -n pyspark-env python=3.9
conda activate pyspark-env
pip install pyspark
```

Java is required for Spark. Java 11 is recommended.

## Notes

The original datasets are not included in this repository because of size and licensing limits. The code expects Yelp-style JSON or CSV input files in the formats described inside each module.

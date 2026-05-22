# Frequent Itemset Mining with SON, A-Priori, and PCY

This module implements frequent itemset mining using Apache Spark. It applies the SON algorithm to identify frequent itemsets from simulated and Yelp-style datasets, using A-Priori and PCY for candidate generation.

## Goals

- Find frequent itemsets in large datasets using distributed processing.
- Implement the SON algorithm with Spark RDDs.
- Use A-Priori for local candidate generation on partitions.
- Use PCY with hash buckets to optimize candidate pair generation.

## Tasks

### Task 1: Simulated Market Basket Data

This task implements two market-basket cases:

1. **Frequent businesses**: create one basket per user containing businesses reviewed by that user.
2. **Frequent users**: create one basket per business containing users who reviewed that business.

For each case, the program finds frequent singletons, pairs, triples, and larger itemsets based on a support threshold.

### Task 2: Yelp Business Basket Mining

This task builds baskets from a preprocessed user-business CSV file and finds frequent business sets. Users with fewer than a specified number of distinct reviewed businesses are filtered out before mining.

The first SON pass uses PCY on each partition to generate candidates efficiently. The second pass counts candidates across the full dataset to identify globally frequent itemsets.

## Algorithms and Concepts

- SON algorithm
- A-Priori algorithm
- PCY algorithm
- Frequent itemset mining
- Market basket analysis
- Candidate generation and pruning
- Distributed counting with Spark RDDs

## Files

```text
frequent-itemset-mining/
├── README.md
├── task1.py
└── task2.py
```

## Example Commands

```bash
python task1.py --y 2016 --r 0 --c 1 --t 7 --input_file small1.csv --output_file task1_output.json
```

```bash
python task2.py --b 4 --t 7 --input_file user_business.csv --output_file task2_output.json
```

## Output

The output JSON includes:

- Candidate itemsets
- Final frequent itemsets with support counts
- Number of candidates by itemset size
- Total number of candidates
- Runtime in seconds

## Skills Demonstrated

- Implementing scalable frequent itemset mining algorithms.
- Reducing search space using candidate pruning.
- Using Spark partitions for distributed candidate generation.
- Optimizing pair generation with PCY hash buckets.

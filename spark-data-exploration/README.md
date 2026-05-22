# Spark Data Exploration

This module uses Apache Spark RDDs to explore Yelp-style review and business datasets. The project focuses on distributed data exploration, text preprocessing, joins across datasets, and partition-level performance analysis.

## Goals

- Practice Spark RDD transformations and actions.
- Analyze review data by rating, year, month, and text content.
- Join review and business datasets to compute state-level review counts.
- Compare default Spark partitioning with a custom partitioning strategy.

## Tasks

### Task 1: Review Data Exploration

The first task analyzes a review dataset and computes:

- Overall average star rating.
- Number of reviews not matching a given year.
- Top months by review count.
- Total word count per month for reviews after a given year.
- Average review length.
- Most frequent non-stopword terms after preprocessing.

Text preprocessing includes lowercasing, removing leading/trailing punctuation, and excluding stopwords.

### Task 2: Review + Business Dataset Analysis

The second task joins review data with business data to identify the top states with the most reviews. Businesses with missing or `None` state values are filtered out before aggregation.

### Task 3: Spark Partition Analysis

The third task compares Spark's default partitioning with a customized partitioning approach. It computes review counts by year and reports:

- Number of partitions.
- Number of items per partition.
- Years with review counts above a given threshold.
- Runtime comparison between default and custom partitioning.

## Algorithms and Concepts

- Spark RDD transformations and actions
- MapReduce-style aggregation
- JSON parsing
- Text preprocessing
- Dataset joins
- Custom partitioning
- Runtime comparison

## Files

```text
spark-data-exploration/
├── README.md
├── task1.py
├── task2.py
├── task3_default.py
└── task3_customized.py
```

## Example Commands

```bash
python task1.py --input_file review.json --output_file task1.json --stopwords stopwords --t_y 2015 --m_l 5 --n 10 --i 10
```

```bash
python task2.py --review_file review.json --business_file business.json --output_file task2.json --n 10
```

```bash
python task3_default.py --input_file review.json --output_file task3_default.json --n 1000
```

```bash
python task3_customized.py --input_file review.json --output_file task3_customized.json --n_partitions 10 --n 1000
```

## Skills Demonstrated

- Writing distributed data analysis programs in PySpark.
- Cleaning and aggregating semi-structured JSON data.
- Using RDD joins and reduce operations.
- Understanding how partitioning affects Spark performance.

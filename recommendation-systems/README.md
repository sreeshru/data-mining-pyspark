# Recommendation Systems with PySpark

This module implements two recommender systems using PySpark: a content-based recommendation system and an item-based collaborative filtering recommendation system.

## Goals

- Build user and business profiles from review text using TF-IDF.
- Predict user-business preference using profile similarity.
- Compute item-item similarity using Pearson correlation.
- Predict ratings using item-based collaborative filtering.
- Evaluate recommendation quality using precision, recall, and RMSE-style metrics.

## Part 1: Content-Based Recommendation System

The content-based model uses review text to create profiles for businesses and users.

### Build Step

The build script creates TF-IDF-based business profiles:

1. Concatenate all reviews for each business into one document.
2. Clean text by removing punctuation, numbers, rare words, and stopwords.
3. Compute TF-IDF scores for words.
4. Select the top 100 words for each business profile.
5. Build each user profile as the union of business profiles for businesses the user reviewed.

### Predict Step

The prediction script loads the saved model and computes Jaccard similarity between each target user's profile and target business's profile. Pairs with similarity above the threshold are output as valid recommendations.

## Part 2: Item-Based Collaborative Filtering

The collaborative filtering model predicts ratings using similarity between businesses.

### Build Step

The build script computes business-business similarity:

1. Apply Inverse User Frequency (IUF) weighting to ratings.
2. Identify business pairs with enough co-rated users.
3. Compute Pearson similarity between business pairs.
4. Save valid business pairs, similarity scores, and co-rated user counts.

### Predict Step

The prediction script estimates ratings for target user-business pairs by using the most similar businesses that the same user has already rated. Similarities are downweighted when business pairs have fewer shared users.

## Algorithms and Concepts

- Content-based filtering
- Item-based collaborative filtering
- TF-IDF
- Jaccard similarity
- Pearson correlation
- Inverse User Frequency weighting
- Rating prediction
- RMSE-based evaluation

## Files

```text
recommendation-systems/
├── README.md
├── task1_build.py
├── task1_predict.py
├── task2_build.py
└── task2_predict.py
```

## Example Commands

### Content-Based Filtering

```bash
python task1_build.py --train_file train_review_text_150k.json --model_file task1.model --stopwords_file stopwords --time_file task1_build.time
```

```bash
python task1_predict.py --test_file val_review.json --model_file task1.model --output_file task1.val.out --time_file task1_predict.time
```

### Item-Based Collaborative Filtering

```bash
python task2_build.py --train_file train_review.json --model_file task2.model --time_file task2_build.time --m 3
```

```bash
python task2_predict.py --train_file train_review.json --test_file val_review.json --model_file task2.model --output_file task2.val.out --time_file task2_predict.time --n 3
```

## Skills Demonstrated

- Building recommender systems from text and rating data.
- Creating TF-IDF business and user profiles.
- Computing item similarity with Pearson correlation.
- Applying IUF weighting to improve collaborative filtering.
- Predicting ratings at scale using Spark RDDs.

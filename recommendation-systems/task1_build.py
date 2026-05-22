import pyspark
import argparse
import json
import math
import time
import string
from collections import Counter


def main(train_file, model_file, stopwords_file, sc):

    with open(stopwords_file, 'r') as f:
        stopwords_list = f.read().splitlines()
    stopwords_bc = sc.broadcast(set(stopwords_list))
 
    punct_digits = string.punctuation + string.digits
    punct_digits_bc = sc.broadcast(punct_digits)

    reviews = sc.textFile(train_file).map(lambda x: json.loads(x)).cache()

    def extract_words(r):
        import string as _string
        table = str.maketrans('', '', punct_digits_bc.value)
        text = r['text'].lower().translate(table)
        sw = stopwords_bc.value
        return (r['business_id'], [w for w in text.split() if w and w not in sw])
 
    business_words = reviews.map(extract_words).reduceByKey(lambda a, b: a + b).cache()

    total_word_count = business_words.flatMap(lambda x: x[1]).count()
    rare_threshold = 0.0001 * total_word_count

    valid_words_bc = sc.broadcast(
        set(
            business_words
            .flatMap(lambda x: x[1])
            .map(lambda w: (w, 1))
            .reduceByKey(lambda a, b: a + b)
            .filter(lambda x: x[1] >= rare_threshold)
            .map(lambda x: x[0])
            .collect()
        )
    )
 
    num_docs_bc = sc.broadcast(business_words.count())

    def compute_tf(pair):
        from collections import Counter as _Counter
        biz_id, words = pair
        words = [w for w in words if w in valid_words_bc.value]
        if not words:
            return (biz_id, {})
        total = len(words)
        counts = _Counter(words)
        return (biz_id, {w: cnt / total for w, cnt in counts.items()})
 
    business_tf = business_words.map(compute_tf).cache()

    word_doc_count_bc = sc.broadcast(
        dict(
            business_tf
            .flatMap(lambda x: [(w, 1) for w in x[1].keys()])
            .reduceByKey(lambda a, b: a + b)
            .collect()
        )
    )

    def compute_business_profile(pair):
        biz_id, tf = pair
        if not tf:
            return (biz_id, [])
        wdc = word_doc_count_bc.value
        n = num_docs_bc.value
        tfidf = {}
        for w, tf_val in tf.items():
            if w in wdc:
                tfidf[w] = tf_val * math.log(n / wdc[w])
        top100 = sorted(tfidf.keys(), key=lambda w: -tfidf[w])[:100]
        return (biz_id, top100)
 
    business_profiles_rdd = business_tf.map(compute_business_profile).cache()
    business_profiles_list = business_profiles_rdd.collect()

    biz_profile_bc = sc.broadcast(dict(business_profiles_list))
 
    def user_biz_to_words(pair):
        uid, bid = pair
        return (uid, set(biz_profile_bc.value.get(bid, [])))
 
    user_profiles_list = (
        reviews
        .map(lambda r: (r['user_id'], r['business_id']))
        .distinct()
        .map(user_biz_to_words)
        .reduceByKey(lambda a, b: a | b)
        .mapValues(list)
        .collect()
    )

    with open(model_file, 'w') as f:
        for biz_id, profile in business_profiles_list:
            f.write(json.dumps({"business": biz_id, "profile": profile}) + '\n')
        for user_id, profile in user_profiles_list:
            f.write(json.dumps({"user": user_id, "profile": profile}) + '\n')
            
            
            
if __name__ == '__main__':
    start_time = time.time()

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    parser = argparse.ArgumentParser(description='hw4-task2-build')
    parser.add_argument('--train_file',     type=str, default='train_review_text_150k.json')
    parser.add_argument('--model_file',     type=str, default='task1.model')
    parser.add_argument('--stopwords_file', type=str, default='stopwords')
    parser.add_argument('--time_file',      type=str, default='task1_build.time')
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.stopwords_file, sc)
    sc.stop()

    with open(args.time_file, 'w') as f:
        json.dump({'time': time.time() - start_time}, f)
    print('Duration:', time.time() - start_time)

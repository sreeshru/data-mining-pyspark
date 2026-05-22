import argparse
import json
import time
import pyspark
import math



def main(train_file, model_file, co_rated_thr, sc):
    reviews = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda r: (r['user_id'], r['business_id'], float(r['stars']))).cache()

    n_users = reviews.map(lambda r: r[0]).distinct().count()

    biz_user_count = reviews.map(lambda r: (r[1], r[0])).distinct() \
        .map(lambda x: (x[0], 1)).reduceByKey(lambda a, b: a + b)
 
    # IUF weight per business- log(n / n_j)
    biz_iuf_bc = sc.broadcast(
        dict(biz_user_count.map(lambda x: (x[0], math.log(n_users / x[1]))).collect())
    )
 
    user_ratings = reviews.map(lambda r: (
        r[0], (r[1], r[2] * biz_iuf_bc.value.get(r[1], 1.0))
    )).groupByKey().mapValues(list).cache()

    def gen_pairs(user_ratings_pair):
        _, biz_ratings = user_ratings_pair
        biz_ratings = list(biz_ratings)
        pairs = []
        for i in range(len(biz_ratings)):
            for j in range(i + 1, len(biz_ratings)):
                b1, r1 = biz_ratings[i]
                b2, r2 = biz_ratings[j]
                if b1 > b2:
                    b1, b2 = b2, b1
                    r1, r2 = r2, r1
                pairs.append(((b1, b2), (r1, r2)))
        return pairs
 
    pair_ratings = user_ratings.flatMap(gen_pairs)

    def create_combiner(v):
        r1, r2 = v
        return (r1, r2, r1*r1, r2*r2, r1*r2, 1)
 
    def merge_value(acc, v):
        r1, r2 = v
        return (acc[0]+r1, acc[1]+r2, acc[2]+r1*r1, acc[3]+r2*r2, acc[4]+r1*r2, acc[5]+1)
 
    def merge_combiners(a, b):
        return (a[0]+b[0], a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4], a[5]+b[5])
 
    pair_stats = pair_ratings.combineByKey(create_combiner, merge_value, merge_combiners)

    def pearson_from_stats(item):
        (b1, b2), (s1, s2, s1sq, s2sq, s12, n) = item
        if n < 1:
            return None
        mean1 = s1 / n
        mean2 = s2 / n
        num = s12 - n * mean1 * mean2
        den1 = math.sqrt(max(0.0, s1sq - n * mean1 * mean1))
        den2 = math.sqrt(max(0.0, s2sq - n * mean2 * mean2))
        if den1 == 0 or den2 == 0:
            sim = 0.0
        else:
            sim = num / (den1 * den2)
        return (b1, b2, sim, n)
 
    results = pair_stats \
        .map(pearson_from_stats) \
        .filter(lambda x: x is not None and x[3] >= co_rated_thr and x[2] >= 0.3)
 
    with open(model_file, 'w') as f:
        for b1, b2, sim, num_co in results.collect():
            f.write(json.dumps({"b1": b1, "b2": b2, "sim": sim, "num_co_rated": num_co}) + '\n')
 



if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_build') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='hw4-task2-build')
    parser.add_argument('--train_file', type=str, default='train_review.json')
    parser.add_argument('--model_file', type=str, default='task2.model')
    parser.add_argument('--time_file',  type=str, default='task2_build.time')
    parser.add_argument('--m',          type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.model_file, args.m, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))

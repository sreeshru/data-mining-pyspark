import argparse
import json
import time
import random
from itertools import combinations
import pyspark

def main(input_file, candidate_file, output_file, jac_thr, seed, sc):
    random.seed(int(seed))

    num_hashes = 100
    bands = 50
    rows_per_band = 2

    def is_prime(x):
        if x < 2:
            return False
        for i in range(2, int(x ** 0.5) + 1):
            if x % i == 0:
                return False
        return True

    def get_next_prime(x):
        while not is_prime(x):
            x += 1
        return x

    def jaccard(set1, set2):
        inter = len(set1 & set2)
        union = len(set1 | set2)
        if union == 0:
            return 0
        return inter / union

    # read input
    data = sc.textFile(input_file).map(lambda x: json.loads(x))

    # keep only (user_id and business_id) remove duplicates
    user_business = data.map(lambda x: (x["user_id"], x["business_id"])).distinct().cache()

    # assign each business an index
    business_list = user_business.map(lambda x: x[1]).distinct().collect()
    business_list.sort()

    business_index = {}
    for i in range(len(business_list)):
        business_index[business_list[i]] = i

    business_index_bc = sc.broadcast(business_index)
    num_businesses = len(business_list)

    # edge case
    if num_businesses == 0:
        with open(candidate_file, "w") as f:
            pass
        with open(output_file, "w") as f:
            pass
        return

    # group businesses by user
    user_biz = (
        user_business
        .map(lambda x: (x[0], business_index_bc.value[x[1]]))
        .groupByKey()
        .mapValues(lambda x: set(x))
        .cache()
    )

    # parameters for double hashing
    P = get_next_prime(num_businesses * 10)
    a1 = random.randint(1, P - 1)
    b1 = random.randint(0, P - 1)
    a2 = random.randint(1, P - 1)
    b2 = random.randint(0, P - 1)

    def make_signature(business_set):
        sig = []

        for i in range(num_hashes):
            min_val = float("inf")

            for x in business_set:
                h1 = ((a1 * x + b1) % P)
                h2 = ((a2 * x + b2) % P)
                h = (h1 + i * h2) % num_businesses

                if h < min_val:
                    min_val = h

            sig.append(min_val)

        return sig

    # compute signatures
    signatures = user_biz.mapValues(make_signature)

    # LSH - create buckets for each band
    band_buckets = signatures.flatMap(
        lambda x: [
            ((band_num, tuple(x[1][band_num * rows_per_band:(band_num + 1) * rows_per_band])), x[0])
            for band_num in range(bands)
        ]
    )

    # get candidate pairs
    candidate_pairs = (
        band_buckets
        .groupByKey()
        .mapValues(lambda x: list(set(x)))
        .filter(lambda x: len(x[1]) > 1)
        .flatMap(
            lambda x: [
                tuple(sorted(pair))
                for pair in combinations(x[1], 2)
            ]
        )
        .distinct()
        .sortBy(lambda x: (x[0], x[1]))
        .cache()
    )

    # write candidate file
    candidate_result = candidate_pairs.collect()
    with open(candidate_file, "w") as f:
        for u1, u2 in candidate_result:
            f.write(json.dumps({"u1": u1, "u2": u2}) + "\n")

    # collect user business sets for exact jaccard
    user_biz_dict = user_biz.collectAsMap()
    user_biz_bc = sc.broadcast(user_biz_dict)

    # compute final similar pairs
    final_pairs = (
        candidate_pairs
        .map(lambda x: (x[0], x[1], jaccard(user_biz_bc.value[x[0]], user_biz_bc.value[x[1]])))
        .filter(lambda x: x[2] > jac_thr)
        .sortBy(lambda x: (x[0], x[1]))
        .collect()
    )
    # write output file
    with open(output_file, "w") as f:
        for u1, u2, sim in final_pairs:
            f.write(json.dumps({"u1": u1, "u2": u2, "sim": sim}) + "\n")
    

if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw3') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    parser = argparse.ArgumentParser(description='A3')
    parser.add_argument('--input_file', type=str, default='./data/tr1.json')
    parser.add_argument('--candidate_file', type=str, default='./outputs/candidate.out')
    parser.add_argument('--output_file', type=str, default='./outputs/task.out')
    parser.add_argument('--time_file', type=str, default='./outputs/task.time')
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--seed', type=float, default=0)
    args = parser.parse_args()

    main(args.input_file, args.candidate_file, args.output_file, 
         args.threshold, args.seed , sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))

    




    
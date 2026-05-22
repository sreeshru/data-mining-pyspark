import pyspark
import argparse
import json
import math
import time


def main(test_file, model_file, output_file, sc):
    
    business_profiles = {}
    user_profiles = {}
 
    with open(model_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            if 'business' in record and 'profile' in record:
                business_profiles[record['business']] = set(record['profile'])
            elif 'user' in record and 'profile' in record:
                user_profiles[record['user']] = set(record['profile'])
            elif len(record) == 1:
                item_id, profile = next(iter(record.items()))
                if item_id.startswith('u'):
                    user_profiles[item_id] = set(profile)
                else:
                    business_profiles[item_id] = set(profile)
 
    biz_profile_bc = sc.broadcast(business_profiles)
    user_profile_bc = sc.broadcast(user_profiles)
 
    # Load test pairs
    test_pairs = sc.textFile(test_file).map(lambda x: json.loads(x)) \
        .map(lambda r: (r['user_id'], r['business_id']))
 
    def compute_jaccard(pair):
        user_id, biz_id = pair
        user_set = user_profile_bc.value.get(user_id, set())
        biz_set = biz_profile_bc.value.get(biz_id, set())
        if not user_set or not biz_set:
            return (user_id, biz_id, 0.0)
        intersection = len(user_set & biz_set)
        union = len(user_set | biz_set)
        sim = intersection / union if union > 0 else 0.0
        return (user_id, biz_id, sim)
 
    results = test_pairs.map(compute_jaccard) \
        .filter(lambda x: x[2] >= 0.01)
 
    with open(output_file, 'w') as f:
        for user_id, biz_id, sim in results.collect():
            f.write(json.dumps({"user_id": user_id, "business_id": biz_id, "sim": sim}) + '\n')

if __name__ == '__main__':
    start_time = time.time()

    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_task1') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel('OFF')

    parser = argparse.ArgumentParser(description='hw4-task1-predict')
    parser.add_argument('--test_file',   type=str, default='val_review.json')
    parser.add_argument('--model_file',  type=str, default='task1.model')
    parser.add_argument('--output_file', type=str, default='task1.val.out')
    parser.add_argument('--time_file',   type=str, default='task1_predict.time')
    args = parser.parse_args()

    main(args.test_file, args.model_file, args.output_file, sc)
    sc.stop()

    with open(args.time_file, 'w') as f:
        json.dump({'time': time.time() - start_time}, f)
    print('Duration:', time.time() - start_time)

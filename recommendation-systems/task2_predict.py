

import argparse
import json
import time
import pyspark



def main(train_file, test_file, model_file, output_file, n_weights, sc):
    biz_sim = {}
    with open(model_file, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            b1, b2, sim, num_co = record['b1'], record['b2'], record['sim'], record['num_co_rated']
            if b1 not in biz_sim:
                biz_sim[b1] = []
            if b2 not in biz_sim:
                biz_sim[b2] = []
            biz_sim[b1].append((b2, sim, num_co))
            biz_sim[b2].append((b1, sim, num_co))
 
    biz_sim_bc = sc.broadcast(biz_sim)

    train_reviews = sc.textFile(train_file).map(lambda x: json.loads(x)) \
        .map(lambda r: (r['user_id'], r['business_id'], float(r['stars'])))

    user_biz_ratings = train_reviews.map(lambda r: (r[0], (r[1], r[2]))) \
        .groupByKey().mapValues(dict)
 
    user_ratings_bc = sc.broadcast(dict(user_biz_ratings.collect()))

    test_pairs = sc.textFile(test_file).map(lambda x: json.loads(x)) \
        .map(lambda r: (r['user_id'], r['business_id']))
 
    def predict_rating(pair):
        user_id, target_biz = pair
        user_ratings = user_ratings_bc.value.get(user_id, {})
        biz_similarities = biz_sim_bc.value.get(target_biz, [])
 
        candidates = []
        for other_biz, sim, num_co in biz_similarities:
            if other_biz in user_ratings:
                
                adjusted_sim = sim * min(1.0, num_co / 50)
                candidates.append((adjusted_sim, user_ratings[other_biz]))

        candidates.sort(key=lambda x: -abs(x[0]))
        candidates = candidates[:n_weights]
 
        if not candidates:
            if user_ratings:
                pred = sum(user_ratings.values()) / len(user_ratings)
            else:
                pred = 3.0
        else:
            num = sum(sim * rating for sim, rating in candidates)
            den = sum(abs(sim) for sim, _ in candidates)
            if den == 0:
                pred = sum(r for _, r in candidates) / len(candidates)
            else:
                pred = num / den

        pred = max(1.0, min(5.0, pred))
        return (user_id, target_biz, pred)
 
    predictions = test_pairs.map(predict_rating)
 
    with open(output_file, 'w') as f:
        for user_id, biz_id, stars in predictions.collect():
            f.write(json.dumps({"user_id": user_id, "business_id": biz_id, "stars": stars}) + '\n')
 




if __name__ == '__main__':
    start_time = time.time()
    sc_conf = pyspark.SparkConf() \
        .setAppName('hw4_predict') \
        .setMaster('local[*]') \
        .set('spark.driver.memory', '4g') \
        .set('spark.executor.memory', '4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")


    parser = argparse.ArgumentParser(description='hw4-task2-predict')
    parser.add_argument('--train_file', type=str, default='train_review.json')
    parser.add_argument('--test_file', type=str, default='val_review.json')
    parser.add_argument('--model_file', type=str, default='task2.model')
    parser.add_argument('--output_file', type=str, default='task2.val.out')
    parser.add_argument('--time_file', type=str, default='task2_predict.time')
    parser.add_argument('--n', type=int, default=3)
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.model_file, args.output_file, args.n, sc)
    sc.stop()

    # log time
    with open(args.time_file, 'w') as outfile:
        json.dump({'time': time.time() - start_time}, outfile)
    print('The run time is: ', (time.time() - start_time))



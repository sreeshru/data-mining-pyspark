import argparse
import pyspark
import json

def main(args):
    sc_conf = pyspark.SparkConf().setAppName('task2').setMaster('local[*]').set('spark.driver.memory','8g').set('spark.executor.memory','4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    
    reviews = sc.textFile(args.review_file).map(lambda line: json.loads(line))
    review_pairs = reviews.map(lambda d: (d["business_id"], 1))
    
    business = sc.textFile(args.business_file).map(lambda line:json.loads(line))
    business_pairs = business.map(lambda d: (d["business_id"], d["state"]))
    
    business_pairs = business_pairs.filter(lambda x: x[1] and x[1].strip())
    joined = review_pairs.join(business_pairs)
    
    state_counts = joined.map(lambda x: (x[1][1], 1))
    state_counts = state_counts.reduceByKey(lambda a, b: a + b)
    
    sorted_states = state_counts.sortBy(lambda x: (-x[1], x[0])).take(args.n)
    result = [[state,count] for state, count in sorted_states]
    
    output = {"result" : result}
    
    with open(args.output_file, "w") as f:
        json.dump(output,f)

    sc.stop()
    
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='A1T2')
    parser.add_argument('--review_file', type=str, default='./data/review.json', help='review file')
    parser.add_argument('--business_file', type=str, default='./data/business.json', help='business file')
    parser.add_argument('--output_file', type=str, default='./a1t2.json', help='the output file contains your answers')
    parser.add_argument('--n', type=int, default=5, help='top n states by number of reviews')
    args = parser.parse_args()
    
    main(args)


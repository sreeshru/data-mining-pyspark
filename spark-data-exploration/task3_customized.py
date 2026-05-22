import pyspark
import json
import argparse




def main(args):
    sc_conf = pyspark.SparkConf().setAppName('task3_c').setMaster('local[*]').set('spark.driver.memory','8g').set('spark.executor.memory','4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")

    def custom_hash_func(key):
        return int(key) % args.n_partitions
    
    reviews = sc.textFile(args.input_file).map(lambda line: json.loads(line))
    year_pairs = reviews.map(lambda d: (d["date"][:4], 1))
    
    partitioned = year_pairs.partitionBy(args.n_partitions, custom_hash_func)
    
    year_counts = partitioned.reduceByKey(lambda a, b: a+b)
    filtered = year_counts.filter(lambda x: x[1] > args.n)
    num_partitions = partitioned.getNumPartitions()
    items_per_part = partitioned.mapPartitions(lambda it: [sum(1 for _ in it)]).collect()

    output = {
        "n_partitions": num_partitions,
        "n_items": items_per_part,
        "result": filtered.collect()
    }

    with open(args.output_file, "w") as f:
        json.dump(output, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A1T3_customized')
    parser.add_argument('--input_file', type=str, default='./data/review.json', help='review file')
    parser.add_argument('--output_file', type=str, default='./a1t3_customized.json', help='the output file contains your answers')
    parser.add_argument('--n', type=int, default=10000, help='the review count threshold per year')
    parser.add_argument('--n_partitions', type=int, default=10, help='the number of partitions ')
    args = parser.parse_args()
    main(args)



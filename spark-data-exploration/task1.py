import argparse
import pyspark
import json

def clean(text, stopwords):
    punct = "([,.!?:;])"
    words = text.lower().split()
    cleaned = []
    for w in words:
        while w and w[0] in punct:
            w = w[1:]
        while w and w[-1] in punct:
            w = w[:-1]
        if w and w not in stopwords:
            cleaned.append(w)
    return cleaned
def main(args):
    sc_conf = pyspark.SparkConf().setAppName('task1').setMaster('local[*]').set('spark.driver.memory','8g').set('spark.executor.memory','4g')
    sc = pyspark.SparkContext(conf=sc_conf)
    sc.setLogLevel("OFF")


    rdd = sc.textFile(args.input_file)
    reviews = rdd.map(lambda line: json.loads(line))
    parsed = reviews.map(lambda d: (d["stars"], d["text"], d["date"][:4], d["date"][5:7]))
    
    #average star rating across all reviews
    stars_rdd = parsed.map(lambda x: x[0])
    total_stars = stars_rdd.reduce(lambda a, b: a + b)
    count_stars = stars_rdd.count()
    average = total_stars/count_stars
    
    #number of reviews whose year is NOT equal to the given year t_y
    reviewsNOTequal = parsed.filter(lambda x: x[2] != str(args.t_y)).count()
    
    with open(args.stopwords) as f:
        stopwords = set([w.strip().lower() for w in f])
        
    
    #top N months with most reviews
    month_counts = parsed.map(lambda x: (x[3], 1)).reduceByKey(lambda a,b: a+b)
    topR_list = month_counts.sortBy(lambda x: (-x[1], -int(x[0]))).take(args.n)
    C = [[month, count] for month, count in topR_list]
    
    #total word count for year >= t_y
    after_year = parsed.filter(lambda x: int(x[2]) >= args.t_y)
    month_wordcounts = after_year.map(lambda x: (x[3], len(clean(x[1], stopwords))))
    D_list = month_wordcounts.reduceByKey(lambda a, b: a + b) \
                             .sortBy(lambda x: x[0]) \
                             .collect()
    D = [[month, count] for month, count in D_list]
    
    #average review length with year >= t_y
    word_counts = after_year.map(lambda x: len(clean(x[1], stopwords)))
    total_words = word_counts.reduce(lambda a, b: a+b)
    num_reviews = word_counts.count()
    E = total_words/ num_reviews
    
    #top i most frequent words with length greater than minimum length m_l
    all_words = after_year.flatMap(lambda x: clean(x[1], stopwords))
    filtered_words = all_words.filter(lambda w: len(w) > args.m_l)
    
    freqs = filtered_words.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)
    top_words = freqs.sortBy(lambda x: (-x[1], x[0])).take(args.i)
    F = [word for word, count in top_words]
    
    output = {
        "A": average,
        "B": reviewsNOTequal,
        "C": C,
        "D": D,
        "E": E,
        "F": F
    }
    with open(args.output_file, "w") as f:
        json.dump(output, f)
    
    sc.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW1T1')
    parser.add_argument('--input_file', type=str, default='./data/review.json', help='input file')
    parser.add_argument('--output_file', type=str, default='./a1t1.json', help='output file')
    parser.add_argument('--stopwords', type=str, default='./data/stopwords', help='stopword file')
    parser.add_argument('--t_y', type=int, default=2015, help='year')
    parser.add_argument('--m_l', type=int, default=3, help='minimum word length')
    parser.add_argument('--n', type=int, default=5, help='top n months')
    parser.add_argument('--i', type=int, default=10, help='top i frequent words')

    args = parser.parse_args()
    main(args)

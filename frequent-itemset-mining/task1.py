import sys
import time
import argparse
import json
import math
from itertools import islice, combinations
import pyspark
from pyspark import SparkContext

# Function for phase 2 of the SON algorithm
def validateCandidates(iterator, flat_candidates):
    candidate_counts = {}
    for basket in iterator:
        basket_set = set(basket)
        for candidate in flat_candidates:
            candidate_tuple = tuple(sorted(candidate))
            if set(candidate_tuple).issubset(basket_set):
                candidate_counts[candidate_tuple] = candidate_counts.get(candidate_tuple, 0) + 1
    return list(candidate_counts.items())
    
    

#Function for use in step 4 of aPriori algorithm    
def getCandidatesForNextSize(freq, nextSize):
    """
    Generate candidate itemsets for the next iteration:
    1. Take frequent itemsets from the current iteration
    2. Join pairs of frequent itemsets to create larger candidates
    3. Apply the a-priori property to prune invalid candidates
    4. Return valid candidate itemsets of the specified size
    """
    if not freq:
        return []
    
    freq_set = set(freq)
    
    if nextSize == 2:
        items = sorted(set(item for itemset in freq for item in itemset))
        return [(a, b) for a, b in combinations(items, 2)
                if (a,) in freq_set and (b,) in freq_set]
    
    # For sizes > 2, use the A-Priori join property
    prev_freq = sorted([list(itemset) for itemset in freq if len(itemset) == nextSize - 1])
    if len(prev_freq) < 2:
        return []

    candidates = []
    for i in range(len(prev_freq)):
        for j in range(i + 1, len(prev_freq)):
            i1, i2 = prev_freq[i], prev_freq[j]
            if i1[:-1] == i2[:-1]:
                candidate = tuple(sorted(set(i1) | {i2[-1]}))
                if len(candidate) == nextSize:
                    if all(tuple(sorted(s)) in freq_set for s in combinations(candidate, nextSize - 1)):
                        candidates.append(candidate)

    return list(set(candidates))

def aPriori(iterator, totalCount, threshold):
    """
    Implement the A-Priori algorithm:
    1. Process baskets to calculate partition-specific threshold
    2. Generate singleton itemsets from the data
    3. Identify frequent itemsets by counting and comparing to threshold
    4. Generate candidates for next iteration using frequent itemsets
    5. Validate each new set of candidates against the data
    6. Repeat until no new frequent itemsets are found
    7. Return the complete set of frequent itemsets
    """
    baskets = list(iterator)
    if not baskets:
        return []
    part_size = len(baskets)
    # part_threshold = math.ceil(threshold * part_size / totalCount * 1.1)  
    part_threshold = threshold * (part_size / totalCount)  
    #singletons
    item_counts = {}
    for basket in baskets:
        for item in basket:
            item_counts[item] = item_counts.get(item, 0) + 1
            
    freq = [(item,) for item, count in item_counts.items() if count >= part_threshold]
    all_freq = set(freq)
    
    k = 2
    while freq:
        candidates = getCandidatesForNextSize(freq, k)
        if not candidates:  # Stop if no candidates generated
            break
            
        candidate_counts = {}
        for basket in baskets:
            basket_set = set(basket)
            for candidate in candidates:
                if set(candidate).issubset(basket_set):
                    candidate_counts[candidate] = candidate_counts.get(candidate, 0) + 1
        
        freq = [candidate for candidate, count in candidate_counts.items() if count >= part_threshold]
        if not freq:  # Stop if no frequent itemsets found
            break
            
        all_freq.update(freq)
        k += 1
        
    return list(all_freq)



def main(rdd, case, threshold, outputJson, year_filter, rate_filter, time0):
    """
    Main function controlling the workflow and implementing the SON algo:
    1. Preprocess data (remove header, filter by year, ect)
    2. Convert data into baskets based on case
    3. Apply the SON algorithm with A-Priori
    4. Output results
    """
 
    out = {}
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)

    # split columns
    data = rdd.map(lambda x: x.split(',')) \
              .filter(lambda x: int(x[0]) == year_filter)
    if rate_filter != 0:
        data = data.filter(lambda x: int(x[3]) > rate_filter)
    
    if case == 1:
        baskets = data.map(lambda x: (x[1], x[2]))
    else:
        baskets = data.map(lambda x: (x[2], x[1]))
    
    baskets = baskets.groupByKey().map(lambda x: list(set(x[1]))).cache()
    total_count = baskets.count()
    
    #son phase one
    local_frequents = baskets.mapPartitions(
        lambda part: aPriori(part, total_count, threshold)
    ).distinct().collect()

    # Organize candidates by size
    candidates_by_size = {}
    for c in local_frequents:
        candidate_list = sorted(list(c)) if isinstance(c, tuple) else [c]
        size = len(candidate_list)
        if size not in candidates_by_size:
            candidates_by_size[size] = []
        candidates_by_size[size].append(candidate_list)

    # Sort candidates within each size
    for size in candidates_by_size:
        candidates_by_size[size].sort(key=lambda x: [int(i) for i in x])

    candidates_output = [candidates_by_size[size] for size in sorted(candidates_by_size.keys())]
    total_candidates = sum(len(v) for v in candidates_by_size.values())

    out['Candidates'] = candidates_output
    out['Num Candidates By Size'] = {str(k): len(v) for k, v in sorted(candidates_by_size.items())}
    out['Num Candidates Total'] = total_candidates


    #son phase two
    #Flatten candidates for phase 2
    flat_candidates = [item for group in candidates_output for item in group]
    
    counts = baskets.mapPartitions(
        lambda part: validateCandidates(part, flat_candidates)
    ).reduceByKey(lambda a, b: a + b) \
    .filter(lambda x: x[1] >= threshold) \
    .collect()

    # Organize frequent itemsets by size (with counts)
    # Change how frequent_by_size is built (remove the counts)
    frequent_by_size = {}
    for itemset_tuple, count in counts:
        itemset_list = sorted(list(itemset_tuple), key=lambda x: int(x))
        size = len(itemset_list)
        
        if size not in frequent_by_size:
            frequent_by_size[size] = []
        frequent_by_size[size].append(itemset_list)

    out['Frequent Itemsets'] = [sorted(frequent_by_size[s]) for s in sorted(frequent_by_size.keys())]
  
    time1 = time.time()
    out['Runtime'] = time1 - time0
    with open(outputJson, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HW2T1')	
    parser.add_argument('--y', type=int, default=2016, help ='Filter year')
    parser.add_argument('--c', type=int, default=1, help ='case number')
    parser.add_argument('--r', type=int, default=0, help ='filter review rate')
    parser.add_argument('--t', type=int, default=10, help ='frequent threshold')
    parser.add_argument('--input_file', type=str, default='../data/small2.csv', help ='input file')
    parser.add_argument('--output_file', type=str, default='./HW2task1.json', help ='output  file')
    
    args = parser.parse_args()
    case = args.c
    threshold = args.t
    inputJson = args.input_file
    outputJson = args.output_file
    year_filter = args.y
    time0 = time.time()

	# Read Input
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    rdd = sc.textFile(inputJson)
    main(rdd, case, threshold, outputJson, year_filter, args.r, time0)
    sc.stop()


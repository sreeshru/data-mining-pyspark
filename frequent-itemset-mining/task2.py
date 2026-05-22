import sys
import time
import argparse
import json
from itertools import islice, combinations
import pyspark
from pyspark import SparkContext

# Function for phase 2 of the SON algorithm
def validateCandidates(iterator, flat_candidates):
    """
    Count occurrences of candidate itemsets in the data
    """
    candidates_set = {tuple(candidate) for candidate in flat_candidates}
    local_counts = {}
    for basket in iterator:
        #converting basket to set for subset operations
        basket_set = set(basket)
        
        for candidate in candidates_set:
            candidate_set = set(candidate)
            if candidate_set.issubset(basket_set):
                local_counts[candidate] = local_counts.get(candidate, 0) + 1
            
    for itemset, count in local_counts.items():
        yield (itemset, count)

# Function for use in step 3 of aPriori algorithm
def getCandidatesForNextSize(freq, nextSize, subThreshold, hashTable, hashSize):
    """
    Generate candidate itemsets for the next iteration:
    1. Take frequent itemsets from the current iteration
    2. Join pairs of frequent itemsets to create larger candidates
    3. Apply the a-priori property to prune invalid candidates
    4. Return valid candidate itemsets of the specified size
    """
    if nextSize == 2:
        #for size 2 using hash tables to filter candidates
        candidates = []
        freq_list = sorted(list(freq))
        
        #all possible pairs from sinhletons
        for i in range(len(freq_list)):
            for j in range (i + 1, len(freq_list)):
                pair = (freq_list[i], freq_list[j])
                #hash pair and check
                hash_val = hash(pair) % hashSize
                if hashTable[hash_val] >= subThreshold:
                    candidates.append(pair)        
        return candidates
    
    else:
        #for size > 2 using a-priori 
        if not freq:
            return []
        
        freq_tuples = {tuple(sorted(itemset)) for itemset in freq}
        freq_list = [list(itemset) for itemset in sorted(freq_tuples)]
        candidates = []
        
        #generate candidates by joining pairs of frequent itemsets
        for i in range(len(freq_list)):
            for j in range(i + 1, len(freq_list)):
                itemset1 = freq_list[i]
                itemset2 = freq_list[j]
                #check if first (k-2) elements are same
                if itemset1[:-1] == itemset2[:-1]:
                    candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                    #check if cand has the correct size
                    if len(candidate) == nextSize:
                        #check all subsets
                        valid = True
                        for subset in combinations(candidate, nextSize - 1):
                            if tuple(sorted(subset)) not in freq_tuples:
                                valid = False
                                break
                        if valid:
                            candidates.append(candidate)
        return list(set(candidates))
                        
def pcy(iterator, totalCount, threshold, hashSize):
   """
   Implement the PCY (Park-Chen-Yu) algorithm:
   1. Process baskets to calculate partition-specific threshold
   2. For singleton pass, hash item pairs into a hash table
   3. Use hash table to identify potentially frequent pairs
   4. Generate candidates with a-priori pruning and hash-based filtering
   5. Validate candidates against data in each iteration
   6. Continue until no new frequent itemsets are found
   7. Return all frequent itemsets discovered
   
   HINT: The Major bottleneck for this data is the processing of the size-2 candidates. It is recomended you use a hashtable instead of a dictionary to store their counts. 
   During step 2 you can also create size-2 combinations from each basket, and hash the pair by index = hash(pair)%hashTableSize, and increment the corresponding entry by 1  hashTable[index]+=1
   """
   baskets = list(iterator)
   partition_size = len(baskets)
   
   if partition_size == 0:
       return set()
   
   # calculate partition-specific threshold
   part_support = (threshold * partition_size) / totalCount
   
   # P1: count singletons and build ht for pairs
   item_counts = {}
   hash_table = [0] * hashSize
   
   # process each basket
   for basket in baskets:
       basket_list = list(basket)
       
       # count singletons
       for item in basket_list:
           item_counts[item] = item_counts.get(item, 0) + 1
       
       # Hash all pairs (size-2 combinations)
       basket_sorted = sorted(basket_list)
       for i in range(len(basket_sorted)):
           for j in range(i + 1, len(basket_sorted)):
               pair = (basket_sorted[i], basket_sorted[j])
               hash_val = hash(pair) % hashSize
               hash_table[hash_val] += 1
   
   # find frequent singletons
   frequent_singletons = set()
   for item, count in item_counts.items():
       if count >= part_support:
           frequent_singletons.add(item)
   
   if not frequent_singletons:
       return set()
   
   # initialize frequent itemsets with singletons 
   frequent_singleton_tuples = {tuple([item]) for item in frequent_singletons}
   frequent_itemsets = [frequent_singleton_tuples]
   all_frequent = set()
   all_frequent.update(frequent_singleton_tuples)
   
   # Generate candidates for size 2 using PCY
   size = 2
   current_frequent = frequent_singleton_tuples
   
   while current_frequent:
       candidates = []
       
       if size == 2:
           # Use PCY for pairs
           candidates = getCandidatesForNextSize(
               frequent_singletons, size, part_support, hash_table, hashSize
           )
       else:
           current_frequent_filtered = [itemset for itemset in current_frequent if len(itemset) == size - 1]
           
           if not current_frequent_filtered:
               break
               
           candidates = getCandidatesForNextSize(current_frequent_filtered, size, part_support, hash_table, hashSize)
       
       if not candidates:
           break
       
       # count candidates in this partition
       candidate_counts = {}
       for basket in baskets:
           basket_set = set(basket)
           for candidate in candidates:
               candidate_set = set(candidate)
               if candidate_set.issubset(basket_set):
                   candidate_tuple = tuple(sorted(candidate))
                   candidate_counts[candidate_tuple] = candidate_counts.get(candidate_tuple, 0) + 1
       
       # find frequent candidates
       new_frequent = set()
       for candidate_tuple, count in candidate_counts.items():
           if count >= part_support:
               new_frequent.add(candidate_tuple)
       
       if new_frequent:
           frequent_itemsets.append(new_frequent)
           all_frequent.update(new_frequent)
           current_frequent = new_frequent
           size += 1
       else:
           break
   
   return all_frequent

# Main function to orchestrate the workflow
def main(rdd, filter_threshold, support_threshold, outputJson, hashSize):
    """
    Main function controlling the workflow and implementing the SON algo:
    1. Preprocess data (remove header, ect)
    2. Build Case 1 market-basket model (user -> businesses)
    3. Filter users who reviewed more than filter_threshold businesses
    4. Apply the SON algorithm with pcy
    5. Output results
    """
	# write answer into a dictionary
    out = {}
    '''
    
    YOUR CODE HERE
    
    '''
    # Remove header
    header = rdd.first()
    data_rdd = rdd.filter(lambda line: line != header)
    
    parsed_rdd = data_rdd.map(lambda line: line.strip().split(',')) \
                         .map(lambda parts: (parts[0], parts[1]))
    
    user_baskets = parsed_rdd.distinct() \
                            .groupByKey() \
                            .mapValues(list) \
                            .filter(lambda x: len(x[1]) > filter_threshold) \
                            .map(lambda x: x[1])  
    
    user_baskets.cache()
    
    # get total count for partition threshold calculation
    total_baskets = user_baskets.count()
    
    if total_baskets == 0:
        #empty case
        out["Candidates"] = []
        out["Frequent Itemsets"] = []
        out["Num Candidates By Size"] = {}
        out["Num Candidates Total"] = 0
    else:
        # First pass - find candidates in each partition using PCY
        candidates_rdd = user_baskets.mapPartitions(
            lambda iterator: [pcy(iterator, total_baskets, support_threshold, hashSize)]
        )
        
        # Collect all unique candidates
        all_candidates = candidates_rdd.flatMap(lambda x: x).distinct().collect()
        
        # organize candidates by size
        candidates_by_size = {}
        for candidate_tuple in all_candidates:
            size = len(candidate_tuple)
            if size not in candidates_by_size:
                candidates_by_size[size] = []
            
            # convert to list for output
            candidate_list = list(candidate_tuple)
            if candidate_list not in candidates_by_size[size]:
                candidates_by_size[size].append(candidate_list)
        
        # sort candidates within each size
        for size in candidates_by_size:
            candidates_by_size[size].sort()
        
        candidates_list = []
        for size in sorted(candidates_by_size.keys()):
            candidates_list.extend([tuple(cand) for cand in candidates_by_size[size]])
        
        # broadcast candidates to all workers
        candidates_broadcast = sc.broadcast(candidates_list)
        
        # Second pass - count global frequencies
        counts_rdd = user_baskets.mapPartitions(
            lambda iterator: validateCandidates(iterator, candidates_broadcast.value)
        )
        
        # reduce to get global counts
        global_counts = counts_rdd.reduceByKey(lambda a, b: a + b).collect()
        
        # Filter globally frequent itemsets
        frequent_itemsets = []
        frequent_by_size = {}
        
        for itemset_tuple, count in global_counts:
            if count >= support_threshold:
                size = len(itemset_tuple)
                itemset_list = list(itemset_tuple)
                frequent_by_size[size] = frequent_by_size.get(size, []) + [(itemset_list, count)]
        
        # Sort frequent itemsets by size and lexicographically
        for size in sorted(frequent_by_size.keys()):
            frequent_by_size[size].sort(key=lambda x: x[0])
            frequent_itemsets.extend(frequent_by_size[size])
        
        # Build candidates list in the required format - list of lists by size
        candidates_output = []
        for size in sorted(candidates_by_size.keys()):
            candidates_output.append(candidates_by_size[size])

        # Format frequent itemsets by size
        frequent_by_size_formatted = {}
        for itemset, count in frequent_itemsets:
            size = len(itemset) if isinstance(itemset, list) else 1
            if size not in frequent_by_size_formatted:
                frequent_by_size_formatted[size] = []
            
            if size == 1:
                frequent_by_size_formatted[size].append((itemset, count))
            else:
                frequent_by_size_formatted[size].append((itemset, count))

        # Build frequent itemsets list in the required format
        frequent_output = []
        for size in sorted(frequent_by_size_formatted.keys()):
            size_list = []
            if size == 1:
                # Sort singletons by item
                frequent_by_size_formatted[size].sort(key=lambda x: x[0])
                for item, count in frequent_by_size_formatted[size]:
                    size_list.append((item, count))
            else:
                # Sort larger itemsets lexicographically
                frequent_by_size_formatted[size].sort(key=lambda x: (x[0], x[1]))
                for itemset, count in frequent_by_size_formatted[size]:
                    size_list.append((itemset, count))
            frequent_output.append(size_list)

        # Calculate total candidates
        total_candidates = sum(len(v) for v in candidates_by_size.values())

        # Build output dictionary
        out["Candidates"] = candidates_output
        out["Frequent Itemsets"] = frequent_output
        out["Num Candidates By Size"] = {str(k): len(v) for k, v in sorted(candidates_by_size.items())}
        out["Num Candidates Total"] = total_candidates
    time1 = time.time()
    out['Runtime'] = time1-time0
    with open(outputJson, 'w') as f:
        json.dump(out, f)
  
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Task 2: SON algorithm on Yelp data')
    parser.add_argument('--b', type=int, help='Filter threshold for users')
    parser.add_argument('--t', type=int, help='Support threshold for frequent itemsets')
    parser.add_argument('--input_file', type=str, help='Input file path')
    parser.add_argument('--output_file', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    # Extract arguments
    filter_threshold = args.b
    support_threshold = args.t
    input_file = args.input_file
    output_file = args.output_file
    
    hashSize = 30000000
    
    # Record start time
    time0 = time.time()
    
    # Read Input
    sc = SparkContext()
    sc.setLogLevel("ERROR")
    sc.setLogLevel("OFF")
    rdd = sc.textFile(input_file)
    main(rdd, filter_threshold, support_threshold, output_file, hashSize)
    sc.stop()

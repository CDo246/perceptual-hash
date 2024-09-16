from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os
import hashlib
import pickle
from typing import List, Union

def parallel_map(array, function):
    """
    Parallelizes the mapping of a function over an array.
    """
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(function, array))
    return results

def cache_wrapper(func, cache_dir, args: Union[str, List[str]]):
    # Create a unique cache key based on all string arguments (using a hash)
    concatenated_args = str(args)
    cache_key = hashlib.sha256(concatenated_args.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f'{cache_key}.pkl')

    # If the cache file exists, load the cached result
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            print(f"Loading cached result for arguments: {args}")
            return pickle.load(f)

    # Otherwise, compute the result, cache it, and return it
    result = func(*args)
    with open(cache_file, 'wb') as f:
        print(f"Caching result for arguments: {args}")
        pickle.dump(result, f)

    return result


def cache_wrapper_for_parallel(data):
    # Extract args from "data"
    func = data[0]
    folder = data[1]
    args = data[2:]
    return cache_wrapper(func, folder, args)

def parallel_map_with_cache(array, function, cache_dir='cache'):
    """
    Parallelizes the mapping of a function over an array.
    """
    # with ProcessPoolExecutor() as executor:
    #     results = list(executor.map(function, array))
    # return results

    with ProcessPoolExecutor() as executor:
        mapped_array = list(map(lambda x: [function, cache_dir, x], array))
        results = list(executor.map(cache_wrapper_for_parallel, mapped_array))
    return results

def cache_function(func, cache_dir='cache'):
    """
    A function that caches the output of a function that takes an arbitrary number of string arguments.
    The cache is stored on disk for use between runs.

    Parameters:
    - func: The function whose output should be cached.
    - cache_dir: The directory where the cache will be stored.
    """

    # Ensure the cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    def wrapper(*args: str):
        return cache_wrapper(func, *args)

    return wrapper
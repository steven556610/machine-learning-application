import time

'''
from utils import timeit

@timeit
def my_slow_function():
    time.sleep(2)
    print("Function executed!")

my_slow_function()
'''

def timeit(func):
    """
    A decorator that measures the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds.")
        return result
    return wrapper
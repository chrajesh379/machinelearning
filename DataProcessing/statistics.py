# we can use import statistics  module to get the all the statistics
# but below logic is used internally to calculate mean, varience and Standard deviation
import numpy as np
import math
def mean_cal(data):
    mean = sum(data)/len(data)
    return mean
def varience(data):
    mean = mean_cal(data)
    deviations = [ (i-mean) ** 2 for i in data]
    varience = sum(deviations)/len(data)
    return varience
def stdev(data):
    var = varience(data)
    std_dev = math.sqrt(var)
    return std_dev


data = np.array([7, 5, 4, 9, 12, 45])
print("Mean of the sample is % s " % (mean_cal(data)))
print("varience of the sample is % s " % (varience(data)))
print("Standard Deviation of the sample is % s " % (stdev(data)))

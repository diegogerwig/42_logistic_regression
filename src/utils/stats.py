import math


def mean(data):
    '''
    Computes the mean value of a list of numbers
    '''
    sum_values = 0
    for i in range(data.size):
        sum_values += data.iloc[i]
    mean = sum_values / data.size
    return mean


def std(data, mean):
    '''
    Computes the standard deviation of a list of numbers

    The standard deviation is a statistical measure indicating the dispersion
      or variability of a dataset relative to its mean. In other words, it
      represents how much individual values deviate from the mean. 
      A high standard deviation indicates that the data points are spread out 
      widely around the mean, while a low standard deviation indicates that 
      the data points are clustered closer to the mean.
    '''
    sum_values = 0
    for i in range(data.size):
        sum_values += (data.iloc[i] - mean) ** 2
    std = math.sqrt(sum_values / data.size)
    return std


def percentile(data, percentile):
    '''
    Computes the expected percentile of a list of numbers

    Percentiles are values that divide a dataset into 100 equal parts,
      where each part represents 1% of the data. Percentiles are used to
      identify the relative standing of a particular value within the dataset.
      For example, the 50th percentile is the median, which represents the
      value below which 50% of the data falls.
    '''
    i = float(percentile * (data.size - 1))
    if i.is_integer():
        percentile = data.iloc[int(i)]
        return percentile
    else:
        percentile = data.iloc[math.floor(i)] * (1 - (i % 1)) +\
              data.iloc[math.ceil(i)] * (i % 1)
        return percentile


def frequency(data):
    """
    Calculate the maximum frequency of any value in a list.
    """
    # Create a dictionary to count the frequency of each element
    frequencies = {}
    for item in data:
        if item in frequencies:
            frequencies[item] += 1
        else:
            frequencies[item] = 1

    # Find the maximum frequency
    max_frequency = 0
    for frequency in frequencies.values():
        if frequency > max_frequency:
            max_frequency = frequency

    return max_frequency


def mode(data):
    """
    Calculate the mode of a list of values.
    """
    # Create a dictionary to count the frequency of each element
    frequencies = {}
    for item in data:
        if item in frequencies:
            frequencies[item] += 1
        else:
            frequencies[item] = 1

    # Find the element with the highest frequency
    mode = None
    max_frequency = 0
    for item, frequency in frequencies.items():
        if frequency > max_frequency:
            mode = item
            max_frequency = frequency

    return mode

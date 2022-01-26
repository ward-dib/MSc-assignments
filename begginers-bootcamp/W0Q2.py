# Bootcamp exercises Q2:

''' Debug the script buggy.py so that it executes cleanly
and yields the expected
behaviour. '''

# A buggy script

# This script aims to print the sequence (1+1/n)**n for a range of n 

''' def function(n)
    (1 + 1/n)**n

for i in range(1000):
    function(n)
print(f"n = {n} f({n}) = {result}") '''


def function(n):
    (1+1/n)**n
for n in range (1, 1000):
# If range is (1000) zero will be included giving us a division by 0.
# So the range has to be defined to specifically exclude 0.
    i = (1+1/n)**n
# Need to define the i as the result of the function at each iteration.
    print(f"i = {n} f({n}) = {i}")

# Bootcamp exercises Q5:
''' Write a while loop that breaks when the 1000th even number of the
    Fibonacci Sequence is reached. '''

# The length of our Fibonacci sequence

length = 1000

# The first two values
n_1 = 0
n_2 = 1
iteration = 0

# Condition to check if the length has a valid input
if length <= 0:
   print("Please provide a number greater than zero")
elif length == 1:
   print("This Fibonacci sequence has {} element".format(length), ":")
   print(n_1)
else:
   print("This Fibonacci sequence has {} elements".format(length), ":")
   while iteration < length:
       print(n_1, end=', ')
       n_3 = n_1 + n_2
       # Modify values
       n_1 = n_2
       n_2 = n_3
       iteration += 1
# Bootcamp exercises Q4:
''' Write a for loop that calculates the first N = 100 numbers in the
    Fibonacci Sequence. Append values to a list.'''


# Define the fibbonacci list as empty so we can fill it later.

fibbonaci = []

# Define fibbonaci function starting values.

def fibonacci(n):
    n_1 = 1
    n_2 = 1
    
# If the statement is true, do this.   
 
    if n==1:
        print(0)

#Else-if statements must between the opening ‘if’ and the closing ‘else’.

    elif n==2:
        print(0, 1)
        
#else do this

    else:
        print("fibbonacci series")
        print(0)
        print(n_1)
        print(n_2)
        
# The loop:
    
        for i in range(n-3):
            total = n_1 + n_2
            n_2 = n_1
            n_1 = total
            print(total)
            
# Command to place the results into a list

            fibbonaci.append(total)
        return n_2  
print (fibbonaci)   

# How many values we want:
    
fibonacci(100)

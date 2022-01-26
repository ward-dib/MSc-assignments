# SIMPLE FIBBONACCI 

def fibbonacci(n):
   if n <= 1:
       return (n)
   else:
       return (fibbonacci(n-1) + fibbonacci(n-2))
length = 20
for i in range(length):
    print(fibbonacci(i))

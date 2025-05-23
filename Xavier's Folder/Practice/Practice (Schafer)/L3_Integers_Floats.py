num1 = 3        # Creates an integer variable.
num2 = 3.14     # Creates a float variable.

print(type(num1))       # Checks the variable type.
print(type(num2))

# Arithmetic Operations
print(3 + 2)        # Addition
print(3 - 2)        # Subtraction
print(3 * 2)        # Multiplication
print(3 / 2)        # Division
print(3 // 2)       # Floor Division    This takes away the remaining decimal.
print(3 ** 2)       # Exponent
print(3 % 2)        # Modulus

print(3 * 2 + 1)        # Follows the typical order of operations.
print(3 * (2 + 1))

num = 1     # Defines an integer and then redefines it by adding one.
num = num + 1
num += 1
print(num)      # Prints '3' since '1' is added twice via different methods.

num = 2     # Defines an integer and then redefines it by multiplying by 10.
num = num * 10
num *= 10
print(num)

print(abs(-3))      # Prints the absolute value of '-3'.

print(round(3.75))      # Rounds '3.75' to '4'.

print(round(3.75, 1))       # Rounds '3.75' to the first decimal place which gives '3.8'.

num1 = 3
num2 = 2

# Comparison Operators
print(num1 == num2)     # Checks if 'num1' equals 'num2'.
print(num1 != num2)     # Checks if 'num1' doesn't equal 'num2'.
print(num1 > num2)      # Checks if 'num1' is greater than 'num2'.
print(num1 < num2)      # Checks if 'num1' is less than 'num2'.
print(num1 >= num2)     # Checks if 'num1' is greater than or equal to 'num2'.
print(num1 <= num2)     # Checks if 'num1' is less than or equal to 'num2'.
# All of these print either 'True' or 'False'.

num1  = '100'       # Defines strings and concatenates them.
num2 = '200'
print(num1 + num2)

num1 = int(num1)    # Redefines the strings as integers and adds them.
num2 = int(num2)
print(num1 + num2)

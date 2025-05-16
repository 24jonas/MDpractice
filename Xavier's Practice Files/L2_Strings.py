print("Hello World") # Prints "Hello World".

print('Bobby\'s World') # Prints "Bobby's World" using single quotes.

print("Bobby's World") # Prints "Bobby's World" using double quotes.

message = "Hello World" # Creates a string.

print(message) # Prints the string object 'message'.

print(len(message)) # Prints the length of 'message'.

print(message[4]) # Prints the 4th index in 'message'. Prints the 5th digit.

print(message[:5]) # Prints up to the 5th index exclusive. Indeces 0-4 are printed.

print(message[6:]) # Prints 'message' starting at the 6th index inclusive. Indeces 6-10 are printed.

print(message[3:8]) # Prints starting from the 3rd index inclusive to the 8th index exclusive. Indeces 3-7 are printed.

print(message.lower()) # Prints 'message' in lowercase.

print(message.upper()) # Prints 'message' in uppercase.

print(message.count('l')) # Returns the number of instances of "l". Single or double quotes work.

print(message.count('Hello')) # Returns the number of instances of "Hello"

print(message.find('l')) # Returns the index of the first instance of "l".

print(message.find('a')) # Returns "-1" since "a" does not appear in 'message'.

new_message = message.replace('World', 'Universe') # Create a new string with "World" replaced by "Universe".
print(new_message)

message = message.replace('World', 'Universe') # Changes 'message' by replacing "World" with "Universe".
print(message)

greeting = 'Hello' # Creating two new strings.
name = 'Michael'

message = greeting + name # Redefines 'message' as a combination of 'greeting' and 'name'.
print(message)
print(greeting + name) # Prints combination of 'greeting' and 'name' which is the same as 'message'.

message = greeting + ', ' + name + '. Welcome!' # Adds spacing a punctuation to 'message'.
print(message)

message = '{}, {}. Welcome!'.format(greeting, name) # A different way to format 'message'.
print(message)

message = f'{greeting}, {name}. Welcome!' # Another way to format 'message'. This is called 'f-string'.
print(message)

print(dir(name)) # Prints all of the various methods that can be applied to the variable 'name'.

print(help(str)) # Provides information about various string methods.

print(help(str.lower)) # Provides informarion about the string method '.lower'.

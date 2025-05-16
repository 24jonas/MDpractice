dict1 = {'name': 'John', 'age': 25, 'courses': ['Math', 'CompSci']}
# VARIABLE = {KEY: VALUE, KEY: VALUE, KEY: VALUE}

print(dict1)        # Prints the whole dictionary.
print(dict1['name'])        # Prints the value for name.
print(dict1['age'])     # Prints the value for age.
print(dict1['courses'])     # Prints the value for courses.

print(dict1.get('name'))        # Prints the value for name.
print(dict1.get('phone'))       # Prints "None" since there is no key called "phone".
print(dict1.get('phone', 'Not Found'))      # Prints "Not Found" since there is no key called "phone".

dict1["phone"] = "(555) 555-5555"       # Adds a key/value pair to 'dict1'.
print(dict1.get('phone', "Not Found"))      # Prints the value associated with "phone".

dict1["name"] = "Jane"     # Changes the "name" value to "Jane".
print(dict1.get('name'))

dict1.update({'name': 'Thomas', 'age': 26, 'phone': '(444) 444-4444'})
print(dict1)        # Prints dict1 with updated values for each of the keys addressed.

del dict1['age']        # Deletes the key/value pair for "age".
print(dict1)

name = dict1.pop('name')        # Defines 'name' as the vlaue for the 'name' key and also removes that
print(name)                     # key/value pair. Note that in this context '.pop()' requires an argument.

print(len(dict1))       # Prints the number of key/value pairs in 'dict1'.

print(dict1.keys())     # Prints the keys in 'dict1'.
print(dict1.values())       # Prints the values in 'dict1'.
print(dict1.items())        # Prints the items in 'dicts1'.

for key in dict1:
    print(key)

for key, value in dict1.items():
    print(key, value)

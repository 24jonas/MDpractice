if True:        # The 'if' condition is always true so this prints "True".
    print("True")

if False:       # The 'if' condition is always false so this never prints.
    print("False")

a = 1

if a == 0:
    print("0")
elif a == 1:
    print("1")
else:
    print("2")

user = 'admin'
logged_in = False

if user != 'admin' and logged_in == False:
    print("Please log in")
else:
    print("Welcome")

a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(id(a))
print(id(b))
print(id(c))

print(a is b)
print(a is c)

# False Values  -   Everything else evaluates to true.
# False
# None
# Zero - 0 or 0.0
# Empty Sequences - '', (), []
# Empty Mapping - {}

condition = False

if condition:
    print('True')
else:
    print("False")

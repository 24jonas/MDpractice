print("Mosh Hamedani")
print('o----')
print(' ||||')
print('*' * 10)

price = 10
print(price)

price = 20
print(price)

name = "John Smith"
age = 20
new = True

name = input('What is your name?: ')
print("Hi " + name)

color = input('What is your favorite color?: ')
print(name + " likes " + color + ".")

birth_year = int(input("Birth Year?: "))
age = 2025 - birth_year
print("Age: " + str(age))

weight_lbs = input("Weight (lbs): ")
weight_kg = int(weight_lbs) * 0.45
print(weight_kg)

course1 = "Python's Course For Beginners."
course2 = "Python's Course For \"Beginners\"."
print(course1)
print(course2)

print(course1[-2])

first = "John"
last = "Smith"

message = first + ' [' + last + '] is a coder'
print(message)

msg = f"{first} [{last}] is a coder"
print(msg)

course = "Python for Beginners"
print(len(course))

print(course.upper())
print(course.find('o'))

print(course.replace("Beginners", "Absolute Beginners"))

print("Python" in course)

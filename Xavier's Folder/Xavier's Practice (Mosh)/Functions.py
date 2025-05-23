def greeting(name):
    print(f"Hello {name}")

# greeting(input("Name: "))
# greeting(name = "Beb")

def square(number):
    return number ** 2

# print(square(3))

try:
    print(int(input("Age: ")))
    print(100 / int(input("Income: ")))
except ValueError:
    print("Input must be an integer.")
except ZeroDivisionError:
    print("Income cannot be zero.")

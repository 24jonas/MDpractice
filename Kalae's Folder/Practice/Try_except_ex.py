try: 
    age = int(input("Enter your age: "))
    income = 100000
    risk = income / age
    print(f"your risk is {risk}") 
except ZeroDivisionError:
    print("Age cannot be zero. Please enter a valid age.")
except ValueError:
    print("Invalid input. Please enter a number.")
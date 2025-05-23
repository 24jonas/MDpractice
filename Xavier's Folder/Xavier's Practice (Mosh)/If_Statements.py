is_hot = False
is_cold = False

if is_hot:
    print("It's a hot day.")
elif is_cold:
    print("It's a cold day.")
else:
    print("Its's a lovely day.")

print("Enjoy your day.")


price = 1000000
credit = True

if credit:
    payment = 0.1 * price
else:
    payment = 0.2 * price

print(f"Down Payment: {payment}")

income = True
credit = True
criminal = False

if income and credit:
    print("Eligible for a loan.")

if income or credit:
    print("Eligible for a loan.")

if credit and not criminal:
    print("Eligible for a loan.")

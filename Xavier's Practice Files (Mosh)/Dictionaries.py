customer = {
    "Name": "John Smith",
    "Age": 30,
    "Verified": True
}

print(customer["Name"])
print(customer.get("Birthdate"))

customer["Birthdate"] = "Jan 1, 1980"

print(customer)

# Exercise

a = {
    '0': "Zero",
    '1': "One",
    '2': "Two",
    '3': "Three",
    '4': "Four",
    '5': "Five",
    '6': "Six",
    '7': "Seven",
    '8': "Eight",
    '9': "Nine"
}

number = input("Phone Number: ")
letter = ""

for n in number:
    letter = letter + a[n] + " "

print(letter)

print("____________________________")

message = input("> ")
words = message.split(" ")
print(words)

emojis = {
    ":)": "😀",
    ":(": "🙁"
}
output = ""
for word in words:
    output += emojis.get(word, word) + " "

print(output)
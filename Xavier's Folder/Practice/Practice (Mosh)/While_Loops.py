a = 1
while a <= 5:
    print(a)
    a = a + 1

print("Done")

# Guessing game.

secret = 9
count = 0
limit = 3

while count < limit:
    guess = int(input("Guess a number: "))
    count += 1
    if guess == secret:
        print("You won.")
        break
else:
    print("You lose.")
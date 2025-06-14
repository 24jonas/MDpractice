#example 1
'''
for i in range(1, 11, 2):
    print(i)
'''
#example 2: loops can be used in strings.
'''
credit_card = "1234-5678-9012-3456"

for i in credit_card:
    print(i)
'''
#example 3: skip a number in a for loop
for i in range(1,11):
    if i ==5:
        continue
    print(i)

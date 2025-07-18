Loop_success = True
value = 1 
while value <= 10:
    print(value)
    value += 1
    if value == 9:
        print("Value is now 9, breaking the loop.")
        Loop_success = False
        break

if Loop_success == True:
    print("Loop completed successfully.")
else:
    print("Loop was interrupted before completion.")
# This code demonstrates a simple while loop that counts from 1 to 10 and breaks when it reaches 9.
# It also checks if the loop completed successfully or was interrupted.
       
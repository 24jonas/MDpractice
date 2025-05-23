def square(number):
    return number * number
print(square(5))
def cube(number):
    return number**3
print(cube(5))

print(square(cube(2)))
""" This code shows the basics of passing functions as arguments to other functions.
    It defines two functions, square and cube, that calculate the square and cube of a number, respectively.
    The code then demonstrates how to use these functions by passing them as arguments to another function.
"""

#emoji converter.

def emoji_converter(message):
    words = message.split(" ")
    # splits words into a list of words.
    emojis = {
        ":)": "😊",
        ":(": "😞",
        ":D": "😄",
        ":P": "😜",
        ":o": "😮",
        ":/": "😕"
    }
    # This dictionary maps text emoticons to their corresponding emoji characters.
    output = ""
    for word in words:
        output += emojis.get(word, word) + " "
        # this loop goes through every word in the function words.
    return output

message = input(">")
print(emoji_converter(message))





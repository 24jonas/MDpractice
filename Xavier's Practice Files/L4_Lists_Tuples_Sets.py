courses = ['History', 'Math', 'Physics', 'Chemistry']        # Creates a list containng string variables.
print(courses)

print(len(courses))     # Prints the length or the list ie the number of variables.

print(courses[0])       # Printss the 0th index of the list. The first item.

print(courses[-1])      # Prints the last item of the list.

# print(courses[5])     Produces an error since the index is out of range.

print(courses[:2])      # Prints the list items up to index 2 exclusive.

print(courses[2:])      # Prints the list items starting from index 2 until the end of the list.

courses.append('Programming')       # Adds a string 'Programming' to the end of the list.

courses.insert(2, 'Biology')        # Inserts the string 'Biology' to the 2nd index. This does not replace any exiting items.

print(courses)

classes = ['Art', 'Music']      # Creates a new list and inserts it as an item in the 'courses' list.
courses.insert(0, classes)
print(courses)

courses.extend(classes)     # Adds the items of 'classes' to the end of 'courses'.
print(courses)

courses.append("Math")      # List items can have duplicates.
print(courses)

courses.remove("Math")      # Removes the first instance of the argument including if the argument is a list.
courses.remove(classes)
print(courses)

courses.pop()       # Removes the last item of the list.
print(courses)

last_item = courses.pop()       # Defines a variable as the last item of the list while simultaneously removing that item.
print(courses)

courses.reverse()       # Reverses the order of items in the list.
courses.sort()          # For strings, this sorts the items alphabetically.
print(courses)

numbers = [1,5,2,2,8,7,5,9,3,5,6]
numbers.sort()      # For numbers, it sorts them from least to greatest.
print(numbers)

numbers.sort(reverse = True)        # Sorts numbers from greatest to least.
print(numbers)

numbers1 = [6,6,4,8,9,1,0,0,0,0,3,5,6,3,2,7]
sorted_numbers = sorted(numbers1)       # Another way to sort items.
print(sorted_numbers)

print(min(numbers))     # Prints the smallest item in 'number'.
print(max(numbers))     # Prints the largest item in 'number'.
print(sum(numbers))     # Prints the sum of all items in 'number'.

print(courses.index("Physics"))     # Prints the index at which "Physics" is found in 'courses'.
# print(courses.index("Math"))      # Prints an error since "Math" is not in the list.

print("Math" in courses)        # Checks if "Math" is in 'courses'.

for item in courses:        # Prints each item of courses sequentially.
    print(item)

for course in courses:      # This does the same thing as before but with a different argument name.
    print(course)

for index, course in enumerate(courses):        # Prints the items along with their index.
    print(index, course)

for index, course in enumerate(courses, start=1):        # Prints the items enumerated, but starting at 1 not 0.
    print(index, course)

course_str = ", ".join(courses)     # Adds commas between the items and prints a string.
print(course_str)

new_list = course_str.split(', ')   # Separates the previous string by the argument and creates a list with the sepatated parts as items.
print(new_list)

courses[3] = "Engineering"      # Replaces the item at index 3 with "Engineering".
print(courses)

tuple1 = ("A", "B", "C", "D", "E")
# tuple1[0] = "F"       # Causes an error because tuples are not mutable.
print(tuple1[3])

set1 = {"A", "B", "C", "D", "E"}
print(set1)     # Prints elements in an arbitrary order each time the print command is executed.

print("F" in set1)      # Checks if "F" is in 'set1'.

set2 = {"C", "D", "E", "F", "G"}

print(set1.intersection(set2))      # Prints elements found in both 'set1'  and 'set2'.

print(set1.difference(set2))        # Prints elements found in 'set1' but not in 'set2'.

print(set1.union(set2))     # Prints elemens found in either 'set1' or 'set2'.

list1 = []      # Empty lists.
list2 = list()

tuple1 = ()     # Empty tuples.
tuple2 = tuple()

set1 = {}       # Creates an empty dictionary.
set2 = set()    # Emppty set. Note that these are parentheses.

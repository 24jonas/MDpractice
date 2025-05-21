#Capitalize the first letter of each word in a strin
class Point:
    #self if reference to the current instance of the class.
    # we construct the class with a constructor.
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move(self): 
        print("move")
    def draw(self):
        print("draw")
    # Objects are instances of a class.
    # A class is a blueprint for creating objects.

point1 = Point(10, 20)
point1.x = 11
point1.y = 20
print(point1.x)
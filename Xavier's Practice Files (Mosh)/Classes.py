class Point:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def move(self):
        print("move")
    
    def draw(self):
        print("draw")

"""
p1 = Point()
p1.draw()

p1.x = 10
p1.y = 20

print(p1.x)
"""

p2 = Point(10,20)
print(p2.x)

print("________________________")

class Person:
    def __init__(self, name):
        self.name = name

    def talk(self):
        print(f"Hi, I am {self.name}")

john = Person("John Smith")
john.talk()

bob = Person("Bob Smith")
bob.talk()
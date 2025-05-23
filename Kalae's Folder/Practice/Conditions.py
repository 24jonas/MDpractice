class Person: 
    def __init__(self, name):
        self.name = name
        #self.name is the name argument passed to the constructor.
        # The constructor initializes the name attribute of the object.
    def talk(self):
        print(f"Hi my name is {self.name}")
Henry = Person("Henry")
print(Henry.name)
#Gets the name element of the henry which is equal to the class person.
Henry.talk()


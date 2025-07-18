def Hello():
    pass

def Hello():
    print('Hello')

print(Hello)
print(Hello())
Hello()

def Hi():
    return 'Hi'

print(Hi)
print(Hi())
a = Hi()
print(a)

# print(Hello().upper())    Gives an error.
print(Hi().upper())

def Greeting(name):
    return 'Hi' + ' ' + name

print(Greeting('John'))

def student_info(*args, **kwargs):
    print(args)
    print(kwargs)

student_info('Math', 'Art', name='John', age=22)

courses = ['Math', 'Art']
info = {'name': 'John', 'age': 22}
student_info(courses, info)
student_info(*courses, **info)
 
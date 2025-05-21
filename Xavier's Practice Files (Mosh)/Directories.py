from pathlib import Path

# Absolute path example:
# /usr/local/bin

path1 = Path("Xavier's Practice Files (Mosh)")
print(path1.exists())

path2 = Path("Package1")
print(path2.exists())

print(path2.mkdir())    # Makes a directory called "Package1"
print(path2.rmdir())    # Removes a directory called "Package1"

# Both of the above print statements print "None".

for file in path1.glob("*.py"):
    print(file)

print("___________________________")

for file in path1.glob("*"):
    print(file)

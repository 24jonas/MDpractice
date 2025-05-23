command = ""
while command != "quit":
    command = input("> ").lower()
    if command == "start":
        print("Car started...")
    elif command == "stop":
        print("Car stopped")
    elif command == "help":
        print("""
            start - To start the car.
            stop - To stop the car.
            quit - To quit.
              """)
    elif command == "quit":
        break
    else:
        print("Sorry, I don't understand that.")

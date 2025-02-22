def params_selection(name, values):
    while True:
        print(f"\nChoose a value for {name}:")
        for i, val in enumerate(values):
            print(f"{i + 1}) {val}")
        
        choice = input("Enter the number of your choice (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            return None  # Signal to terminate execution
        
        if choice.isdigit():
            choice = int(choice) - 1
            if 0 <= choice < len(values):
                return values[choice]
        
        print("❌ Invalid choice, please try again.")


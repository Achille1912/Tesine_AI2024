def params_selection(name, values):
    """
    Allows the user to interactively select a value from a provided list of choices. The function continuously prompts 
    the user until a valid selection is made or the user chooses to quit. The function returns the selected value 
    or `None` if the user opts to quit.

    :param name: The name or label of the category for which a choice is being made, used in display messages.
    :type name: str
    :param values: List of possible values from which the user can make a selection.
    :type values: list
    :return: The selected value if a valid selection is made, or `None` if the user chooses to quit.
    :rtype: str | None
    """
    while True:
        print(f"\nChoose a value for {name}:")
        for i, val in enumerate(values):
            print(f"{i + 1}) {val}")
        
        choice = input("Enter the number of your choice (or 'q' to quit): ")
        
        if choice.lower() == 'q':
            return None  
        
        if choice.isdigit():
            choice = int(choice) - 1
            if 0 <= choice < len(values):
                return values[choice]
        
        print("❌ Invalid choice, please try again.")


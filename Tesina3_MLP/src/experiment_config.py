def get_user_choice(prompt, options):

    print("\n" + prompt)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        choice = input(f"Select an option (1-{len(options)}): ").strip()

        # Check if the number is a valid one
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(options):
                return options[choice - 1]


        print(f"⚠️  Invalid input! You must enter a number between 1 and {len(options)}.")


def get_user_config():

    hidden_layer_sizes_options = [(200,), (400,), (600,), (400, 200), (600, 300), (800, 400), (400, 200, 100), (600, 300, 150)]
    activation_options = ['identity', 'logistic', 'tanh', 'relu']
    solver_options = ['adam', 'sgd', 'lbfgs']
    alpha_options = [0.0001, 0.001, 0.01, 0.1, 0.5]
    batch_size_options = [16, 32, 64]
    learning_rate_policy_options = ['constant', 'invscaling', 'adaptive']
    learning_rate_init_options = [0.0001, 0.001, 0.01, 0.1]
    validation_fraction_options = [0.1, 0.15, 0.2]
    n_iter_no_change_options = [5, 10, 20]

    config = {
        "hidden_layer_sizes": get_user_choice("Choose the size of hidden layers:", hidden_layer_sizes_options),
        "activation": get_user_choice("Choose the activation function:", activation_options),
        "solver": get_user_choice("Choose the solver:", solver_options),
        "alpha": float(get_user_choice("Choose the value of alpha:", alpha_options)),
        "batch_size": int(get_user_choice("Choose batch size:", batch_size_options)),
        "learning_rate_policy": get_user_choice("Choose the type of learning rate:", learning_rate_policy_options),
        "learning_rate_init": get_user_choice("Choose the initial value of the learning rate:", learning_rate_init_options),
        "validation_fraction": float(get_user_choice("Select the validation fraction:", validation_fraction_options)),
        "n_iter_no_change": int(get_user_choice("Choose the number of iterations without improvement:", n_iter_no_change_options))
    }

    return config

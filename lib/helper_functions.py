import sys

def print_input_info():
    print("Please input if you want random play, training or old network.\n"
          "Either:\n "
          " - random\n"
          " - train\n"
          " - oldnetwork")

def check_arg_sys_input():
    option_dict = {"random": False, "train": False, "oldnetwork": False}
    try:
        argument = sys.argv[1]
        for key in option_dict:
            if argument in key.lower():
                option_dict[key] = True
    except:
        print_input_info()
        raise AttributeError(": None argument was given")

    if all(value == 1 for value in option_dict.values()):
        print_input_info()
        raise AttributeError(": Wrong argument was given")

    return option_dict
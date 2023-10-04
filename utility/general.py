
def yes_no(msg: str):
    """
    A function that prompts the user with a message and expects a yes or no answer.
    
    Args:
        msg (str): The message to display to the user.
        
    Returns:
        bool: True if the user answers "y", False if the user answers "n".
    """
    while True:
        inp = input(f"{msg} (y/n) ")
        match inp.lower():
            case "y":
                return True
            case "n":
                return False
            case _:
                pass

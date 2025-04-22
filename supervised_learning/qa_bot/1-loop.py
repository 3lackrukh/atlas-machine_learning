#!/usr/bin/env python3
"""Module defines the loop method"""


def loop():
    """
    Loop function for building out the question_answer system.

    Prints "Q:" prompting the user for input.

    Prints "A:" as a response

    If the user enters 'exit', 'quit', 'goodbye', or 'bye' (case insensitive),
    the function prints "A: Goodbye" and exits.
    """
    while True:
        # Prompt the user for input
        user_input = input("Q: ")

        # Check if the user wants to exit
        if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        # Print the response
        print("A:")


if __name__ == "__main__":
    loop()

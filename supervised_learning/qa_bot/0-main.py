#!/usr/bin/env python3
"""
This script implements a question-answering interface that uses a BERT-based model
to find answers to user questions from a provided reference document.
"""

question_answer = __import__('0-qa').question_answer


def main():
    """
    Main function that handles user input and calls the question_answer function.
    Loads the reference document and allows users to ask multiple questions.
    """
    # Load the reference document
    reference_file = 'ZendeskArticles/PeerLearningDays.md'
    try:
        with open(reference_file, 'r') as f:
            reference = f.read()
            print(f"Successfully loaded reference document: {reference_file}")
    except FileNotFoundError:
        print(f"Error: Reference file '{reference_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading reference file: {str(e)}")
        return
    
    print("\nQuestion-Answering System")
    print("========================")
    print("Ask questions about Peer Learning Days (type 'exit' to quit)")
    
    # Main interaction loop
    while True:
        # Get user question
        user_input = input("\nYour question: ").strip()
        
        # Check if user wants to exit
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Exiting the program. Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            print("Please enter a question or type 'exit' to quit.")
            continue
        
        try:
            # Get answer from question_answer function
            print("Finding answer...")
            answer = question_answer(user_input, reference)
            
            # Display the answer
            print("\nAnswer:", answer)
            
        except Exception as e:
            print(f"Error processing your question: {str(e)}")


if __name__ == "__main__":
    main()
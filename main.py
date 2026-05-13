from core.llm import chat
from ui.cli_loader import LoadingAnimation

def main():
    """
    Main entry point of the AI agent.
    Handles user input and displays response with loading animation.
    """
    print("AI Agent Started. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == "quit":
            break

        # Show loading animation while waiting for AI response
        loader = LoadingAnimation("Responsing...")
        loader.start()
        
        response = chat(user_input)
        
        loader.stop()
        
        print(f"AI: {response}\n")

if __name__ == "__main__":
    main()
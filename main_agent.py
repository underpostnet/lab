# main_agent.py
import os
import argparse
import sys
from dotenv import load_dotenv

# Import the main application class and the LLM adapter.
from src.gemini_agent_app import GeminiAgentApp
from src.llm_adapters import GeminiLLMAdapter  # New import for LLM adapter

load_dotenv()

# --- Constants and Configurations ---
# Default Gemini model to use if not specified via command-line arguments.
DEFAULT_MODEL = "models/gemini-2.5-flash-preview-04-17"
# Default temperature for the Gemini model, influencing creativity (0.0 to 1.0).
DEFAULT_TEMPERATURE = 0.7
# The command a user can type to exit the conversational loop.
EXIT_COMMAND = "exit"

# --- Main Execution Block ---
if __name__ == "__main__":
    # Initialize the argument parser to handle command-line arguments.
    parser = argparse.ArgumentParser(
        description="Run a LangChain Gemini Agent with conversational capabilities."
    )

    # Add argument for the Google Gemini API key.
    # It defaults to the value of the GEMINI_API_KEY environment variable if set.
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("GEMINI_API_KEY"),
        help="Your Google Gemini API key. Can also be set via GEMINI_API_KEY environment variable.",
    )

    # Add argument for the Gemini model name.
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"The Gemini model to use (default: {DEFAULT_MODEL}).",
    )

    # Add argument for the model's temperature.
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"The model's temperature for creativity (0.0-1.0, default: {DEFAULT_TEMPERATURE}).",
    )

    # Parse the command-line arguments provided by the user.
    args = parser.parse_args()

    try:
        # Initialize the LLM adapter based on the chosen model type (currently only Gemini)
        # In a more complex scenario, you might have a factory function here
        # to select between different adapters (e.g., OpenAIAdapter, CohereAdapter).
        llm_adapter = GeminiLLMAdapter(
            model_name=args.model,
            temperature=args.temperature,
            api_key=args.api_key,  # Pass API key to the adapter
        )

        # Get the actual LangChain LLM instance from the adapter
        llm_instance = llm_adapter.get_llm()

        # Initialize an instance of GeminiAgentApp with the LLM instance.
        # This separates the LLM creation logic from the application logic.
        app = GeminiAgentApp(llm_instance=llm_instance)

        # Start the conversational loop of the agent.
        app.run_conversation(exit_command=EXIT_COMMAND)
    except ValueError as ve:
        # Catch specific ValueError for configuration issues (e.g., missing API key).
        print(f"Configuration Error: {ve}")
        print("Please provide a valid Google API Key. See README.md for more details.")
    except Exception as e:
        # Catch any other unexpected exceptions that might occur during startup or conversation.
        print(f"An unexpected error occurred during application startup: {e}")

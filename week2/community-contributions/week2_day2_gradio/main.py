import os
import gradio as gr
import google.generativeai as genai
from typing import List, Tuple

# --- Configuration ---
# It's best practice to use environment variables for API keys.
# Make sure to set your GOOGLE_API_KEY in your environment.
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("ERROR: Please set the GOOGLE_API_KEY environment variable.")
    exit()

# --- Tool Definition ---
# This is the function the Gemini model will be able to call.
def price_function(item: str) -> str:
    """
    Get the price of a given item.
    Args:
        item: The name of the item for which to find the price.
    """
    print(f"Tool 'price_function' called with item: {item}")
    if "shirt" in item.lower():
        return "The price of a shirt is $25."
    elif "jeans" in item.lower():
        return "The price of jeans is $50."
    else:
        return f"Sorry, I don't have a price for {item}."

# --- Model and Chat Configuration ---
system_message = (
    "You are a helpful shopping assistant. "
    "You can use tools to find the price of items. "
    "When a user asks for a price, use the provided tool."
)

# This is the list of tools you provide to the model.
tools_For_Gemini = [price_function]

# --- Helper Function for History ---
def convert_history_to_gemini_format(history: List[Tuple[str, str]]) -> List[dict]:
    """
    Converts Gradio's chat history format to Gemini's format.
    Gradio: [("user message", "model message"), ...]
    Gemini: [{"role": "user", ...}, {"role": "model", ...}, ...]
    """
    gemini_history = []
    for user_msg, model_msg in history:
        gemini_history.append({"role": "user", "parts": [{"text": user_msg}]})
        gemini_history.append({"role": "model", "parts": [{"text": model_msg}]})
    return gemini_history

# --- Core Chat Logic ---
def chat_Gemini_with_Tool(message: str, history: List[Tuple[str, str]]):
    """
    Handles the chat interaction with the Gemini model, including tool calls.
    """
    # 1. Initialize the Gemini model
    # We pass the tools directly to the model during initialization.
    model_instance = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_message,
        tools=tools_For_Gemini
    )

    # 2. Convert and build the chat history for the API call
    gemini_history = convert_history_to_gemini_format(history)
    gemini_history.append({"role": "user", "parts": [{"text": message}]})

    # Loop to handle potential multi-turn tool calls until a text response is received
    # or a maximum number of iterations is reached.
    max_tool_calls = 5 # Set a reasonable limit to prevent infinite loops
    for _ in range(max_tool_calls):
        print("Sending request to Gemini...")
        response = model_instance.generate_content(
            gemini_history,
            tool_config={"function_calling_config": "any"}
        )

        # Check if the response contains any candidates (it should, but good practice)
        if not response.candidates:
            print("Gemini returned no candidates.")
            return "Sorry, I couldn't generate a response."

        candidate = response.candidates[0]

        if candidate.finish_reason == 'TOOL_CALLS':
            print("Gemini responded with a tool call.")
            # The model wants to call a function.
            # For this example, we'll process the first tool call in the parts list.
            # In more complex scenarios, you might iterate through all parts.
            tool_call = candidate.content.parts[0].function_call
            tool_name = tool_call.name

            if tool_name == 'price_function':
                # A. Extract arguments and call the actual Python function
                args = {key: value for key, value in tool_call.args.items()}
                tool_result = price_function(**args)

                # B. Append the model's tool call and our tool's response to the history
                gemini_history.append(candidate.content)  # Add model's request to history
                gemini_history.append({
                    "role": "tool",
                    "parts": [{
                        "function_response": {
                            "name": "price_function",
                            "response": {"result": tool_result}
                        }
                    }]
                })
                # Continue the loop to send the tool result back to the model
                print("Tool result sent back to Gemini. Waiting for final text response...")
            else:
                # Handle cases where the model calls a function you haven't defined
                return f"Error: Model tried to call an unknown function: {tool_name}"
        elif candidate.finish_reason == 'STOP':
            # Model returned a text response, which is what we want to return to Gradio
            print("Gemini responded with text.")
            return response.text
        else:
            # Handle other finish reasons (e.g., 'RECITATION', 'SAFETY', 'OTHER')
            print(f"Gemini finished with reason: {candidate.finish_reason}. Attempting to return text if available.")
            return response.text if response.text else f"Sorry, Gemini finished with reason: {candidate.finish_reason} and no text was generated."

    # Fallback if the loop exhausts without a final text response
    return "Maximum tool call iterations reached without a final text response."

# --- Gradio UI ---
if __name__ == "__main__":
    print("Starting Gradio interface...")
    iface = gr.ChatInterface(
        fn=chat_Gemini_with_Tool,
        title="Gemini Shopping Assistant",
        description="Ask me for the price of a shirt or jeans!",
        examples=[["How much is a shirt?"], ["What is the price of a pair of jeans?"]]
    )
    iface.launch()
    print("Gradio interface running.")
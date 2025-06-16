# src/gemini_agent_app.py

import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.agents.format_scratchpad import format_log_to_messages
from langchain.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_core.language_models import BaseChatModel  # Import BaseChatModel

from src.agent_tools import AgentTools

# Models:

# name: models/embedding-gecko-001 supported_generation_methods: ['embedText', 'countTextTokens']
# name: models/gemini-1.0-pro-vision-latest supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-pro-vision supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-1.5-pro-latest supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-1.5-pro-002 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent']
# name: models/gemini-1.5-pro supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-1.5-flash-latest supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-1.5-flash supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-1.5-flash-002 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent']
# name: models/gemini-1.5-flash-8b supported_generation_methods: ['createCachedContent', 'generateContent', 'countTokens']
# name: models/gemini-1.5-flash-8b-001 supported_generation_methods: ['createCachedContent', 'generateContent', 'countTokens']
# name: models/gemini-1.5-flash-8b-latest supported_generation_methods: ['createCachedContent', 'generateContent', 'countTokens']
# name: models/gemini-2.5-pro-exp-03-25 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-pro-preview-03-25 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-flash-preview-04-17 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-flash-preview-05-20 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-flash-preview-04-17-thinking supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-pro-preview-05-06 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-pro-preview-06-05 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-exp supported_generation_methods: ['generateContent', 'countTokens', 'bidiGenerateContent']
# name: models/gemini-2.0-flash supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-001 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-exp-image-generation supported_generation_methods: ['generateContent', 'countTokens', 'bidiGenerateContent']
# name: models/gemini-2.0-flash-lite-001 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-lite supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-preview-image-generation supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemini-2.0-flash-lite-preview-02-05 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-lite-preview supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-pro-exp supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-pro-exp-02-05 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-exp-1206 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-thinking-exp-01-21 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-thinking-exp supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.0-flash-thinking-exp-1219 supported_generation_methods: ['generateContent', 'countTokens', 'createCachedContent', 'batchGenerateContent']
# name: models/gemini-2.5-flash-preview-tts supported_generation_methods: ['countTokens', 'generateContent']
# name: models/gemini-2.5-pro-preview-tts supported_generation_methods: ['countTokens', 'generateContent']
# name: models/learnlm-2.0-flash-experimental supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemma-3-1b-it supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemma-3-4b-it supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemma-3-12b-it supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemma-3-27b-it supported_generation_methods: ['generateContent', 'countTokens']
# name: models/gemma-3n-e4b-it supported_generation_methods: ['generateContent', 'countTokens']
# name: models/embedding-001 supported_generation_methods: ['embedContent']
# name: models/text-embedding-004 supported_generation_methods: ['embedContent']
# name: models/gemini-embedding-exp-03-07 supported_generation_methods: ['embedContent', 'countTextTokens', 'countTokens']
# name: models/gemini-embedding-exp supported_generation_methods: ['embedContent', 'countTextTokens', 'countTokens']
# name: models/aqa supported_generation_methods: ['generateAnswer']
# name: models/imagen-3.0-generate-002 supported_generation_methods: ['predict']
# name: models/veo-2.0-generate-001 supported_generation_methods: ['predictLongRunning']
# name: models/gemini-2.5-flash-preview-native-audio-dialog supported_generation_methods: ['countTokens', 'bidiGenerateContent']
# name: models/gemini-2.5-flash-preview-native-audio-dialog-rai-v3 supported_generation_methods: ['countTokens', 'bidiGenerateContent']
# name: models/gemini-2.5-flash-exp-native-audio-thinking-dialog supported_generation_methods: ['countTokens', 'bidiGenerateContent']
# name: models/gemini-2.0-flash-live-001 supported_generation_methods: ['bidiGenerateContent', 'countTokens']


class GeminiAgentApp:
    """
    Encapsulates the LangChain Gemini agent application logic.
    Handles agent initialization, conversational loop, and interaction with the agent.
    It now accepts an initialized LLM instance, allowing for model abstraction.
    """

    def __init__(self, llm_instance: BaseChatModel):
        """
        Initializes the GeminiAgentApp with an already initialized LangChain LLM instance.

        Args:
            llm_instance (BaseChatModel): An initialized LangChain-compatible LLM instance.
                                          This allows for flexibility in choosing the underlying model.
        """
        if not isinstance(llm_instance, BaseChatModel):
            raise TypeError(
                "llm_instance must be an instance of LangChain's BaseChatModel."
            )

        self.llm = llm_instance
        self.tools = AgentTools.get_tools()  # Get tools from the AgentTools class
        self.chat_history = []
        self.agent_executor = self._initialize_agent_executor()
        print(
            f"Gemini Agent initialized with model: {getattr(self.llm, 'model', 'Custom LLM')}"
        )

    def _initialize_agent_executor(self) -> AgentExecutor:
        """
        Initializes and returns the LangChain AgentExecutor.
        This executor is responsible for running the agent, managing its steps,
        and handling tool execution based on the ReAct framework.
        """
        # The prompt template guides the agent's behavior and thinking process.
        # It must include 'tools', 'tool_names', 'agent_scratchpad', and 'input'.
        # 'chat_history' is for conversational memory.
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant. Answer the following questions as best you can.\n"
                    "You have access to the following tools:\n"
                    "{tools}\n\n"
                    "To use a tool, you MUST use the following format and then STOP. The system will provide the Observation.\n"
                    "Thought: Do I need to use a tool? Yes\n"
                    "Action: The action to take, should be one of [{tool_names}]\n"
                    "Action Input: The input to the action\n\n"
                    "After you provide the Action and Action Input, the system will run the tool. You will then receive an 'Observation:' with the tool's result. "
                    "Based on this observation, you will continue with a new 'Thought:' and then either another Action or a Final Answer.\n\n"
                    "When you have a response to say to the Human, or if you do not need to use a tool, "
                    "you MUST use the format:\n\n"
                    "Thought: Do I need to use a tool? No\n"
                    "Final Answer: [your final answer here]"
                    "\nReminder: When using a tool, ONLY provide Thought, Action, and Action Input. Wait for the Observation.",
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),  # Placeholder for the current user input
                ("placeholder", "{agent_scratchpad}"),
            ]
        )

        # Prepare the string representation of tools and tool names for the prompt
        rendered_tools = render_text_description(self.tools)
        tool_names_str = ", ".join([tool.name for tool in self.tools])

        # Manually construct the agent runnable to ensure correct scratchpad formatting.
        # This provides more control than relying on create_react_agent's auto-detection,
        # especially when chat_history is involved in the prompt.
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_log_to_messages(
                    x["intermediate_steps"]
                ),
                tools=lambda x: rendered_tools,  # Pass rendered tools
                tool_names=lambda x: tool_names_str,  # Pass tool names
            )
            | prompt_template
            | self.llm
            | ReActSingleInputOutputParser()
        )

        # Create the AgentExecutor to execute the agent's steps.
        return AgentExecutor(
            agent=agent, tools=self.tools, verbose=True, handle_parsing_errors=True
        )

    def run_conversation(self, exit_command: str):
        """
        Runs the conversational loop, accepting user input and
        generating agent responses until the user types the specified exit command.

        Args:
            exit_command (str): The string that, when typed by the user, will exit the conversation.
        """
        print(f"LangChain Gemini Agent activated! Type '{exit_command}' to quit.")

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == exit_command.lower():
                print("Exiting agent. Goodbye!")
                break

            try:
                # Invoke the agent with the current input and the ongoing chat history
                response = self.agent_executor.invoke(
                    {
                        "input": user_input,
                        "chat_history": self.chat_history,
                    }
                )
                ai_response = response["output"]
                print(f"Agent: {ai_response}")

                # Update chat history for the next turn, maintaining the conversation context
                self.chat_history.append(HumanMessage(content=user_input))
                self.chat_history.append(AIMessage(content=ai_response))

            except Exception as e:
                print(f"An error occurred during conversation: {e}")
                print(
                    "Please ensure your LLM configuration is valid and the model is accessible."
                )
                # Breaking here prevents infinite loops on persistent errors.
                break

# src/agent_tools.py

from langchain_core.tools import Tool


class AgentTools:
    """
    Manages the collection of tools available to the LangChain agent.
    Each static method represents a specific tool that the agent can utilize.
    """

    @staticmethod
    def calculator(expression: str) -> str:
        """
        Performs a simple mathematical calculation.
        Example: '2 + 2', '10 * 5', '100 / 4'.
        Note: Using eval() is generally unsafe for untrusted input.
        For production environments, consider using a safer mathematical expression parser.
        """
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error performing calculation: {e}"

    @staticmethod
    def fake_search(query: str) -> str:
        """
        A placeholder for a search tool. In a real application, this would
        integrate with an actual search engine API (e.g., Google Search, DuckDuckGo).
        Currently provides simulated results for specific predefined queries.
        """
        if "current weather" in query.lower():
            return "The current weather in New York is partly cloudy with a temperature of 25 degrees Celsius."
        elif "population of tokyo" in query.lower():
            return "The estimated population of Tokyo, Japan is approximately 14 million people."
        else:
            return f"Simulated search result for '{query}': Information about {query}."

    @classmethod
    def get_tools(cls) -> list[Tool]:
        """
        Returns a list of LangChain Tool objects, mapping the static methods
        of this class to tools the agent can use.
        """
        return [
            Tool(
                name="Calculator",
                func=cls.calculator,
                description="Useful for performing mathematical calculations. Input should be a mathematical expression (e.g., '2+2').",
            ),
            Tool(
                name="Search",
                func=cls.fake_search,
                description="Useful for answering questions about current events, facts, or anything requiring external knowledge. Input should be a search query.",
            ),
        ]

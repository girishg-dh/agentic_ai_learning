# Week 2: Tools, APIs, and Agentic AI

Welcome to **Week 2** of the Agentic AI Learning Series! This week focuses on understanding and implementing tools and APIs within agentic AI systems. By the end of the week, you'll have hands-on experience with tool-using agents and the ability to integrate external APIs into your AI workflows.

## Learning Objectives

- Understand the concept of "tools" in the context of agentic AI.
- Implement basic tool-use within an agent framework.
- Integrate external APIs and services with AI agents.
- Explore agent orchestration and tool calling patterns.

## Directory Structure

```
week_02_tools/
│
├── src/                # Source code for Week 2 exercises and projects
│   ├── ...             # Python modules, tool scripts, agent code
│
├── README.md           # This file
└── requirements.txt    # Python dependencies for this week
```

## Topics Covered

1. **What are Tools in Agentic AI?**
   - Definitions and motivations.
   - Examples: calculators, web search, code execution, API wrappers.

2. **Tool-Using Agents**
   - Prompting agents to use tools.
   - Tool registration and invocation.

3. **API Integration**
   - Making external API calls (e.g., OpenWeather, Wikipedia, etc.).
   - Handling API responses and errors.

4. **Agent Orchestration**
   - Managing multiple tools.
   - Decision logic for tool selection.
   - Chaining tool calls.

5. **Practical Exercises**
   - Build a simple agent that uses a calculator tool.
   - Add a web search tool to your agent.
   - Integrate a public API (e.g., weather or news).
   - Experiment with agent reasoning for tool selection.

## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Explore the Source Code**

   - All main scripts and modules are in the `src/` directory.
   - Read through and run the example agents to see tool usage in action.

3. **Try the Exercises**

   - Follow along with the notebook or Python scripts to implement your own tools and agents.
   - Extend the agent to use additional APIs or custom tools.

## Resources

- [LangChain Documentation: Tools & Agents](https://python.langchain.com/docs/modules/agents/tools/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [Awesome-Agents](https://github.com/f/awesome-agents) — Curated list of agentic AI frameworks and resources

## Assessment

- Complete the coding exercises in `src/`.
- Build and demonstrate an agent that uses at least two different tools.
- Submit your code and a short write-up describing your approach.

---

Happy learning! 🚀

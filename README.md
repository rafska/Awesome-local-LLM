# Awesome-local-LLM
A curated list of awesome platforms, tools, practices and resources that helps run LLMs locally

## Inference platforms

- [LM Studio](https://lmstudio.ai/) - discover, download and run local LLMs
- <img src="https://img.shields.io/github/stars/lemonade-sdk/lemonade?style=social" height="17"/> [lemonade](https://github.com/lemonade-sdk/lemonade) - a local LLM server with GPU and NPU Acceleration

## Inference engines

- <img src="https://img.shields.io/github/stars/ollama/ollama?style=social" height="17"/> [ollama](https://github.com/ollama/ollama) - get up and running with LLMs
- <img src="https://img.shields.io/github/stars/ggml-org/llama.cpp?style=social" height="17"/> [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++
- <img src="https://img.shields.io/github/stars/ikawrakow/ik_llama.cpp?style=social" height="17"/> [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) - llama.cpp fork with additional SOTA quants and improved performance
- <img src="https://img.shields.io/github/stars/LostRuins/koboldcpp?style=social" height="17"/> [koboldcpp](https://github.com/LostRuins/koboldcpp) - run GGUF models easily with a KoboldAI UI
- <img src="https://img.shields.io/github/stars/vllm-project/vllm?style=social" height="17"/> [vllm](https://github.com/vllm-project/vllm) - a high-throughput and memory-efficient inference and serving engine for LLMs
- <img src="https://img.shields.io/github/stars/GeeeekExplorer/nano-vllm?style=social" height="17"/> [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - a lightweight vLLM implementation built from scratch
- <img src="https://img.shields.io/github/stars/nlzy/vllm-gfx906?style=social" height="17"/> [vllm-gfx906](https://github.com/nlzy/vllm-gfx906) - vLLM for AMD gfx906 GPUs, e.g. Radeon VII / MI50 / MI60
- <img src="https://img.shields.io/github/stars/FastFlowLM/FastFlowLM?style=social" height="17"/> [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) - run LLMs on AMD Ryzen™ AI NPUs
- <img src="https://img.shields.io/github/stars/exo-explore/exo?style=social" height="17"/> [exo](https://github.com/exo-explore/exo) - run your own AI cluster at home with everyday devices
- <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" height="17"/> [sglang](https://github.com/sgl-project/sglang) - a fast serving framework for large language models and vision language models

## User Interfaces

- <img src="https://img.shields.io/github/stars/open-webui/open-webui?style=social" height="17"/> [Open WebUI](https://github.com/open-webui/open-webui) - User-friendly AI Interface (Supports Ollama, OpenAI API, ...)

## Large Language Models

### Explorers, Benchmarks, Leaderboards

- [AI Models & API Providers Analysis](https://artificialanalysis.ai/) - understand the AI landscape to choose the best model and provider for your use case
- [LLM Explorer](https://llm-explorer.com/) - explore list of the open-source LLM models
- [Dubesor LLM Benchmark table](https://dubesor.de/benchtable) - small-scale manual performance comparison benchmark
- [oobabooga benchmark](https://oobabooga.github.io/benchmark.html) - a list sorted by size (on disk) for each score

### Model providers

- [Qwen](https://huggingface.co/Qwen) - powered by Alibaba Cloud
- [Mistral AI](https://huggingface.co/mistralai) - a pioneering French artificial intelligence startup
- [Unsloth AI](https://huggingface.co/unsloth) - focusing on making AI more accessible to everyone (GGUFs etc.)
- [bartowski](https://huggingface.co/bartowski) - providing GGUF versions of popular LLMs
- [Beijing Academy of Artificial Intelligence](https://huggingface.co/BAAI) - a private non-profit organization engaged in AI research and development
- [Open Thoughts](https://huggingface.co/open-thoughts) - a team of researchers and engineers curating the best open reasoning datasets

### Specific models

- [Qwen3](https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f) - a collection of the latest generation Qwen LLMs
- [Qwen3-Coder](https://huggingface.co/collections/Qwen/qwen3-coder-687fc861e53c939e52d52d10) - a collection of the Qwen's most agentic code models to date
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17"/> [Gemma 3](https://huggingface.co/collections/google/gemma-3-release-67c6c6f89c4f76621268bb6d) - a family of lightweight, state-of-the-art open models from Google, built from the same research and technology used to create the Gemini models
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17"/> [OpenAI-o3-open](https://www.youtube.com/watch?v=hhWwFUtDAJo) - a very first OpenAI's open weight LLM model
- [Mistral-Small-3.2-24B-Instruct-2506](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) - a versatile model designed to handle a wide range of generative AI tasks, including instruction following, conversational assistance, image understanding, and function calling
- [Magistral-Small-2507](https://huggingface.co/mistralai/Magistral-Small-2507) - a Mistral Small 3.1 (2503) with added reasoning capabilities
- [Devstral-Small-2507](https://huggingface.co/mistralai/Devstral-Small-2507) - an agentic LLM for software engineering tasks fine-tuned from Mistral-Small-3.1
- [Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) - an enhancement of Mistral Small 3, incorporating state-of-the-art audio input capabilities while retaining best-in-class text performance
- [Mellum-4b-base](https://huggingface.co/JetBrains/Mellum-4b-base) - an LLM optimized for code-related tasks
- [OlympicCoder-32B](https://huggingface.co/open-r1/OlympicCoder-32B) - a code model that achieves very strong performance on competitive coding benchmarks such as LiveCodeBench and the 2024 International Olympiad in Informatics
- [NextCoder](https://huggingface.co/collections/microsoft/nextcoder-6815ee6bfcf4e42f20d45028) - a family of code-editing LLMs developed using the Qwen2.5-Coder Instruct variants as base
- [GLM-4.5](https://huggingface.co/collections/zai-org/glm-45-687c621d34bda8c9e4bf503b) - a collection of hybrid reasoning models designed for intelligent agents
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) - a lightweight open model built upon synthetic data and filtered publicly available websites
- [Granite-3.3-2B-Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) - an LLM fine-tuned for improved reasoning and instruction-following capabilities
- [Jan-nano](https://huggingface.co/Menlo/Jan-nano) - a compact 4-billion parameter language model specifically designed and trained for deep research tasks
- [Jan-nano-128k](https://huggingface.co/Menlo/Jan-nano-128k) - an enhanced version of Jan-nano features a native 128k context window that enables deeper, more comprehensive research capabilities without the performance degradation typically associated with context extension method
- [HunyuanWorld-1](https://huggingface.co/tencent/HunyuanWorld-1) - an open-source 3D world generation model
- [Arch-Router-1.5B](https://huggingface.co/katanemo/Arch-Router-1.5B) - the fastest LLM router model that aligns to subjective usage preferences

## Tools

### Coding Agents

- <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=social" height="17"/> [OpenHands](https://github.com/All-Hands-AI/OpenHands) - a platform for software development agents powered by AI
- <img src="https://img.shields.io/github/stars/cline/cline?style=social" height="17"/> [cline](https://github.com/cline/cline) - autonomous coding agent right in your IDE, capable of creating/editing files, executing commands, using the browser, and more with your permission every step of the way
- <img src="https://img.shields.io/github/stars/Aider-AI/aider?style=social" height="17"/> [aider](https://github.com/Aider-AI/aider) - AI pair programming in your terminal
- <img src="https://img.shields.io/github/stars/continuedev/continue?style=social" height="17"/> [continue](https://github.com/continuedev/continue) - create, share, and use custom AI code assistants with our open-source IDE extensions and hub of models, rules, prompts, docs, and other building blocks
- <img src="https://img.shields.io/github/stars/voideditor/void?style=social" height="17"/> [void](https://github.com/voideditor/void) - an open-source Cursor alternative, use AI agents on your codebase, checkpoint and visualize changes, and bring any model or host locally
- <img src="https://img.shields.io/github/stars/RooCodeInc/Roo-Code?style=social" height="17"/> [Roo-Code](https://github.com/RooCodeInc/Roo-Code) - a whole dev team of AI agents in your code editor
- <img src="https://img.shields.io/github/stars/block/goose?style=social" height="17"/> [goose](https://github.com/block/goose) - an open-source, extensible AI agent that goes beyond code suggestions 
- <img src="https://img.shields.io/github/stars/sst/opencode?style=social" height="17"/> [opencode](https://github.com/sst/opencode) - a AI coding agent built for the terminal
- <img src="https://img.shields.io/github/stars/Kilo-Org/kilocode?style=social" height="17"/> [kilocode](https://github.com/Kilo-Org/kilocode) - open source AI coding assistant for planning, building, and fixing code

### Agent Frameworks

- <img src="https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=social" height="17"/> [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - a powerful platform that allows you to create, deploy, and manage continuous AI agents that automate complex workflows
- <img src="https://img.shields.io/github/stars/langchain-ai/langchain?style=social" height="17"/> [langchain](https://github.com/langchain-ai/langchain) - build context-aware reasoning applications
- <img src="https://img.shields.io/github/stars/langflow-ai/langflow?style=social" height="17"/> [langflow](https://github.com/langflow-ai/langflow) - a powerful tool for building and deploying AI-powered agents and workflows
- <img src="https://img.shields.io/github/stars/microsoft/autogen?style=social" height="17"/> [autogen](https://github.com/microsoft/autogen) - a programming framework for agentic AI
- <img src="https://img.shields.io/github/stars/run-llama/llama_index?style=social" height="17"/> [llama_index](https://github.com/run-llama/llama_index) - the leading framework for building LLM-powered agents over your data
- <img src="https://img.shields.io/github/stars/crewAIInc/crewAI?style=social" height="17"/> [crewAI](https://github.com/crewAIInc/crewAI) - a framework for orchestrating role-playing, autonomous AI agents
- <img src="https://img.shields.io/github/stars/agno-agi/agno?style=social" height="17"/> [agno](https://github.com/agno-agi/agno) - a full-stack framework for building Multi-Agent Systems with memory, knowledge and reasoning
- <img src="https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=social" height="17"/> [SuperAGI](https://github.com/TransformerOptimus/SuperAGI) - an open-source framework to build, manage and run useful Autonomous AI Agents
- <img src="https://img.shields.io/github/stars/camel-ai/camel?style=social" height="17"/> [camel](https://github.com/camel-ai/camel) - the first and the best multi-agent framework
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17"/> <img src="https://img.shields.io/github/stars/openai/openai-agents-python?style=social" height="17"/> [openai-agents-python](https://github.com/openai/openai-agents-python) - a lightweight, powerful framework for multi-agent workflows
- <img src="https://img.shields.io/github/stars/badboysm890/ClaraVerse?style=social" height="17"/> [ClaraVerse](https://github.com/badboysm890/ClaraVerse) - privacy-first, fully local AI workspace with Ollama LLM chat, tool calling, agent builder, Stable Diffusion, and embedded n8n-style automation

### Computer Use

- <img src="https://img.shields.io/github/stars/OpenInterpreter/open-interpreter?style=social" height="17"/> [open-interpreter](https://github.com/OpenInterpreter/open-interpreter) - a natural language interface for computers
- <img src="https://img.shields.io/github/stars/microsoft/OmniParser?style=social" height="17"/> [OmniParser](https://github.com/microsoft/OmniParser) - a simple screen parsing tool towards pure vision based GUI agent
- <img src="https://img.shields.io/github/stars/OthersideAI/self-operating-computer?style=social" height="17"/> [self-operating-computer](https://github.com/OthersideAI/self-operating-computer) - a framework to enable multimodal models to operate a computer
- <img src="https://img.shields.io/github/stars/trycua/cua?style=social" height="17"/> [cua](https://github.com/trycua/cua) - the Docker Container for Computer-Use AI Agents
- <img src="https://img.shields.io/github/stars/simular-ai/Agent-S?style=social" height="17"/> [Agent-S](https://github.com/simular-ai/Agent-S) - an open agentic framework that uses computers like a human

### Browser Automation

- <img src="https://img.shields.io/github/stars/puppeteer/puppeteer?style=social" height="17"/> [puppeteer](https://github.com/puppeteer/puppeteer) - a JavaScript API for Chrome and Firefox
- <img src="https://img.shields.io/github/stars/microsoft/playwright?style=social" height="17"/> [playwright](https://github.com/microsoft/playwright) - a framework for Web Testing and Automation
- <img src="https://img.shields.io/github/stars/microsoft/playwright-mcp?style=social" height="17"/> [Playwright MCP server](https://github.com/microsoft/playwright-mcp) - an MCP server that provides browser automation capabilities using Playwright
- <img src="https://img.shields.io/github/stars/browser-use/browser-use?style=social" height="17"/> [browser-use](https://github.com/browser-use/browser-use) - make websites accessible for AI agents
- <img src="https://img.shields.io/github/stars/mendableai/firecrawl?style=social" height="17"/> [firecrawl](https://github.com/mendableai/firecrawl) - turn entire websites into LLM-ready markdown or structured data
- <img src="https://img.shields.io/github/stars/browserbase/stagehand?style=social" height="17"/> [stagehand](https://github.com/browserbase/stagehand) -  the AI Browser Automation Framework

### Memory Management

- <img src="https://img.shields.io/github/stars/mem0ai/mem0?style=social" height="17"/> [mem0](https://github.com/mem0ai/mem0) - universal memory layer for AI Agents
- <img src="https://img.shields.io/github/stars/letta-ai/letta?style=social" height="17"/> [letta](https://github.com/letta-ai/letta) - the stateful agents framework with memory, reasoning, and context management

### Testing, Evaluation, and Observability

- <img src="https://img.shields.io/github/stars/langfuse/langfuse?style=social" height="17"/> [langfuse](https://github.com/langfuse/langfuse) - an open-source LLM engineering platform: LLM Observability, metrics, evals, prompt management, playground, datasets. Integrates with OpenTelemetry, Langchain, OpenAI SDK, LiteLLM, and more
- <img src="https://img.shields.io/github/stars/traceloop/openllmetry?style=social" height="17"/> [openllmetry](https://github.com/traceloop/openllmetry) - an open-source observability for your LLM application, based on OpenTelemetry
- <img src="https://img.shields.io/github/stars/Giskard-AI/giskard?style=social" height="17"/> [giskard](https://github.com/Giskard-AI/giskard) - an open-source evaluation & testing for AI & LLM systems
- <img src="https://img.shields.io/github/stars/Agenta-AI/agenta?style=social" height="17"/> [agenta](https://github.com/Agenta-AI/agenta) - an open-source LLMOps platform: prompt playground, prompt management, LLM evaluation, and LLM observability all in one place

### Research

- <img src="https://img.shields.io/github/stars/assafelovic/gpt-researcher?style=social" height="17"/> [gpt-researcher](https://github.com/assafelovic/gpt-researcher) - an LLM based autonomous agent that conducts deep local and web research on any topic and generates a long report with citations
- <img src="https://img.shields.io/github/stars/LearningCircuit/local-deep-research?style=social" height="17"/> [local-deep-research](https://github.com/LearningCircuit/local-deep-research) - an AI-powered research assistant for deep, iterative research
- <img src="https://img.shields.io/github/stars/murtaza-nasir/maestro?style=social" height="17"/> [maestro](https://github.com/murtaza-nasir/maestro) - an AI-powered research application designed to streamline complex research tasks
- <img src="https://img.shields.io/github/stars/MODSetter/SurfSense?style=social" height="17"/> [SurfSense](https://github.com/MODSetter/SurfSense) - an open-source alternative to NotebookLM / Perplexity / Glean
- <img src="https://img.shields.io/github/stars/lfnovo/open-notebook?style=social" height="17"/> [open-notebook](https://github.com/lfnovo/open-notebook) - an open-source implementation of Notebook LM with more flexibility and features

### Training and Fine-tuning

- <img src="https://img.shields.io/github/stars/kiln-ai/kiln?style=social" height="17"/> [Kiln](https://github.com/kiln-ai/kiln) - the easiest tool for fine-tuning LLM models, synthetic data generation, and collaborating on datasets
- <img src="https://img.shields.io/github/stars/e-p-armstrong/augmentoolkit?style=social" height="17"/> [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) - train an open-source LLM on new facts

### Miscellaneous

- <img src="https://img.shields.io/github/stars/vanna-ai/vanna?style=social" height="17"/> [vanna](https://github.com/vanna-ai/vanna) - an open-source Python RAG framework for SQL generation and related functionality
- <img src="https://img.shields.io/github/stars/presenton/presenton?style=social" height="17"/> [presenton](https://github.com/presenton/presenton) - an open-source AI presentation generator and API
- <img src="https://img.shields.io/github/stars/VectorSpaceLab/OmniGen2?style=social" height="17"/> [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) - exploration to advanced multimodal generation
- <img src="https://img.shields.io/github/stars/TheAhmadOsman/4o-ghibli-at-home?style=social" height="17"/> [4o-ghibli-at-home](https://github.com/TheAhmadOsman/4o-ghibli-at-home) - a powerful, self-hosted AI photo stylizer built for performance and privacy
- <img src="https://img.shields.io/github/stars/Roy3838/Observer?style=social" height="17"/> [Observer](https://github.com/Roy3838/Observer) - local open-source micro-agents that observe, log and react, all while keeping your data private and secure

## Hardware

- <img src="https://img.shields.io/youtube/channel/subscribers/UCiaQzXI5528Il6r2NNkrkJA?style=social" height="17"/> [Digital Spaceport](https://www.youtube.com/@DigitalSpaceport) - reviews of various builds designed for LLM inference
- <img src="https://img.shields.io/youtube/channel/subscribers/UCQs0lwV6E4p7LQaGJ6fgy5Q?style=social" height="17"/> [JetsonHacks](https://www.youtube.com/@JetsonHacks) - information about developing on NVIDIA Jetson Development Kits
- <img src="https://img.shields.io/youtube/channel/subscribers/UC8h2Sf-yyo1WXeEUr-OHgyg?style=social" height="17"/> [Miyconst](https://www.youtube.com/@Miyconst) - tests of various types of hardware capable of running LLMs
- <img src="https://img.shields.io/youtube/channel/subscribers/UCajiMK_CY9icRhLepS8_3ug?style=social" height="17"/> [Alex Ziskind](https://www.youtube.com/@AZisk) - tests of pcs, laptops, gpus etc. capable of running LLMs
- [LLM Inference VRAM & GPU Requirement Calculator](https://app.linpp2009.com/en/llm-gpu-memory-calculator) - calculate how many GPUs you need to deploy LLMs

## Tutorials

### Prompt Engineering

- <img src="https://img.shields.io/github/stars/NirDiamant/Prompt_Engineering?style=social" height="17"/> [Prompt Engineering by NirDiamant](https://github.com/NirDiamant/Prompt_Engineering) - a comprehensive collection of tutorials and implementations for Prompt Engineering techniques, ranging from fundamental concepts to advanced strategies
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17"/> [Prompting guide 101](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf) - a quick-start handbook for effective prompts by Google
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17"/> [Prompt Engineering by Google](https://drive.google.com/file/d/1AbaBYbEa_EbPelsT40-vj64L-2IwUJHy/view) - prompt engineering by Google
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17"/> [Prompt Engineering by Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) - prompt engineering by Anthropic
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17"/> [Prompt Engineering Interactive Tutorial](https://github.com/anthropics/courses/blob/master/prompt_engineering_interactive_tutorial/README.md) - Prompt Engineering Interactive Tutorial by Anthropic
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17"/> [Real world prompting](https://github.com/anthropics/courses/blob/master/real_world_prompting/README.md) - real world prompting tutorial by Anthropic
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17"/> [Prompt evaluations](https://github.com/anthropics/courses/blob/master/prompt_evaluations/README.md) - prompt evaluations course by Anthropic
- <img src="https://img.shields.io/github/stars/x1xhlol/system-prompts-and-models-of-ai-tools?style=social" height="17"/> [system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) - a collection of system prompts extracted from AI tools
- <img src="https://img.shields.io/github/stars/upstash/context7?style=social" height="17"/> [Context7 MCP Server](https://github.com/upstash/context7) - up-to-date code documentation for LLMs and AI code editors

### Context Engineering

- <img src="https://img.shields.io/github/stars/davidkimai/Context-Engineering?style=social" height="17"/> [Context-Engineering](https://github.com/davidkimai/Context-Engineering) - a frontier, first-principles handbook inspired by Karpathy and 3Blue1Brown for moving beyond prompt engineering to the wider discipline of context design, orchestration, and optimization
- <img src="https://img.shields.io/github/stars/Meirtz/Awesome-Context-Engineering?style=social" height="17"/> [Awesome-Context-Engineering](https://github.com/Meirtz/Awesome-Context-Engineering) - a comprehensive survey on Context Engineering: from prompt engineering to production-grade AI systems

### Agents

- <img src="https://img.shields.io/github/stars/NirDiamant/GenAI_Agents?style=social" height="17"/> [GenAI Agents](https://github.com/NirDiamant/GenAI_Agents) - tutorials and implementations for various Generative AI Agent techniques
- <img src="https://img.shields.io/github/stars/NirDiamant/agents-towards-production?style=social" height="17"/> [Agents towards production](https://github.com/NirDiamant/agents-towards-production) - end-to-end, code-first tutorials covering every layer of production-grade GenAI agents, guiding you from spark to scale with proven patterns and reusable blueprints for real-world launche
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17"/> [601 real-world gen AI use cases](https://cloud.google.com/transform/101-real-world-generative-ai-use-cases-from-industry-leaders) - 601 real-world gen AI use cases from the world's leading organizations by Google
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17"/> [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) - a practical guide to building agents by OpenAI

### Retrieval-Augmented Generation

- <img src="https://img.shields.io/github/stars/NirDiamant/RAG_Techniques?style=social" height="17"/> [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) - various advanced techniques for Retrieval-Augmented Generation (RAG) systems
- <img src="https://img.shields.io/github/stars/NirDiamant/Controllable-RAG-Agent?style=social" height="17"/> [Controllable RAG Agent](https://github.com/NirDiamant/Controllable-RAG-Agent) - an advanced Retrieval-Augmented Generation (RAG) solution for complex question answering that uses sophisticated graph based algorithm to handle the tasks

### Miscellaneous

- [Self-hosted AI coding that just works](https://www.reddit.com/r/LocalLLaMA/comments/1lt4y1z/selfhosted_ai_coding_that_just_works/)

## Communities

- <img src="https://img.shields.io/reddit/subreddit-subscribers/LocalLLaMA?style=social" height="17"/> [LocalLLaMA](https://www.reddit.com/r/LocalLLaMA)
- <img src="https://img.shields.io/reddit/subreddit-subscribers/LocalLLM?style=social" height="17"/> [LocalLLM](https://www.reddit.com/r/LocalLLM)
- <img src="https://img.shields.io/reddit/subreddit-subscribers/LLMDevs?style=social" height="17"/> [LLMDevs](https://www.reddit.com/r/LLMDevs)
- <img src="https://img.shields.io/reddit/subreddit-subscribers/LocalAIServers?style=social" height="17"/> [LocalAIServers](https://www.reddit.com/r/LocalAIServers/)

# Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

# Awesome local LLM <img src="https://awesome.re/badge.svg"/>
A curated list of awesome platforms, tools, practices and resources that helps run LLMs locally

## Table of Contents

- [Inference platforms](#inference-platforms)
- [Inference engines](#inference-engines)
- [User Interfaces](#user-interfaces)
- [Large Language Models](#large-language-models)
  - [Explorers, Benchmarks, Leaderboards](#explorers-benchmarks-leaderboards)
  - [Model providers](#model-providers)
  - [Specific models](#specific-models)
    - [General purpose](#general-purpose)
    - [Coding](#coding)
    - [Multimodal](#multimodal)
    - [Image](#image)
    - [Audio](#audio)
    - [Retrieval-Augmented Generation](#retrieval-augmented-generation)
    - [Safeguards](#safeguards)
    - [Miscellaneous](#miscellaneous)
- [Tools](#tools)
  - [Models](#models)
  - [Agent Frameworks](#agent-frameworks)
  - [Model Context Protocol](#model-context-protocol)
  - [Retrieval-Augmented Generation](#retrieval-augmented-generation-1)
  - [Coding Agents](#coding-agents)
  - [Computer Use](#computer-use)
  - [Browser Automation](#browser-automation)
  - [Memory Management](#memory-management)
  - [Testing, Evaluation and Observability](#testing-evaluation-and-observability)
  - [Research](#research)
  - [Training and Fine-tuning](#training-and-fine-tuning)
  - [Security and Sandboxing](#security-and-sandboxing)
  - [Miscellaneous](#miscellaneous-1)
- [Hardware](#hardware)
- [Tutorials](#tutorials)
  - [Models](#models-1)
  - [Prompt Engineering](#prompt-engineering)
  - [Context Engineering](#context-engineering)
  - [Inference](#inference)
  - [Agents](#agents)
  - [Retrieval-Augmented Generation](#retrieval-augmented-generation-2)
  - [Miscellaneous](#miscellaneous-2)
- [Communities](#communities)

## Inference platforms

- [LM Studio](https://lmstudio.ai/) - discover, download and run local LLMs
- <img src="https://img.shields.io/github/stars/unslothai/unsloth?style=social" height="17" align="texttop"/> [unsloth](https://github.com/unslothai/unsloth) -  unified web UI for training and running open models like Qwen, DeepSeek, and Gemma locally
- <img src="https://img.shields.io/github/stars/mudler/LocalAI?style=social" height="17" align="texttop"/> [LocalAI](https://github.com/mudler/LocalAI) -  the free, open-source alternative to OpenAI, Claude and others
- <img src="https://img.shields.io/github/stars/menloresearch/jan?style=social" height="17" align="texttop"/> [jan](https://github.com/menloresearch/jan) - an open source alternative to ChatGPT that runs 100% offline on your computer
- <img src="https://img.shields.io/github/stars/ChatBoxAI/ChatBox?style=social" height="17" align="texttop"/> [ChatBox](https://github.com/ChatBoxAI/ChatBox) - user-friendly desktop client app for AI models/LLMs
- <img src="https://img.shields.io/github/stars/lemonade-sdk/lemonade?style=social" height="17" align="texttop"/> [lemonade](https://github.com/lemonade-sdk/lemonade) - a local LLM server with GPU and NPU Acceleration

[Back to Table of Contents](#table-of-contents)

## Inference engines

- <img src="https://img.shields.io/github/stars/ollama/ollama?style=social" height="17" align="texttop"/> [ollama](https://github.com/ollama/ollama) - get up and running with LLMs
- <img src="https://img.shields.io/github/stars/ggml-org/llama.cpp?style=social" height="17" align="texttop"/> [llama.cpp](https://github.com/ggml-org/llama.cpp) - LLM inference in C/C++
- <img src="https://img.shields.io/github/stars/vllm-project/vllm?style=social" height="17" align="texttop"/> [vllm](https://github.com/vllm-project/vllm) - a high-throughput and memory-efficient inference and serving engine for LLMs
- <img src="https://img.shields.io/github/stars/exo-explore/exo?style=social" height="17" align="texttop"/> [exo](https://github.com/exo-explore/exo) - run your own AI cluster at home with everyday devices
- <img src="https://img.shields.io/github/stars/microsoft/BitNet?style=social" height="17" align="texttop"/> [BitNet](https://github.com/microsoft/BitNet) - official inference framework for 1-bit LLMs
- <img src="https://img.shields.io/github/stars/sgl-project/sglang?style=social" height="17" align="texttop"/> [sglang](https://github.com/sgl-project/sglang) - a fast serving framework for large language models and vision language models
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=social" height="17" align="texttop"/> [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - provides users with an easy-to-use Python API to define Large Language Models (LLMs) and supports state-of-the-art optimizations to perform inference efficiently on NVIDIA GPUs
- <img src="https://img.shields.io/github/stars/GeeeekExplorer/nano-vllm?style=social" height="17" align="texttop"/> [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - a lightweight vLLM implementation built from scratch
- <img src="https://img.shields.io/github/stars/jundot/omlx?style=social" height="17" align="texttop"/> [omlx](https://github.com/jundot/omlx) - LLM inference server with continuous batching & SSD caching for Apple Silicon — managed from the macOS menu bar
- <img src="https://img.shields.io/github/stars/drumih/turbo-fieldfare?style=social" height="17" align="texttop"/> [TurboFieldfare](https://github.com/drumih/turbo-fieldfare) - run Gemma 4 26B-A4B on Apple Silicon Macs in about 2 GB of RAM by streaming routed experts from SSD
- <img src="https://img.shields.io/github/stars/LostRuins/koboldcpp?style=social" height="17" align="texttop"/> [koboldcpp](https://github.com/LostRuins/koboldcpp) - run GGUF models easily with a KoboldAI UI
- <img src="https://img.shields.io/github/stars/EricLBuehler/mistral.rs?style=social" height="17" align="texttop"/> [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - fast, flexible LLM inference
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/ai-dynamo/dynamo?style=social" height="17" align="texttop"/> [dynamo](https://github.com/ai-dynamo/dynamo) - a datacenter scale distributed inference serving framework
- <img src="https://img.shields.io/github/stars/flashinfer-ai/flashinfer?style=social" height="17" align="texttop"/> [flashinfer](https://github.com/flashinfer-ai/flashinfer) - kernel library for LLM serving
- <img src="https://img.shields.io/github/stars/ml-explore/mlx-lm?style=social" height="17" align="texttop"/> [mlx-lm](https://github.com/ml-explore/mlx-lm) - generate text and fine-tune large language models on Apple silicon with MLX
- <img src="https://img.shields.io/github/stars/gpustack/gpustack?style=social" height="17" align="texttop"/> [gpustack](https://github.com/gpustack/gpustack) - simple, scalable AI model deployment on GPU clusters
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/google-ai-edge/LiteRT-LM?style=social" height="17" align="texttop"/> [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) - Google's production-ready, high-performance, open-source inference framework for deploying Large Language Models on edge devices
- <img src="https://img.shields.io/github/stars/Blaizzy/mlx-vlm?style=social" height="17" align="texttop"/> [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) - a package for inference and fine-tuning of Vision Language Models (VLMs) on your Mac using MLX
- <img src="https://img.shields.io/github/stars/pytorch/executorch?style=social" height="17" align="texttop"/> [executorch](https://github.com/pytorch/executorch) - on-device AI across mobile, embedded and edge for PyTorch
- <img src="https://img.shields.io/github/stars/sgl-project/mini-sglang?style=social" height="17" align="texttop"/> [mini-sglang](https://github.com/sgl-project/mini-sglang) - a lightweight yet high-performance inference framework for Large Language Models
- <img src="https://img.shields.io/github/stars/b4rtaz/distributed-llama?style=social" height="17" align="texttop"/> [distributed-llama](https://github.com/b4rtaz/distributed-llama) - connect home devices into a powerful cluster to accelerate LLM inference
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/google-ai-edge/litert?style=social" height="17" align="texttop"/> [LiteRT](https://github.com/google-ai-edge/litert) - Google's on-device framework for high-performance ML & GenAI deployment on edge platforms, via efficient conversion, runtime, and optimization
- <img src="https://img.shields.io/github/stars/ikawrakow/ik_llama.cpp?style=social" height="17" align="texttop"/> [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) - llama.cpp fork with additional SOTA quants and improved performance
- <img src="https://img.shields.io/github/stars/dphnAI/sonar?style=social" height="17" align="texttop"/> [sonar](https://github.com/dphnAI/sonar) - large-scale LLM inference engine based on vLLM
- <img src="https://img.shields.io/github/stars/FastFlowLM/FastFlowLM?style=social" height="17" align="texttop"/> [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) - run LLMs on AMD Ryzen™ AI NPUs
- <img src="https://img.shields.io/github/stars/lightseekorg/tokenspeed?style=social" height="17" align="texttop"/> [tokenspeed](https://github.com/lightseekorg/tokenspeed) - a speed-of-light LLM inference engine
- <img src="https://img.shields.io/github/stars/brontoguana/krasis?style=social" height="17" align="texttop"/> [krasis](https://github.com/brontoguana/krasis) - a Hybrid LLM runtime which focuses on efficient running of larger models on consumer grade VRAM limited hardware
- <img src="https://img.shields.io/github/stars/nlzy/vllm-gfx906?style=social" height="17" align="texttop"/> [vllm-gfx906](https://github.com/nlzy/vllm-gfx906) - vLLM for AMD gfx906 GPUs, e.g. Radeon VII / MI50 / MI60
- <img src="https://img.shields.io/github/stars/intel/llm-scaler?style=social" height="17" align="texttop"/> [llm-scaler](https://github.com/intel/llm-scaler) - run LLMs on Intel Arc™ Pro B60 GPUs

[Back to Table of Contents](#table-of-contents)

## User Interfaces

- <img src="https://img.shields.io/github/stars/open-webui/open-webui?style=social" height="17" align="texttop"/> [Open WebUI](https://github.com/open-webui/open-webui) - User-friendly AI Interface (Supports Ollama, OpenAI API, ...)
- <img src="https://img.shields.io/github/stars/lobehub/lobe-chat?style=social" height="17" align="texttop"/> [Lobe Chat](https://github.com/lobehub/lobe-chat) - an open-source, modern design AI chat framework
- <img src="https://img.shields.io/github/stars/oobabooga/text-generation-webui?style=social" height="17" align="texttop"/> [Text generation web UI](https://github.com/oobabooga/text-generation-webui) - LLM UI with advanced features, easy setup, and multiple backend support
- <img src="https://img.shields.io/github/stars/SillyTavern/SillyTavern?style=social" height="17" align="texttop"/> [SillyTavern](https://github.com/SillyTavern/SillyTavern) - LLM Frontend for Power Users
- <img src="https://img.shields.io/github/stars/n4ze3m/page-assist?style=social" height="17" align="texttop"/> [Page Assist](https://github.com/n4ze3m/page-assist) - Use your locally running AI models to assist you in your web browsing

[Back to Table of Contents](#table-of-contents)

## Large Language Models

### Explorers, Benchmarks, Leaderboards

- [Arena](https://arena.ai/) - benchmark & compare the best AI models
- [AI Models & API Providers Analysis](https://artificialanalysis.ai/) - understand the AI landscape to choose the best model and provider for your use case
- [SWE-rebench](https://swe-rebench.com/) - a continuously evolving and decontaminated benchmark for software engineering LLMs
- <img src="https://img.shields.io/github/stars/petergpt/bullshit-benchmark?style=social" height="17" align="texttop"/> [BullshitBench](https://github.com/petergpt/bullshit-benchmark) - measure whether AI models challenge nonsensical prompts instead of confidently answering them
- [LLM Explorer](https://llm-explorer.com/) - explore list of the open-source LLM models
- [Dubesor LLM Benchmark table](https://dubesor.de/benchtable) - small-scale manual performance comparison benchmark
- [oobabooga benchmark](https://oobabooga.github.io/benchmark.html) - a list sorted by size (on disk) for each score
- [CyberGym](https://www.cybergym.io/) - evaluating AI agents' real-world cybersecurity capabilities at scale
- <img src="https://img.shields.io/github/stars/IBM/vakra?style=social" height="17" align="texttop"/> [vakra](https://github.com/IBM/vakra) -  a benchmark for evaluating multi-hop, multi-source tool-calling in AI agents

[Back to Table of Contents](#table-of-contents)

### Model providers

- [Qwen](https://huggingface.co/Qwen) - powered by Alibaba Cloud
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Mistral AI](https://huggingface.co/mistralai) - a pioneering French artificial intelligence startup
- [Tencent](https://huggingface.co/tencent) - a profile of a Chinese multinational technology conglomerate and holding company
- [Unsloth AI](https://huggingface.co/unsloth) - focusing on making AI more accessible to everyone (GGUFs etc.)
- [bartowski](https://huggingface.co/bartowski) - providing GGUF versions of popular LLMs
- [Beijing Academy of Artificial Intelligence](https://huggingface.co/BAAI) - a private non-profit organization engaged in AI research and development
- [Open Thoughts](https://huggingface.co/open-thoughts) - a team of researchers and engineers curating the best open reasoning datasets

[Back to Table of Contents](#table-of-contents)

### Specific models

#### General purpose

- [DeepSeek-V4](https://huggingface.co/collections/deepseek-ai/deepseek-v4) - a collection of the DeepSeek V4 LLMs
- [Qwen3.8](https://huggingface.co/collections/Qwen/qwen38) - a collection of the latest generation Qwen LLMs
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [NVIDIA Nemotron v3](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3) - a family of open models from NVIDIA with open weights, training data and recipes, delivering leading efficiency and accuracy for building specialized AI agents
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> [Gemma 4](https://huggingface.co/collections/google/gemma-4) - a family of open models built by Google DeepMind, that are multimodal, handling text and image input (with audio supported on small models) and generating text output
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Mistral Medium 3.5](https://huggingface.co/collections/mistralai/mistral-medium-35) - The first flaship models from Mistral AI handling instruction-following, reasoning, and coding in a single set of opened-weights
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> [gpt-oss](https://huggingface.co/collections/openai/gpt-oss-68911959590a1634ba11c7a4) - a collection of open-weight models from OpenAI, designed for powerful reasoning, agentic tasks, and versatile developer use cases
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [gpt-oss-puzzle-88B](https://huggingface.co/nvidia/gpt-oss-puzzle-88B) - a deployment-optimized large language model developed by NVIDIA, derived from OpenAI's gpt-oss-120b
- [Hunyuan](https://huggingface.co/collections/tencent/hunyuan-dense-model-6890632cda26b19119c9c5e7) - a collection of Tencent's open-source efficient LLMs designed for versatile deployment across diverse computational environments
- [Phi-4](https://huggingface.co/collections/microsoft/phi-4) - a family of small language, multi-modal and reasoning models from Microsoft
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [OpenReasoning-Nemotron](https://huggingface.co/collections/nvidia/openreasoning-nemotron-687730dae0170059860f1f01) - a collection of models from NVIDIA, trained on 5M reasoning traces for math, code and science
- [Kimi K2.5](https://huggingface.co/collections/moonshotai/kimi-k25) - a collection of open-source, native multimodal agentic models from Moonshot AI that advances practical capabilities in long-horizon coding, coding-driven design, proactive autonomous execution, and swarm-based task orchestration
- [GLM-5.3](https://huggingface.co/collections/zai-org/glm-53) - a Z.ai's flagship model for long-horizon tasks
- [Ling-3.0-flash](https://huggingface.co/inclusionAI/Ling-3.0-flash) - a native hybrid reasoning model from inclusionAI, operating with 124B total and 5.1B active parameters 
- [Granite 4.1](https://huggingface.co/collections/ibm-granite/granite-41-language-models) - efficient language models from IBM for multilingual generation, coding, RAG, and AI assistant workflows
- [EXAONE-4.5](https://huggingface.co/collections/LGAI-EXAONE/exaone-45) - LG's First Open-Weight Vision-Language Model for Industrial Intelligence
- [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) - most capable open-source foundation model, engineered to deliver frontier reasoning and agentic capabilities with exceptional efficiency
- [Nex-N2](https://huggingface.co/collections/nex-agi/nex-n2) - a collection of agent models built for real-world productivity scenarios
- [MiniCPM5](https://huggingface.co/collections/openbmb/minicpm5) - a collection of SOTA on-device LLMs, small yet powerful

[Back to Table of Contents](#table-of-contents)

#### Coding

- [Qwen3-Coder-Next](https://huggingface.co/collections/Qwen/qwen3-coder-next) - a collection of Qwen's open-weight language models designed specifically for coding agents and local development
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Devstral 2](https://huggingface.co/collections/mistralai/devstral-2) - a couple of agentic LLMs for software engineering tasks, excelling at using tools to explore codebases, edit multiple files, and power SWE Agents
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Mellum 2](https://huggingface.co/collections/JetBrains/mellum-2) - an assistant model trained by JetBrain
- [MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) - a native multimodal model with 1M context
- [MiniMax-M2](https://huggingface.co/collections/MiniMaxAI/minimax-m2) - a collection of SOTA models for real-world dev & agents
- [Laguna-S-2.1](https://huggingface.co/poolside/Laguna-S-2.1) - a 118B total parameter Mixture-of-Experts model with 8B activated parameters per token, designed for agentic coding and long-horizon work
- [SWE-FastContext](https://huggingface.co/collections/microsoft/swe-fastcontext) - a family of code-search models from Microsoft powering the Explore subagent for coding agents
- [OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B) - a 9-billion parameter coding agent model built by Tesslate, fine-tuned on top of Qwen3.5-9B's hybrid architecture
- [NousCoder-14B](https://huggingface.co/NousResearch/NousCoder-14B) - a competitive programming model post-trained on Qwen3-14B via reinforcement learning
- [MusaCoder-27B](https://huggingface.co/MooreThreads/MusaCoder-27B) - a code model developed by Moore Threads for PyTorch-to-CUDA/MUSA native kernel generation

[Back to Table of Contents](#table-of-contents)

#### Multimodal

- [Qwen3-Omni](https://huggingface.co/collections/Qwen/qwen3-omni-68d100a86cd0906843ceccbe) - a collection of the natively end-to-end multilingual omni-modal foundation models from Qwen
- [GLM-4.6V](https://huggingface.co/collections/zai-org/glm-46v) - a collection of open source multimodal models with native tool use from Zhipu AI

[Back to Table of Contents](#table-of-contents)

#### Image

- [Qwen-Image](https://huggingface.co/collections/Qwen/qwen-image) - a collection of models for image generation, edit and decomposition from Qwen
- [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe) - a collection of the most powerful vision-language models in the Qwen series to date
- [GLM-Image](https://huggingface.co/zai-org/GLM-Image) - an image generation model
- [Granite Vision](https://huggingface.co/collections/ibm-granite/granite-vision) - multimodal models from IBM built for visual document analysis and image understanding
- [HunyuanImage](https://huggingface.co/collections/tencent/hunyuanimage) - a collection of image generation models from Tencent
- [HunyuanVideo](https://huggingface.co/collections/tencent/hunyuanvideo) - a collection of video generation models from Tencent 
- [Vidi](https://huggingface.co/collections/bytedance-research/vidi) - a collection of models for multimodal video understanding and creation
- [FastVLM](https://huggingface.co/collections/apple/fastvlm-68ac97b9cd5cacefdd04872e) - a collection of VLMs with efficient vision encoding from Apple
- [MiniCPM-o & MiniCPM-V](https://huggingface.co/collections/openbmb/minicpm-o-and-minicpm-v) - multimodal models with leading performance
- [LFM2-VL](https://huggingface.co/collections/LiquidAI/lfm2-vl-68963bbc84a610f7638d5ffa) - a colection of vision-language models, designed for on-device deployment
- [ClipTagger-12b](https://huggingface.co/inference-net/ClipTagger-12b) -  a vision-language model (VLM) designed for video understanding at massive scale

[Back to Table of Contents](#table-of-contents)

#### Audio

- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) - a state-of-the-art model for automatic speech recognition (ASR) and speech translation from OpenAI
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [Nemotron Speech](https://huggingface.co/collections/nvidia/nemotron-speech) - a collection of open, state-of-the-art, production‑ready enterprise speech models from NVIDIA for ASR, TTS, Speaker Diarization and S2SOpenAI
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [NVIDIA NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) - a 11B end-to-end, real-time speech full duplex (FD) model from NVIDIA for conversational AI that jointly performs streaming speech understanding and speech generation
- [Qwen3-ASR](https://huggingface.co/collections/Qwen/qwen3-asr) - a collection of models that support language identification and ASR for 52 languages and dialects
- [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) - a collection of TTS models that cover 10 major languages as well as multiple dialectal voice profiles to meet global application needs
- [Granite Speech](https://huggingface.co/collections/ibm-granite/granite-speech) - a collection of compact and efficient speech-language models from IBM, specifically designed for multilingual automatic speech recognition (ASR) and bidirectional automatic speech translation (AST)
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Voxtral-Small-24B-2507](https://huggingface.co/mistralai/Voxtral-Small-24B-2507) - an enhancement of Mistral Small 3, incorporating state-of-the-art audio input capabilities while retaining best-in-class text performance
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Voxtral-Mini-4B-Realtime-2602](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) - a multilingual, realtime speech-transcription model and among the first open-source solutions to achieve accuracy comparable to offline systems with a delay of <500ms
- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) - frontier, open-weights text-to-speech model that’s fast, instantly adaptable, and produces lifelike speech for voice agents
- [chatterbox](https://huggingface.co/ResembleAI/chatterbox) - first production-grade open-source TTS model
- [VibeVoice](https://huggingface.co/collections/microsoft/vibevoice-68a2ef24a875c44be47b034f) - a collection of frontier text-to-speech models from Microsoft
- [Kitten TTS](https://huggingface.co/KittenML/models) - a collection of open-source realistic text-to-speech models designed for lightweight deployment and high-quality voice synthesis
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [Streaming Sortformer Diarizer 4spk v2.1](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) - a streaming version of a novel end-to-end neural model for speaker diarization from NVIDIA

[Back to Table of Contents](#table-of-contents)

#### Retrieval-Augmented Generation

- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [Nemotron RAG](https://huggingface.co/collections/nvidia/nemotron-rag) - a set of tools to build retrieval-augmented generation (RAG) systems, improve search and ranking accuracy, and extract structured data from complex docs
- [Qwen3-Embedding](https://huggingface.co/collections/Qwen/qwen3-embedding) - a collection of the latest proprietary Qwen models, specifically designed for text embedding and ranking tasks
- [Qwen3-VL-Embedding](https://huggingface.co/collections/Qwen/qwen3-vl-embedding) - an addition to the Qwen embedding models, specifically designed for multimodal information retrieval and cross-modal understanding
- [Qwen3-Reranker](https://huggingface.co/collections/Qwen/qwen3-vl-reranker) - a collection of the latest proprietary Qwen models, engineered to refine embedding results
- [Qwen3-VL-Reranker](https://huggingface.co/collections/Qwen/qwen3-vl-reranker) - an addition to the Qwen embedding models, specifically designed for multimodal information retrieval and cross-modal understanding

[Back to Table of Contents](#table-of-contents)

#### Safeguards

- <img src="https://img.shields.io/badge/Mistral%20AI-%23FA520F?logo=mistralai&logoColor=%23FFFFFF" height="17" align="texttop"/> [Shieldstral 1.0 3B](https://huggingface.co/mistralai/Shieldstral-1.0-3B) - a compact 3B-parameter, policy-adaptive multimodal safety classifier
- [Granite Guardian](https://huggingface.co/collections/ibm-granite/granite-guardian) - a collection of safety models from IBM for detecting risks, toxicity, and hallucinations in LLM workflows
- [Qwen3Guard](https://huggingface.co/collections/Qwen/qwen3guard-68d2729abbfae4716f3343a1) - a collection of safety moderation models built upon Qwen3
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [NemoGuard](https://huggingface.co/collections/nvidia/nemoguard) - a collection of models from NVIDIA for content safety, topic-following and security guardrails
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [Nemotron-3.5-Content-Safety](https://huggingface.co/nvidia/Nemotron-3.5-Content-Safety) - a small language model (SLM) that uses Google's Gemma-3-4B-it as the base and is fine-tuned by NVIDIA on multimodal, multilingual, and reasoning-oriented content-safety datasets
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [Privasis](https://huggingface.co/collections/nvidia/privasis) - a collection of lightweight text-sanitization models from NVIDIA designed to remove or abstract sensitive information from text according to a user-provided sanitization instruction
- [SingGuard](https://huggingface.co/collections/inclusionAI/singguard) - a collection of policy-adaptive multimodal LLM Guardrails with dynamic reasoning 
- [HARC](https://huggingface.co/collections/microsoft/harc) - a family of safety-aligned instruction models from Microsoft trained with HARC
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> [gpt-oss-safeguard](https://huggingface.co/collections/openai/gpt-oss-safeguard) - a collection of safety reasoning models built-upon gpt-oss from OpenAI
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> [privacy-filter](https://huggingface.co/openai/privacy-filter) - a bidirectional token-classification model from OpenAI for personally identifiable information (PII) detection and masking in text
- [AprielGuard](https://huggingface.co/ServiceNow-AI/AprielGuard) - a safeguard model designed to detect and mitigate both safety risks and security threats in LLM interactions

[Back to Table of Contents](#table-of-contents)

#### Miscellaneous

- [Intern-S2](https://huggingface.co/collections/internlm/intern-s2) - a collection of multimodal foundation models for scientific intelligence and long-horizon agents
- [Fara1.5](https://huggingface.co/collections/microsoft/fara15) - a collection of multimodal computer use agents (CUA) for web browsers from Microsoft
- [Marco-MoE](https://huggingface.co/collections/AIDC-AI/marco-moe) - a suit of multilingual MoE models with highly-sparse architectures
- [Jan-v3](https://huggingface.co/collections/janhq/jan-v3) - a 4B baseline model for fine-tuning, designed for downstream work: improved instruction following out of the box, strong starting point for fine-tuning and effective lightweight coding assistance
- [Jan-v2-VL](https://huggingface.co/collections/janhq/jan-v2-vl) - a family of VLM focused on reliable, many-step task execution
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> [Nemotron-Orchestrator-8B](https://huggingface.co/nvidia/Nemotron-Orchestrator-8B) - a state-of-the-art 8B orchestration model designed to solve complex, multi-turn agentic tasks by coordinating a diverse set of expert models and tools
- [Arch-Router-1.5B](https://huggingface.co/katanemo/Arch-Router-1.5B) - the fastest LLM router model that aligns to subjective usage preferences
- [Waypoint](https://huggingface.co/collections/Overworld/waypoint) - a collection of real-time interactive video world models
- [Hunyuan3D](https://huggingface.co/collections/tencent/hunyuan3d) - a collection of everything related (models, datasets etc.) to 3D assets generation from Tencent
- [Hunyuan-GameCraft-1.0](https://huggingface.co/tencent/Hunyuan-GameCraft-1.0) - a novel framework for high-dynamic interactive video generation in game environments
- [void-model](https://huggingface.co/netflix/void-model) - a model from Netflix that removes objects from videos along with all interactions they induce on the scene — not just secondary effects like shadows and reflections, but physical interactions like objects falling when a person is removed

[Back to Table of Contents](#table-of-contents)

## Tools

### Models

- <img src="https://img.shields.io/github/stars/AlexsJones/llmfit?style=social" height="17" align="texttop"/> [llmfit](https://github.com/AlexsJones/llmfit) - hundreds of models & providers, one command to find what runs on your hardware
- <img src="https://img.shields.io/github/stars/dottxt-ai/outlines?style=social" height="17" align="texttop"/> [outlines](https://github.com/dottxt-ai/outlines) - structured outputs for LLMs
- <img src="https://img.shields.io/github/stars/mostlygeek/llama-swap?style=social" height="17" align="texttop"/> [llama-swap](https://github.com/mostlygeek/llama-swap) - reliable model swapping for any local OpenAI compatible server - llama.cpp, vllm, etc.
- <img src="https://img.shields.io/github/stars/guidance-ai/llguidance?style=social" height="17" align="texttop"/> [llguidance](https://github.com/guidance-ai/llguidance) - super-fast structured outputs

[Back to Table of Contents](#table-of-contents)

### Agent Frameworks

- <img src="https://img.shields.io/github/stars/Significant-Gravitas/AutoGPT?style=social" height="17" align="texttop"/> [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) - a powerful platform that allows you to create, deploy, and manage continuous AI agents that automate complex workflows
- <img src="https://img.shields.io/github/stars/langflow-ai/langflow?style=social" height="17" align="texttop"/> [langflow](https://github.com/langflow-ai/langflow) - a powerful tool for building and deploying AI-powered agents and workflows
- <img src="https://img.shields.io/github/stars/langchain-ai/langchain?style=social" height="17" align="texttop"/> [langchain](https://github.com/langchain-ai/langchain) - build context-aware reasoning applications
- <img src="https://img.shields.io/github/stars/Mintplex-Labs/anything-llm?style=social" height="17" align="texttop"/> [anything-llm](https://github.com/Mintplex-Labs/anything-llm) - the all-in-one Desktop & Docker AI application with built-in RAG, AI agents, No-code agent builder, MCP compatibility, and more
- <img src="https://img.shields.io/github/stars/microsoft/autogen?style=social" height="17" align="texttop"/> [autogen](https://github.com/microsoft/autogen) - a programming framework for agentic AI
- <img src="https://img.shields.io/github/stars/FlowiseAI/Flowise?style=social" height="17" align="texttop"/> [Flowise](https://github.com/FlowiseAI/Flowise) - build AI agents, visually
- <img src="https://img.shields.io/github/stars/earendil-works/pi?style=social" height="17" align="texttop"/> [pi](https://github.com/earendil-works/pi) - AI agent toolkit: coding agent CLI, unified LLM API, TUI & web UI libraries, Slack bot, vLLM pods
- <img src="https://img.shields.io/github/stars/run-llama/llama_index?style=social" height="17" align="texttop"/> [llama_index](https://github.com/run-llama/llama_index) - the leading framework for building LLM-powered agents over your data
- <img src="https://img.shields.io/github/stars/crewAIInc/crewAI?style=social" height="17" align="texttop"/> [crewAI](https://github.com/crewAIInc/crewAI) - a framework for orchestrating role-playing, autonomous AI agents
- <img src="https://img.shields.io/github/stars/agno-agi/agno?style=social" height="17" align="texttop"/> [agno](https://github.com/agno-agi/agno) - a full-stack framework for building Multi-Agent Systems with memory, knowledge and reasoning
- <img src="https://img.shields.io/github/stars/simstudioai/sim?style=social" height="17" align="texttop"/> [sim](https://github.com/simstudioai/sim) - open-source platform to build and deploy AI agent workflows
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/openai/openai-agents-python?style=social" height="17" align="texttop"/> [openai-agents-python](https://github.com/openai/openai-agents-python) - a lightweight, powerful framework for multi-agent workflows
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA/NemoClaw?style=social" height="17" align="texttop"/> [NemoClaw](https://github.com/NVIDIA/NemoClaw) - run OpenClaw more securely inside NVIDIA OpenShell with managed inference
- <img src="https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=social" height="17" align="texttop"/> [SuperAGI](https://github.com/TransformerOptimus/SuperAGI) - an open-source framework to build, manage and run useful Autonomous AI Agents
- <img src="https://img.shields.io/github/stars/camel-ai/camel?style=social" height="17" align="texttop"/> [camel](https://github.com/camel-ai/camel) - the first and the best multi-agent framework
- <img src="https://img.shields.io/github/stars/pydantic/pydantic-ai?style=social" height="17" align="texttop"/> [pydantic-ai](https://github.com/pydantic/pydantic-ai) - a Python agent framework designed to help you quickly, confidently, and painlessly build production grade applications and workflows with Generative AI
- <img src="https://img.shields.io/github/stars/neuml/txtai?style=social" height="17" align="texttop"/> [txtai](https://github.com/neuml/txtai) - all-in-one open-source AI framework for semantic search, LLM orchestration and language model workflows
- <img src="https://img.shields.io/github/stars/microsoft/agent-framework?style=social" height="17" align="texttop"/> [agent-framework](https://github.com/microsoft/agent-framework) - a framework for building, orchestrating and deploying AI agents and multi-agent workflows with support for Python and .NET
- <img src="https://img.shields.io/github/stars/katanemo/archgw?style=social" height="17" align="texttop"/> [archgw](https://github.com/katanemo/archgw) - a high-performance proxy server that handles the low-level work in building agents: like applying guardrails, routing prompts to the right agent, and unifying access to LLMs, etc.
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/genkit-ai/genkit?style=social" height="17" align="texttop"/> [genkit](https://github.com/genkit-ai/genkit) - open-source framework for building AI-powered apps in JavaScript, Go, and Python, built and used in production by Google
- <img src="https://img.shields.io/github/stars/badboysm890/ClaraVerse?style=social" height="17" align="texttop"/> [ClaraVerse](https://github.com/badboysm890/ClaraVerse) - privacy-first, fully local AI workspace with Ollama LLM chat, tool calling, agent builder, Stable Diffusion, and embedded n8n-style automation
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA/NeMo-Agent-Toolkit?style=social" height="17" align="texttop"/> [NeMo-Agent-Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) - an open-source library for efficiently connecting and optimizing teams of AI agents
- <img src="https://img.shields.io/github/stars/deepsense-ai/ragbits?style=social" height="17" align="texttop"/> [ragbits](https://github.com/deepsense-ai/ragbits) - building blocks for rapid development of GenAI applications

[Back to Table of Contents](#table-of-contents)

### Model Context Protocol

- <img src="https://img.shields.io/github/stars/mindsdb/mindsdb?style=social" height="17" align="texttop"/> [mindsdb](https://github.com/mindsdb/mindsdb) - federated query engine for AI - the only MCP Server you'll ever need 
- <img src="https://img.shields.io/github/stars/github/github-mcp-server?style=social" height="17" align="texttop"/> [github-mcp-server](https://github.com/github/github-mcp-server) - GitHub's official MCP Server
- <img src="https://img.shields.io/github/stars/microsoft/playwright-mcp?style=social" height="17" align="texttop"/> [playwright-mcp](https://github.com/microsoft/playwright-mcp) - Playwright MCP server
- <img src="https://img.shields.io/github/stars/ChromeDevTools/chrome-devtools-mcp?style=social" height="17" align="texttop"/> [chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) - Chrome DevTools for coding agents
- <img src="https://img.shields.io/github/stars/czlonkowski/n8n-mcp?style=social" height="17" align="texttop"/> [n8n-mcp](https://github.com/czlonkowski/n8n-mcp) - a MCP for Claude Desktop / Claude Code / Windsurf / Cursor to build n8n workflows for you
- <img src="https://img.shields.io/github/stars/awslabs/mcp?style=social" height="17" align="texttop"/> [awslabs/mcp](https://github.com/awslabs/mcp) - AWS MCP Servers — helping you get the most out of AWS, wherever you use MCP
- <img src="https://img.shields.io/github/stars/sooperset/mcp-atlassian?style=social" height="17" align="texttop"/> [mcp-atlassian](https://github.com/sooperset/mcp-atlassian) - MCP server for Atlassian tools (Confluence, Jira)
- <img src="https://img.shields.io/github/stars/bytebase/dbhub?style=social" height="17" align="texttop"/> [dbhub](https://github.com/bytebase/dbhub) - zero-dependency, token-efficient database MCP server for Postgres, MySQL, SQL Server, MariaDB, SQLite

[Back to Table of Contents](#table-of-contents)

### Retrieval-Augmented Generation

- <img src="https://img.shields.io/github/stars/pathwaycom/pathway?style=social" height="17" align="texttop"/> [pathway](https://github.com/pathwaycom/pathway) - Python ETL framework for stream processing, real-time analytics, LLM pipelines and RAG
- <img src="https://img.shields.io/github/stars/microsoft/graphrag?style=social" height="17" align="texttop"/> [graphrag](https://github.com/microsoft/graphrag) - a modular graph-based RAG system
- <img src="https://img.shields.io/github/stars/HKUDS/LightRAG?style=social" height="17" align="texttop"/> [LightRAG](https://github.com/HKUDS/LightRAG) - simple and fast RAG
- <img src="https://img.shields.io/github/stars/deepset-ai/haystack?style=social" height="17" align="texttop"/> [haystack](https://github.com/deepset-ai/haystack) - AI orchestration framework to build customizable, production-ready LLM applications, best suited for building RAG, question answering, semantic search or conversational agent chatbots
- <img src="https://img.shields.io/github/stars/vanna-ai/vanna?style=social" height="17" align="texttop"/> [vanna](https://github.com/vanna-ai/vanna) - an open-source Python RAG framework for SQL generation and related functionality
- <img src="https://img.shields.io/github/stars/getzep/graphiti?style=social" height="17" align="texttop"/> [graphiti](https://github.com/getzep/graphiti) - build real-time knowledge graphs for AI Agents
- <img src="https://img.shields.io/github/stars/onyx-dot-app/onyx?style=social" height="17" align="texttop"/> [onyx](https://github.com/onyx-dot-app/onyx) - the AI platform connected to your company's docs, apps, and people
- <img src="https://img.shields.io/github/stars/zilliztech/claude-context?style=social" height="17" align="texttop"/> [claude-context](https://github.com/zilliztech/claude-context) - make entire codebase the context for any coding agent
- <img src="https://img.shields.io/github/stars/pipeshub-ai/pipeshub-ai?style=social" height="17" align="texttop"/> [pipeshub-ai](https://github.com/pipeshub-ai/pipeshub-ai) - a fully extensible and explainable workplace AI platform for enterprise search and workflow automation

[Back to Table of Contents](#table-of-contents)

### Coding Agents

- <img src="https://img.shields.io/github/stars/sst/opencode?style=social" height="17" align="texttop"/> [opencode](https://github.com/sst/opencode) - a AI coding agent built for the terminal
- <img src="https://img.shields.io/github/stars/zed-industries/zed?style=social" height="17" align="texttop"/> [zed](https://github.com/zed-industries/zed) - a next-generation code editor designed for high-performance collaboration with humans and AI
- <img src="https://img.shields.io/github/stars/All-Hands-AI/OpenHands?style=social" height="17" align="texttop"/> [OpenHands](https://github.com/All-Hands-AI/OpenHands) - a platform for software development agents powered by AI
- <img src="https://img.shields.io/github/stars/cline/cline?style=social" height="17" align="texttop"/> [cline](https://github.com/cline/cline) - autonomous coding agent right in your IDE, capable of creating/editing files, executing commands, using the browser, and more with your permission every step of the way
- <img src="https://img.shields.io/github/stars/Aider-AI/aider?style=social" height="17" align="texttop"/> [aider](https://github.com/Aider-AI/aider) - AI pair programming in your terminal
- <img src="https://img.shields.io/github/stars/TabbyML/tabby?style=social" height="17" align="texttop"/> [tabby](https://github.com/TabbyML/tabby) -  an open-source GitHub Copilot alternative, set up your own LLM-powered code completion server
- <img src="https://img.shields.io/github/stars/continuedev/continue?style=social" height="17" align="texttop"/> [continue](https://github.com/continuedev/continue) - create, share, and use custom AI code assistants with our open-source IDE extensions and hub of models, rules, prompts, docs, and other building blocks
- <img src="https://img.shields.io/github/stars/voideditor/void?style=social" height="17" align="texttop"/> [void](https://github.com/voideditor/void) - an open-source Cursor alternative, use AI agents on your codebase, checkpoint and visualize changes, and bring any model or host locally
- <img src="https://img.shields.io/github/stars/block/goose?style=social" height="17" align="texttop"/> [goose](https://github.com/block/goose) - an open-source, extensible AI agent that goes beyond code suggestions 
- <img src="https://img.shields.io/github/stars/RooCodeInc/Roo-Code?style=social" height="17" align="texttop"/> [Roo-Code](https://github.com/RooCodeInc/Roo-Code) - a whole dev team of AI agents in your code editor
- <img src="https://img.shields.io/github/stars/charmbracelet/crush?style=social" height="17" align="texttop"/> [crush](https://github.com/charmbracelet/crush) - the glamourous AI coding agent for your favourite terminal
- <img src="https://img.shields.io/github/stars/Kilo-Org/kilocode?style=social" height="17" align="texttop"/> [kilocode](https://github.com/Kilo-Org/kilocode) - open source AI coding assistant for planning, building, and fixing code
- <img src="https://img.shields.io/github/stars/humanlayer/humanlayer?style=social" height="17" align="texttop"/> [humanlayer](https://github.com/humanlayer/humanlayer) - the best way to get AI coding agents to solve hard problems in complex codebases
- <img src="https://img.shields.io/github/stars/ThePrimeagen/99?style=social" height="17" align="texttop"/> [99](https://github.com/ThePrimeagen/99) - neovim AI agent done right
- <img src="https://img.shields.io/github/stars/carlrobertoh/ProxyAI?style=social" height="17" align="texttop"/> [ProxyAI](https://github.com/carlrobertoh/ProxyAI) - the leading open-source AI copilot for JetBrains

[Back to Table of Contents](#table-of-contents)

### Computer Use

- <img src="https://img.shields.io/github/stars/OpenInterpreter/open-interpreter?style=social" height="17" align="texttop"/> [open-interpreter](https://github.com/OpenInterpreter/open-interpreter) - a natural language interface for computers
- <img src="https://img.shields.io/github/stars/microsoft/OmniParser?style=social" height="17" align="texttop"/> [OmniParser](https://github.com/microsoft/OmniParser) - a simple screen parsing tool towards pure vision based GUI agent
- <img src="https://img.shields.io/github/stars/different-ai/openwork?style=social" height="17" align="texttop"/> [openwork](https://github.com/different-ai/openwork) - an open-source alternative to Claude Cowork, powered by OpenCode
- <img src="https://img.shields.io/github/stars/trycua/cua?style=social" height="17" align="texttop"/> [cua](https://github.com/trycua/cua) - the Docker Container for Computer-Use AI Agents
- <img src="https://img.shields.io/github/stars/simular-ai/Agent-S?style=social" height="17" align="texttop"/> [Agent-S](https://github.com/simular-ai/Agent-S) - an open agentic framework that uses computers like a human
- <img src="https://img.shields.io/github/stars/OthersideAI/self-operating-computer?style=social" height="17" align="texttop"/> [self-operating-computer](https://github.com/OthersideAI/self-operating-computer) - a framework to enable multimodal models to operate a computer
- <img src="https://img.shields.io/github/stars/MiniMax-AI/OpenRoom?style=social" height="17" align="texttop"/> [OpenRoom](https://github.com/MiniMax-AI/OpenRoom) - a browser-based desktop where AI Agent operates every app through natural language, from MiniMaxAI

[Back to Table of Contents](#table-of-contents)

### Browser Automation

- <img src="https://img.shields.io/github/stars/puppeteer/puppeteer?style=social" height="17" align="texttop"/> [puppeteer](https://github.com/puppeteer/puppeteer) - a JavaScript API for Chrome and Firefox
- <img src="https://img.shields.io/github/stars/microsoft/playwright?style=social" height="17" align="texttop"/> [playwright](https://github.com/microsoft/playwright) - a framework for Web Testing and Automation
- <img src="https://img.shields.io/github/stars/browser-use/browser-use?style=social" height="17" align="texttop"/> [browser-use](https://github.com/browser-use/browser-use) - make websites accessible for AI agents
- <img src="https://img.shields.io/github/stars/mendableai/firecrawl?style=social" height="17" align="texttop"/> [firecrawl](https://github.com/mendableai/firecrawl) - turn entire websites into LLM-ready markdown or structured data
- <img src="https://img.shields.io/github/stars/browserbase/stagehand?style=social" height="17" align="texttop"/> [stagehand](https://github.com/browserbase/stagehand) -  the AI Browser Automation Framework
- <img src="https://img.shields.io/github/stars/nanobrowser/nanobrowser?style=social" height="17" align="texttop"/> [nanobrowser](https://github.com/nanobrowser/nanobrowser) -  open-source Chrome extension for AI-powered web automation

[Back to Table of Contents](#table-of-contents)

### Memory Management

- <img src="https://img.shields.io/github/stars/mem0ai/mem0?style=social" height="17" align="texttop"/> [mem0](https://github.com/mem0ai/mem0) - universal memory layer for AI Agents
- <img src="https://img.shields.io/github/stars/milla-jovovich/mempalace?style=social" height="17" align="texttop"/> [mempalace](https://github.com/milla-jovovich/mempalace) - the highest-scoring AI memory system ever benchmarked
- <img src="https://img.shields.io/github/stars/letta-ai/letta?style=social" height="17" align="texttop"/> [letta](https://github.com/letta-ai/letta) - the stateful agents framework with memory, reasoning, and context management
- <img src="https://img.shields.io/github/stars/supermemoryai/supermemory?style=social" height="17" align="texttop"/> [supermemory](https://github.com/supermemoryai/supermemory) - memory engine and app that is extremely fast, scalable
- <img src="https://img.shields.io/github/stars/topoteretes/cognee?style=social" height="17" align="texttop"/> [cognee](https://github.com/topoteretes/cognee) - memory for AI Agents in 5 lines of code
- <img src="https://img.shields.io/github/stars/LMCache/LMCache?style=social" height="17" align="texttop"/> [LMCache](https://github.com/LMCache/LMCache) - supercharge your LLM with the fastest KV Cache Layer
- <img src="https://img.shields.io/github/stars/NevaMind-AI/memU?style=social" height="17" align="texttop"/> [memU](https://github.com/NevaMind-AI/memU) - an open-source memory framework for AI companions
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/google-research/reasoning-bank?style=social" height="17" align="texttop"/> [reasoning-bank](https://github.com/google-research/reasoning-bank) - a memory mechanism for agents that learns from both successful and failed trajectories, with reasoning stored as memory content

[Back to Table of Contents](#table-of-contents)

### Testing, Evaluation and Observability

- <img src="https://img.shields.io/github/stars/langfuse/langfuse?style=social" height="17" align="texttop"/> [langfuse](https://github.com/langfuse/langfuse) - an open-source LLM engineering platform: LLM Observability, metrics, evals, prompt management, playground, datasets. Integrates with OpenTelemetry, Langchain, OpenAI SDK, LiteLLM, and more
- <img src="https://img.shields.io/github/stars/comet-ml/opik?style=social" height="17" align="texttop"/> [opik](https://github.com/comet-ml/opik) - debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards
- <img src="https://img.shields.io/github/stars/traceloop/openllmetry?style=social" height="17" align="texttop"/> [openllmetry](https://github.com/traceloop/openllmetry) - an open-source observability for your LLM application, based on OpenTelemetry
- <img src="https://img.shields.io/github/stars/Giskard-AI/giskard?style=social" height="17" align="texttop"/> [giskard](https://github.com/Giskard-AI/giskard) - an open-source evaluation & testing for AI & LLM systems
- <img src="https://img.shields.io/github/stars/Agenta-AI/agenta?style=social" height="17" align="texttop"/> [agenta](https://github.com/Agenta-AI/agenta) - an open-source LLMOps platform: prompt playground, prompt management, LLM evaluation, and LLM observability all in one place
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA-NeMo/evaluator?style=social" height="17" align="texttop"/> [Evaluator](https://github.com/NVIDIA-NeMo/evaluator) - open-source library for scalable, reproducible evaluation of AI models and benchmarks

[Back to Table of Contents](#table-of-contents)

### Research

- <img src="https://img.shields.io/github/stars/ItzCrazyKns/Perplexica?style=social" height="17" align="texttop"/> [Perplexica](https://github.com/ItzCrazyKns/Perplexica) -  an open-source alternative to Perplexity AI, the AI-powered search engine
- <img src="https://img.shields.io/github/stars/assafelovic/gpt-researcher?style=social" height="17" align="texttop"/> [gpt-researcher](https://github.com/assafelovic/gpt-researcher) - an LLM based autonomous agent that conducts deep local and web research on any topic and generates a long report with citations
- <img src="https://img.shields.io/github/stars/MODSetter/SurfSense?style=social" height="17" align="texttop"/> [SurfSense](https://github.com/MODSetter/SurfSense) - an open-source alternative to NotebookLM / Perplexity / Glean
- <img src="https://img.shields.io/github/stars/lfnovo/open-notebook?style=social" height="17" align="texttop"/> [open-notebook](https://github.com/lfnovo/open-notebook) - an open-source implementation of Notebook LM with more flexibility and features
- <img src="https://img.shields.io/github/stars/microsoft/RD-Agent?style=social" height="17" align="texttop"/> [RD-Agent](https://github.com/microsoft/RD-Agent) - automate the most critical and valuable aspects of the industrial R&D process
- <img src="https://img.shields.io/github/stars/langchain-ai/local-deep-researcher?style=social" height="17" align="texttop"/> [local-deep-researcher](https://github.com/langchain-ai/local-deep-researcher) - fully local web research and report writing assistant
- <img src="https://img.shields.io/github/stars/LearningCircuit/local-deep-research?style=social" height="17" align="texttop"/> [local-deep-research](https://github.com/LearningCircuit/local-deep-research) - an AI-powered research assistant for deep, iterative research
- <img src="https://img.shields.io/github/stars/murtaza-nasir/maestro?style=social" height="17" align="texttop"/> [maestro](https://github.com/murtaza-nasir/maestro) - an AI-powered research application designed to streamline complex research tasks

[Back to Table of Contents](#table-of-contents)

### Training and Fine-tuning

- <img src="https://img.shields.io/github/stars/p-e-w/heretic?style=social" height="17" align="texttop"/> [heretic](https://github.com/p-e-w/heretic) - fully automatic censorship removal for language models
- <img src="https://img.shields.io/github/stars/huggingface/sentence-transformers?style=social" height="17" align="texttop"/> [sentence-transformers](https://github.com/huggingface/sentence-transformers) - a Python library for using and training embedding and reranker models for applications like retrieval augmented generation, semantic search, and more
- <img src="https://img.shields.io/github/stars/huggingface/trl?style=social" height="17" align="texttop"/> [trl](https://github.com/huggingface/trl) - train transformer language models with reinforcement learning
- <img src="https://img.shields.io/github/stars/OpenRLHF/OpenRLHF?style=social" height="17" align="texttop"/> [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) - an easy-to-use, high-performance open-source RLHF framework built on Ray, vLLM, ZeRO-3 and HuggingFace Transformers, designed to make RLHF training simple and accessible
- <img src="https://img.shields.io/github/stars/THUDM/slime?style=social" height="17" align="texttop"/> [slime](https://github.com/THUDM/slime) - an LLM post-training framework for RL Scaling
- <img src="https://img.shields.io/github/stars/kiln-ai/kiln?style=social" height="17" align="texttop"/> [Kiln](https://github.com/kiln-ai/kiln) - the easiest tool for fine-tuning LLM models, synthetic data generation, and collaborating on datasets
- <img src="https://img.shields.io/github/stars/meta-pytorch/OpenEnv?style=social" height="17" align="texttop"/> [OpenEnv](https://github.com/meta-pytorch/OpenEnv) - an interface library for RL post training with environments
- <img src="https://img.shields.io/github/stars/e-p-armstrong/augmentoolkit?style=social" height="17" align="texttop"/> [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) - train an open-source LLM on new facts
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/nvidia-nemo/rl?style=social" height="17" align="texttop"/> [RL](https://github.com/nvidia-nemo/rl) - scalable toolkit for efficient model reinforcement
- <img src="https://img.shields.io/github/stars/radixark/miles?style=social" height="17" align="texttop"/> [miles](https://github.com/radixark/miles) - an enterprise-facing reinforcement learning framework for LLM and VLM post-training, forked from and co-evolving with slime
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA-NeMo/gym?style=social" height="17" align="texttop"/> [Gym](https://github.com/NVIDIA-NeMo/gym) - evaluate and improve models and agents using environments
- <img src="https://img.shields.io/github/stars/sgl-project/SpecForge?style=social" height="17" align="texttop"/> [SpecForge](https://github.com/sgl-project/SpecForge) - train speculative decoding models effortlessly and port them smoothly to SGLang serving

[Back to Table of Contents](#table-of-contents)

### Security and Sandboxing

- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA/garak?style=social" height="17" align="texttop"/> [garak](https://github.com/NVIDIA/garak) - the LLM vulnerability scanner from NVIDIA
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA-NeMo/Guardrails?style=social" height="17" align="texttop"/> [Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) - an open-source toolkit from NVIDIA for easily adding programmable guardrails to LLM-based conversational systems
- <img src="https://img.shields.io/badge/NVIDIA-%25?logo=nvidia&labelColor=white" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/NVIDIA/OpenShell?style=social" height="17" align="texttop"/> [OpenShell](https://github.com/NVIDIA/OpenShell) - the safe, private runtime for autonomous AI agents from NVIDIA
- <img src="https://img.shields.io/github/stars/TencentCloud/CubeSandbox?style=social" height="17" align="texttop"/> [CubeSandbox](https://github.com/TencentCloud/CubeSandbox) - instant, concurrent, secure & lightweight sandbox for AI agents

[Back to Table of Contents](#table-of-contents)

### Miscellaneous

- <img src="https://img.shields.io/github/stars/upstash/context7?style=social" height="17" align="texttop"/> [context7](https://github.com/upstash/context7) - up-to-date code documentation for LLMs and AI code editors
- <img src="https://img.shields.io/github/stars/AsyncFuncAI/deepwiki-open?style=social" height="17" align="texttop"/> [deepwiki-open](https://github.com/AsyncFuncAI/deepwiki-open) - open source DeepWiki: AI-powered wiki generator for GitHub/Gitlab/Bitbucket repositories
- <img src="https://img.shields.io/github/stars/aliasrobotics/cai?style=social" height="17" align="texttop"/> [cai](https://github.com/aliasrobotics/cai) - Cybersecurity AI (CAI), the framework for AI Security
- <img src="https://img.shields.io/github/stars/murtaza-nasir/speakr?style=social" height="17" align="texttop"/> [speakr](https://github.com/murtaza-nasir/speakr) - a personal, self-hosted web application designed for transcribing audio recordings
- <img src="https://img.shields.io/github/stars/presenton/presenton?style=social" height="17" align="texttop"/> [presenton](https://github.com/presenton/presenton) - an open-source AI presentation generator and API
- <img src="https://img.shields.io/github/stars/VectorSpaceLab/OmniGen2?style=social" height="17" align="texttop"/> [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2) - exploration to advanced multimodal generation
- <img src="https://img.shields.io/github/stars/TheAhmadOsman/4o-ghibli-at-home?style=social" height="17" align="texttop"/> [4o-ghibli-at-home](https://github.com/TheAhmadOsman/4o-ghibli-at-home) - a powerful, self-hosted AI photo stylizer built for performance and privacy
- <img src="https://img.shields.io/github/stars/Roy3838/Observer?style=social" height="17" align="texttop"/> [Observer](https://github.com/Roy3838/Observer) - local open-source micro-agents that observe, log and react, all while keeping your data private and secure
- <img src="https://img.shields.io/github/stars/minitap-ai/mobile-use?style=social" height="17" align="texttop"/> [mobile-use](https://github.com/minitap-ai/mobile-use) - a powerful, open-source AI agent that controls your Android or IOS device using natural language
- <img src="https://img.shields.io/github/stars/gabber-dev/gabber?style=social" height="17" align="texttop"/> [gabber](https://github.com/gabber-dev/gabber) - build AI applications that can see, hear, and speak using your screens, microphones, and cameras as inputs
- <img src="https://img.shields.io/github/stars/sevenreasons/promptcat?style=social" height="17" align="texttop"/> [promptcat](https://github.com/sevenreasons/promptcat) - a zero-dependency prompt manager/catalog/library in a single HTML file

[Back to Table of Contents](#table-of-contents)

## Hardware

- <img src="https://img.shields.io/youtube/channel/subscribers/UCajiMK_CY9icRhLepS8_3ug?style=social" height="17" align="texttop"/> [Alex Ziskind](https://www.youtube.com/@AZisk) - tests of pcs, laptops, gpus etc. capable of running LLMs
- <img src="https://img.shields.io/youtube/channel/subscribers/UCiaQzXI5528Il6r2NNkrkJA?style=social" height="17" align="texttop"/> [Digital Spaceport](https://www.youtube.com/@DigitalSpaceport) - reviews of various builds designed for LLM inference
- <img src="https://img.shields.io/youtube/channel/subscribers/UCP0QFok6EimQYTMj5qOLNow?style=social" height="17" align="texttop"/> [Donato Capitella](https://www.youtube.com/@donatocapitella) - practical and insightful tutorials on running LLMs locally
- <img src="https://img.shields.io/youtube/channel/subscribers/UCQs0lwV6E4p7LQaGJ6fgy5Q?style=social" height="17" align="texttop"/> [JetsonHacks](https://www.youtube.com/@JetsonHacks) - information about developing on NVIDIA Jetson Development Kits
- <img src="https://img.shields.io/youtube/channel/subscribers/UC8h2Sf-yyo1WXeEUr-OHgyg?style=social" height="17" align="texttop"/> [Miyconst](https://www.youtube.com/@Miyconst) - tests of various types of hardware capable of running LLMs
- [Kolosal - LLM Memory calculator](https://www.kolosal.ai/memory-calculator) - estimate the RAM requirements of any GGUF model instantly
- [LLM Inference VRAM & GPU Requirement Calculator](https://app.linpp2009.com/en/llm-gpu-memory-calculator) - calculate how many GPUs you need to deploy LLMs
- <img src="https://img.shields.io/github/stars/vosen/ZLUDA?style=social" height="17" align="texttop"/> [ZLUDA](https://github.com/vosen/ZLUDA) - CUDA on non-NVIDIA GPUs
- [Strix Halo AI Toolboxes](https://strix-halo-toolboxes.com/) - toolboxes for GenAI on AMD Ryzen AI MAX+: containerized environments for LLMs, Image Generation, and Fine-tuning
- [Strix Halo Wiki](https://strixhalo.wiki/) - a website to gather important information and practical guides for systems powered by AMD Ryzen AI MAX and MAX+ processors
- <img src="https://img.shields.io/github/stars/paudley/ai-notes?style=social" height="17" align="texttop"/> [ai-notes](https://github.com/paudley/ai-notes) - random AI notes for working with local models or playing around with random machine learning bits

[Back to Table of Contents](#table-of-contents)

## Tutorials

### Models

- <img src="https://img.shields.io/youtube/views/l8pRSuU81PU?style=social" height="17" align="texttop"/> [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- <img src="https://img.shields.io/github/stars/karpathy/nanochat?style=social" height="17" align="texttop"/> [nanochat](https://github.com/karpathy/nanochat) - a full-stack implementation of an LLM like ChatGPT in a single, clean, minimal, hackable, dependency-lite codebase, designed to run on a single 8XH100 node via scripts like speedrun.sh, that run the entire pipeline start to end
- <img src="https://img.shields.io/youtube/views/jrJKRYAdh7I?style=social" height="17" align="texttop"/> [Knowledge Distillation: How LLMs train each other](https://www.youtube.com/watch?v=jrJKRYAdh7I)
- <img src="https://img.shields.io/github/stars/iuliaturc/gguf-docs?style=social" height="17" align="texttop"/> [gguf-docs](https://github.com/iuliaturc/gguf-docs) - Docs for GGUF quantization (unofficial)
- <img src="https://img.shields.io/badge/arXiv-%2525?logo=arxiv&logoColor=white&labelColor=%23B31B1B&color=black" height="17" align="texttop"/> [Embarrassingly Simple Self-Distillation
Improves Code Generation](https://arxiv.org/pdf/2604.01193)

[Back to Table of Contents](#table-of-contents)

### Prompt Engineering

- <img src="https://img.shields.io/github/stars/dair-ai/Prompt-Engineering-Guide?style=social" height="17" align="texttop"/> [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) - guides, papers, lecture, notebooks and resources for prompt engineering
- <img src="https://img.shields.io/github/stars/NirDiamant/Prompt_Engineering?style=social" height="17" align="texttop"/> [Prompt Engineering by NirDiamant](https://github.com/NirDiamant/Prompt_Engineering) - a comprehensive collection of tutorials and implementations for Prompt Engineering techniques, ranging from fundamental concepts to advanced strategies
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> [Prompting guide 101](https://services.google.com/fh/files/misc/gemini-for-google-workspace-prompting-guide-101.pdf) - a quick-start handbook for effective prompts by Google
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> [Prompt Engineering by Google](https://drive.google.com/file/d/1AbaBYbEa_EbPelsT40-vj64L-2IwUJHy/view) - prompt engineering by Google
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17" align="texttop"/> [Prompt Engineering by Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) - prompt engineering by Anthropic
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17" align="texttop"/> [Prompt Engineering Interactive Tutorial](https://github.com/anthropics/courses/blob/master/prompt_engineering_interactive_tutorial/README.md) - Prompt Engineering Interactive Tutorial by Anthropic
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17" align="texttop"/> [Real world prompting](https://github.com/anthropics/courses/blob/master/real_world_prompting/README.md) - real world prompting tutorial by Anthropic
- <img src="https://img.shields.io/badge/Anthropic-%23191919?logo=anthropic" height="17" align="texttop"/> [Prompt evaluations](https://github.com/anthropics/courses/blob/master/prompt_evaluations/README.md) - prompt evaluations course by Anthropic
- <img src="https://img.shields.io/github/stars/x1xhlol/system-prompts-and-models-of-ai-tools?style=social" height="17" align="texttop"/> [system-prompts-and-models-of-ai-tools](https://github.com/x1xhlol/system-prompts-and-models-of-ai-tools) - a collection of system prompts extracted from AI tools
- <img src="https://img.shields.io/github/stars/asgeirtj/system_prompts_leaks?style=social" height="17" align="texttop"/> [system_prompts_leaks](https://github.com/asgeirtj/system_prompts_leaks) - a collection of extracted System Prompts from popular chatbots like ChatGPT, Claude & Gemini
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> <img src="https://img.shields.io/github/stars/openai/codex?style=social" height="17" align="texttop"/> [Prompt from Codex](https://github.com/openai/codex/blob/main/codex-rs/core/prompt.md) - Prompt used to steer behavior of OpenAI's Codex

[Back to Table of Contents](#table-of-contents)

### Context Engineering

- <img src="https://img.shields.io/github/stars/davidkimai/Context-Engineering?style=social" height="17" align="texttop"/> [Context-Engineering](https://github.com/davidkimai/Context-Engineering) - a frontier, first-principles handbook inspired by Karpathy and 3Blue1Brown for moving beyond prompt engineering to the wider discipline of context design, orchestration, and optimization
- <img src="https://img.shields.io/github/stars/Meirtz/Awesome-Context-Engineering?style=social" height="17" align="texttop"/> [Awesome-Context-Engineering](https://github.com/Meirtz/Awesome-Context-Engineering) - a comprehensive survey on Context Engineering: from prompt engineering to production-grade AI systems

[Back to Table of Contents](#table-of-contents)

### Inference

- <img src="https://img.shields.io/github/stars/vllm-project/production-stack?style=social" height="17" align="texttop"/> [vLLM Production Stack](https://github.com/vllm-project/production-stack) - vLLM’s reference system for K8S-native cluster-wide deployment with community-driven performance optimization

[Back to Table of Contents](#table-of-contents)

### Agents

- <img src="https://img.shields.io/github/stars/obra/superpowers?style=social" height="17" align="texttop"/> [superpowers](https://github.com/obra/superpowers) - an agentic skills framework & software development methodology that works
- <img src="https://img.shields.io/github/stars/NirDiamant/GenAI_Agents?style=social" height="17" align="texttop"/> [GenAI Agents](https://github.com/NirDiamant/GenAI_Agents) - tutorials and implementations for various Generative AI Agent techniques
- <img src="https://img.shields.io/github/stars/ashishpatel26/500-AI-Agents-Projects?style=social" height="17" align="texttop"/> [500+ AI Agent Projects](https://github.com/ashishpatel26/500-AI-Agents-Projects) - a curated collection of AI agent use cases across various industries
- <img src="https://img.shields.io/github/stars/humanlayer/12-factor-agents?style=social" height="17" align="texttop"/> [12-Factor Agents](https://github.com/humanlayer/12-factor-agents) - principles for building reliable LLM applications
- <img src="https://img.shields.io/github/stars/NirDiamant/agents-towards-production?style=social" height="17" align="texttop"/> [Agents towards production](https://github.com/NirDiamant/agents-towards-production) - end-to-end, code-first tutorials covering every layer of production-grade GenAI agents, guiding you from spark to scale with proven patterns and reusable blueprints for real-world launches
- <img src="https://img.shields.io/github/stars/agentsmd/agents.md?style=social" height="17" align="texttop"/> [agents.md](https://github.com/agentsmd/agents.md) - a simple, open format for guiding coding agents
- <img src="https://img.shields.io/github/stars/agentskills/agentskills?style=social" height="17" align="texttop"/> [Agent Skills](https://github.com/agentskills/agentskills) - a simple, open format for giving agents new capabilities and expertise
- <img src="https://img.shields.io/github/stars/huggingface/skills?style=social" height="17" align="texttop"/> [skills](https://github.com/huggingface/skills) - Hugging Face Skills are definitions for AI/ML tasks like dataset creation, model training and evaluation
- <img src="https://img.shields.io/github/stars/oxbshw/LLM-Agents-Ecosystem-Handbook?style=social" height="17" align="texttop"/> [LLM Agents & Ecosystem Handbook](https://github.com/oxbshw/LLM-Agents-Ecosystem-Handbook) - one-stop handbook for building, deploying, and understanding LLM agents with 60+ skeletons, tutorials, ecosystem guides, and evaluation tools
- <img src="https://img.shields.io/badge/Google-%234285F4?logo=google&logoColor=red" height="17" align="texttop"/> [601 real-world gen AI use cases](https://cloud.google.com/transform/101-real-world-generative-ai-use-cases-from-industry-leaders) - 601 real-world gen AI use cases from the world's leading organizations by Google
- <img src="https://img.shields.io/badge/OpenAI-%23412991?logo=openai" height="17" align="texttop"/> [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) - a practical guide to building agents by OpenAI

[Back to Table of Contents](#table-of-contents)

### Retrieval-Augmented Generation

- <img src="https://img.shields.io/github/stars/pathwaycom/llm-app?style=social" height="17" align="texttop"/> [Pathway AI Pipelines](https://github.com/pathwaycom/llm-app) - ready-to-run cloud templates for RAG, AI pipelines, and enterprise search with live data
- <img src="https://img.shields.io/github/stars/NirDiamant/RAG_Techniques?style=social" height="17" align="texttop"/> [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) - various advanced techniques for Retrieval-Augmented Generation (RAG) systems
- <img src="https://img.shields.io/github/stars/NirDiamant/Controllable-RAG-Agent?style=social" height="17" align="texttop"/> [Controllable RAG Agent](https://github.com/NirDiamant/Controllable-RAG-Agent) - an advanced Retrieval-Augmented Generation (RAG) solution for complex question answering that uses sophisticated graph based algorithm to handle the tasks
- <img src="https://img.shields.io/github/stars/lokeswaran-aj/langchain-rag-cookbook?style=social" height="17" align="texttop"/> [LangChain RAG Cookbook](https://github.com/lokeswaran-aj/langchain-rag-cookbook) - a collection of modular RAG techniques, implemented in LangChain + Python

[Back to Table of Contents](#table-of-contents)

### Miscellaneous

- <img src="https://img.shields.io/github/stars/jamesob/local-llm?style=social" height="17" align="texttop"/> [local-llm](https://github.com/jamesob/local-llm) - everything jamesob knows about running LLMs locally
- [Self-hosted AI coding that just works](https://www.reddit.com/r/LocalLLaMA/comments/1lt4y1z/selfhosted_ai_coding_that_just_works/)

[Back to Table of Contents](#table-of-contents)

## Communities

- <img src="https://img.shields.io/reddit/subreddit-subscribers/LocalLLaMA?style=social" height="17" align="texttop"/> [LocalLLaMA](https://www.reddit.com/r/LocalLLaMA)
- <img src="https://img.shields.io/reddit/subreddit-subscribers/LLMDevs?style=social" height="17" align="texttop"/> [LLMDevs](https://www.reddit.com/r/LLMDevs)
- <img src="https://img.shields.io/reddit/subreddit-subscribers/LocalLLM?style=social" height="17" align="texttop"/> [LocalLLM](https://www.reddit.com/r/LocalLLM)
- <img src="https://img.shields.io/reddit/subreddit-subscribers/LocalAIServers?style=social" height="17" align="texttop"/> [LocalAIServers](https://www.reddit.com/r/LocalAIServers/)
- <img src="https://img.shields.io/badge/Telegram-%25?logo=telegram&labelColor=white&color=blue" height="17" align="texttop"/> [GenAI monitor](https://t.me/genaimon) - monitoring updates & fresh releases related to LLMs, diffusion models and Generative AI

[Back to Table of Contents](#table-of-contents)

# Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to get started.

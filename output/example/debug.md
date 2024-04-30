



 April 26, 2024
 
[AINews] Apple's OpenELM beats OLMo with 50% of its dataset, using DeLighT
==========================================================================



> This is AI News! an MVP of a service that goes thru all AI discords/Twitters/reddits and summarizes what people are talking about, so that you can keep up without the fatigue. Signing up
>  [here](https://buttondown.email/ainews/) 
>  opts you in to the real thing when we launch it 🔜




---



> AI News for 4/24/2024-4/26/2024. We checked 7 subreddits and
>  [**373** 
>  Twitters](https://twitter.com/i/lists/1585430245762441216?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
>  and
>  **27** 
>  Discords (
>  **395** 
>  channels, and
>  **5502** 
>  messages) for you. Estimated reading time saved (at 200wpm):
>  **599 minutes** 
>  .



[Apple's AI emergence](https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/) 
 continues apace ahead of WWDC. We've covered
 [OLMo](https://buttondown.email/ainews/archive/ainews-ai2-releases-olmo-the-4th-open-everything/) 
 before, and it looks like OpenELM is Apple's first
 [actually open LLM](https://arxiv.org/abs/2404.14619?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 (
 [weights](https://huggingface.co/apple/OpenELM?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ,
 [code](https://github.com/apple/corenet?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ) release sharing some novel research in the efficient architecture direction.
 



![image.png](https://assets.buttondown.email/images/3bd4b772-df2f-46b7-8318-2cc230b7eb46.png?w=960&fit=max)




 It's not
 *totally* 
 open, but it's pretty open. As
 [Sebastian Raschka put it](https://twitter.com/rasbt/status/1783480053847736713/photo/1?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 :
 



> Let's start with the most interesting tidbits:
>  
> 
> 
> * OpenELM comes in 4 relatively small and convenient sizes: 270M, 450M, 1.1B, and 3B
> * OpenELM performs slightly better than OLMo even though it's trained on 2x fewer tokens
> * The main architecture tweak is a layer-wise scaling strategy



 But:
 



> "Sharing details is not the same as explaining them, which is what research papers were aimed to do when I was a graduate student. For instance, they sampled a relatively small subset of 1.8T tokens from various publicly available datasets (RefinedWeb, RedPajama, The PILE, and Dolma). This subset was 2x smaller than Dolma, which was used for training OLMo. What was the rationale for this subsampling, and what were the criteria?"



![image.png](https://assets.buttondown.email/images/5a0bcc71-6f46-41a3-a34b-6efff203c64d.png?w=960&fit=max)




 The layer-wise scaling comes from
 [DeLight](https://arxiv.org/abs/2008.00623?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , a 2021 paper deepening the standard attention mechanism 2.5-5x in number of layers but matching 2-3x larger models by parameter count. These seem paradoxical but the authors described the main trick of varying the depth between the input and the output, rather than uniform:
 



![image.png](https://assets.buttondown.email/images/64a3ecf6-fbca-4816-9233-f4100454aca8.png?w=960&fit=max)




![image.png](https://assets.buttondown.email/images/a70b5ba1-00bb-482d-a4a4-f1027eec0266.png?w=960&fit=max)





---



**Table of Contents** 




* [AI Reddit Recap](#ai-reddit-recap)
* [AI Twitter Recap](#ai-twitter-recap)
* [AI Discord Recap](#ai-discord-recap)
* [PART 1: High level Discord summaries](#part-1-high-level-discord-summaries) 
	+ [Unsloth AI (Daniel Han) Discord](#unsloth-ai-daniel-han-discord)
	+ [Perplexity AI Discord](#perplexity-ai-discord)
	+ [CUDA MODE Discord](#cuda-mode-discord)
	+ [LM Studio Discord](#lm-studio-discord)
	+ [Nous Research AI Discord](#nous-research-ai-discord)
	+ [OpenAI Discord](#openai-discord)
	+ [Stability.ai (Stable Diffusion) Discord](#stabilityai-stable-diffusion-discord)
	+ [HuggingFace Discord](#huggingface-discord)
	+ [Eleuther Discord](#eleuther-discord)
	+ [OpenRouter (Alex Atallah) Discord](#openrouter-alex-atallah-discord)
	+ [LlamaIndex Discord](#llamaindex-discord)
	+ [LAION Discord](#laion-discord)
	+ [OpenInterpreter Discord](#openinterpreter-discord)
	+ [OpenAccess AI Collective (axolotl) Discord](#openaccess-ai-collective-axolotl-discord)
	+ [Cohere Discord](#cohere-discord)
	+ [tinygrad (George Hotz) Discord](#tinygrad-george-hotz-discord)
	+ [Modular (Mojo 🔥) Discord](#modular-mojo-discord)
	+ [LangChain AI Discord](#langchain-ai-discord)
	+ [Latent Space Discord](#latent-space-discord)
	+ [Mozilla AI Discord](#mozilla-ai-discord)
	+ [DiscoResearch Discord](#discoresearch-discord)
	+ [Interconnects (Nathan Lambert) Discord](#interconnects-nathan-lambert-discord)
	+ [Skunkworks AI Discord](#skunkworks-ai-discord)
	+ [Datasette - LLM (@SimonW) Discord](#datasette-llm-simonw-discord)
	+ [Alignment Lab AI Discord](#alignment-lab-ai-discord)
* [PART 2: Detailed by-Channel summaries and links](#part-2-detailed-by-channel-summaries-and-links)





---


AI Reddit Recap
===============



> Across r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity. Comment crawling works now but has lots to improve!



**LLaMA Developments** 



* **LLaMA 3 increases context to 160K+ tokens** 
 : In /r/LocalLLaMA, LLaMA 3 increases context length to
 [**over 160K tokens while maintaining perfect recall**](https://www.reddit.com/r/LocalLLaMA/comments/1ccqmjz/llama_3_now_with_160k_context/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 . Commenters note this is impressive but will require significant consumer hardware to run locally at good speeds. Meta's Llama 3 has been downloaded over 1.2M times, with over 600 derivative models on Hugging Face.
* **First LLama-3 8B-Instruct model with 262K context released** 
 : In /r/LocalLLaMA, the first LLama-3 8B-Instruct model with
 [**over 262K context length is released on Hugging Face**](https://www.reddit.com/r/LocalLLaMA/comments/1cd4yim/llama38binstruct_with_a_262k_context_length/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 , enabling advanced reasoning beyond simple prompts.
* **Llama 3 70B outperforms 8B model** 
 : In /r/LocalLLaMA, comparisons show the
 [**quantized Llama 3 70B IQ2\_XS outperforms the uncompressed Llama 3 8B f16 model**](https://www.reddit.com/r/LocalLLaMA/comments/1cda0fv/llama_3_8b_f16_vs_llama_3_70b_q2/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 . The 70B IQ3\_XS version is found to be best for 32GB VRAM users.
* **New paper compares AI alignment approaches** 
 : In /r/LocalLLaMA, a new paper compares DPO to other alignment approaches, finding
 [**KTO performs best on most benchmarks and alignment methods are sensitive to training data volume**](https://www.reddit.com/r/LocalLLaMA/comments/1ccz84a/insights_into_alignment_dpo_and_its_variants/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 .



**AI Ethics & Regulation** 



* **Eric Schmidt warns about risks of open-source AI** 
 : In /r/singularity, former Google CEO Eric Schmidt cautions that
 [**open-source AI models give risky capabilities to bad actors and China**](https://www.reddit.com/r/singularity/comments/1ccyqkr/former_google_ceo_eric_schmidt_warns_that_open/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 . Many see this as an attempt by large tech companies to stifle competition, noting China likely has the capability to develop powerful models without relying on open-source.
* **U.S. proposal aims to end anonymous cloud usage** 
 : In /r/singularity, a
 [**U.S. proposal seeks to implement "Know Your Customer" requirements to end anonymous cloud usage**](https://www.reddit.com/r/singularity/comments/1ccr2ub/us_know_your_customer_proposal_will_put_an_end_to/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 .
* **Baltimore coach allegedly used AI for defamation** 
 : In /r/OpenAI, a Baltimore coach allegedly
 [**used AI voice cloning to attempt to get a high school principal fired by generating fake racist audio**](https://www.reddit.com/r/OpenAI/comments/1cd5h9c/baltimore_high_school_athletic_director_used_ai/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 .



**Hardware Developments** 



* **TSMC unveils 1.6nm process node** 
 : In /r/singularity, TSMC announces a
 [**1.6nm process node with backside power delivery**](https://www.reddit.com/r/singularity/comments/1ccr4hy/tsmc_unveils_16nm_process_technology_with/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 , enabling continued exponential hardware progress over the next few years.
* **Ultra-thin solar cells enable self-charging drones** 
 : In /r/singularity, German researchers develop
 [**ultra-thin, flexible solar cells that allow small drones to self-charge during operation**](https://www.reddit.com/r/singularity/comments/1ccr6aq/german_researchers_have_developed_a_solar_cell/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 .
* **Micron secures $6.1B in CHIPS Act funding** 
 : In /r/singularity, Micron secures
 [**$6.1 billion in CHIPS Act funding to build semiconductor manufacturing facilities in New York and Idaho**](https://www.reddit.com/r/singularity/comments/1cd0s5k/micron_set_to_receive_61b_in_chips_act_funding_to/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 .



**Memes & Humor** 



* **AI assistant confidently asserts flat Earth** 
 : In /r/singularity, a humorous image depicts an
 [**AI assistant confidently asserting that the Earth is flat**](https://www.reddit.com/r/singularity/comments/1ccqhzv/chat_is_this_real/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)
 , sparking jokes about needing AI capable of believing absurdities or that humanity has its best interests at heart.




---


AI Twitter Recap
================



> all recaps done by Claude 3 Opus, best of 4 runs. We are working on clustering and flow engineering with Haiku.



 Here is a summary of the key topics and insights from the provided tweets:
 



**Meta Llama 3 Release and Impact** 



* **Rapid Adoption** 
 : In the week since release, Llama 3 models have been downloaded over 1.2M times with 600+ derivative models on Hugging Face, showing exciting early impact. (
 [@AIatMeta](https://twitter.com/AIatMeta/status/1783602908845748685?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Training Optimizations** 
 : Meta is moving fast on optimizations, with Llama 3 70B training 18% faster and Llama 3 8B training 20% faster. (
 [@svpino](https://twitter.com/svpino/status/1783888989025431933?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Context Extension** 
 : The community extended Llama 3 8B's context from 8k to nearly 100k tokens by combining PoSE, continued pre-training, and RoPE scaling. (
 [@winglian](https://twitter.com/winglian/status/1783842736833016289?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Inference Acceleration** 
 : Colossal-Inference now supports Llama 3 inference acceleration, enhancing efficiency by ~20% for 8B and 70B models. (
 [@omarsar0](https://twitter.com/omarsar0/status/1783895931043111088?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Benchmark Performance** 
 : Llama 3 70B is tied for 1st place for English queries on the LMSYS leaderboard. (
 [@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1783570318230978783?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )



**Phi-3 Model Release and Reception** 



* **Overfitting Benchmarks** 
 : Some argue Phi-3 overfits public benchmarks but underperforms in practical usage compared to models like Llama-3 8B. (
 [@svpino](https://twitter.com/svpino/status/1783556635543339310?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ,
 [@abacaj](https://twitter.com/abacaj/status/1783898711623352686?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Unexpected Behavior** 
 : As a fundamentally different model, Phi-3 can exhibit surprising results, both good and bad. (
 [@srush\_nlp](https://twitter.com/SebastienBubeck/status/1783885843943616524?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )



**Extending LLM Context Windows** 



* **PoSE Technique** 
 : The Positional Skip-wisE (PoSE) method simulates long inputs during training to increase context length, powering Llama 3's extension to 128k tokens. (
 [@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1783574428858696161?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Axolotl and Gradient AI** 
 : Tools like Axolotl and approaches from Gradient AI are enabling context extension for Llama and other models to 160k+ tokens. (
 [@winglian](https://twitter.com/winglian/status/1783469196011016696?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ,
 [@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1783736130321408011?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )



**Cohere Toolkit Release** 



* **Enterprise Focus** 
 : Cohere released a toolkit to accelerate LLM deployment in enterprises, targeting secure RAG with private data and local code interpreters. (
 [@aidangomez](https://twitter.com/aidangomez/status/1783533461401227563?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Flexible Deployment** 
 : The toolkit's components can be deployed to any cloud and reused to build applications. (
 [@aidangomez](https://twitter.com/aidangomez/status/1783533465960378561?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ,
 [@aidangomez](https://twitter.com/aidangomez/status/1783533471777935433?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )



**OpenAI Employee Suspension and GPT-5 Speculation** 



* **Sentience Claims** 
 : An OpenAI employee who claimed GPT-5 is sentient has been suspended from Twitter. (
 [@bindureddy](https://twitter.com/bindureddy/status/1783847600824995850?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Hype Generation** 
 : OpenAI is seen as a hype-creation engine around AGI and AI sentience claims, even as competitors match GPT-4 at lower costs. (
 [@bindureddy](https://twitter.com/bindureddy/status/1783852748636905716?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* **Agent Capabilities** 
 : Some believe GPT-5 will be an "agent GPT" based on the performance boost from agent infrastructure on top of language models. (
 [@OfirPress](https://twitter.com/OfirPress/status/1783870394581074110?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )



**Other Noteworthy Topics** 



* Concerns about the AI summit board's lack of diverse representation to address power concentration risks. (
 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1783882237764633052?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* OpenAI and Moderna's partnership as a positive sign of traditional businesses adopting generative AI. (
 [@gdb](https://twitter.com/gdb/status/1783529202974687527?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ,
 [@rohanpaul\_ai](https://twitter.com/rohanpaul_ai/status/1783533728846827681?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )
* Apple's open-sourced on-device language models showing poor performance but providing useful architecture and training details. (
 [@bindureddy](https://twitter.com/bindureddy/status/1783635037365436462?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ,
 [@rasbt](https://twitter.com/rasbt/status/1783480053847736713?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 )




---


AI Discord Recap
================



> A summary of Summaries of Summaries


1. **Extending LLM Context Lengths** 



	* **Llama 3 Performance and Context Length Innovations** 
	 : Discussions centered around
	 **Llama 3's capabilities** 
	 , with some expressing mixed opinions on its code recall and configuration compared to
	 **GPT-4** 
	 . However, innovations in extending Llama 3's
	 **context length to 96k tokens for the 8B model** 
	 using techniques like
	 **PoSE (Positional Skip-wisE)** 
	 and continued pre-training with 300M tokens generated excitement, as detailed in this
	 [tweet thread](https://x.com/winglian/status/1783456379199484367?s=46&t=stOPrwZiN_fxSK0RuC8Flg&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 .
	* The
	 [EasyContext project](https://github.com/jzhang38/EasyContext?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 aims to extrapolate LLM context lengths to
	 **1 million tokens** 
	 with minimal hardware requirements.
2. **Optimizing LLM Training and Deployment** 



	* [Nvidia's Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its#introduction) 
	 is utilized for
	 **kernel profiling** 
	 to optimize CUDA code for LLM training.
	* **Finetuning LLMs for Domain-Specific Gains** 
	 : Interest grew in
	 **finetuning large language models** 
	 for domain-specific improvements, with examples like
	 **[Meditron](https://arxiv.org/abs/2311.16079?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)**
	 for medical applications. Discussions also covered
	 **data synthesis** 
	 strategies using tools like
	 **[Argilla's Distilabel](https://github.com/argilla-io/distilabel?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)**
	 , and the challenges of multi-document, long-context finetuning. Cost-performance tradeoffs were debated, such as spending
	 [$2,368 for 4 epochs vs $41,440 for 50 epochs](https://discord.com/channels/1053877538025386074/1154120232051408927/1232958591955112028?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 with potentially minor gains.
	* PyTorch introduces
	 [Torchtitan](https://github.com/pytorch/torchtitan?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 , a library dedicated to aiding LLM training from scratch.
	* The
	 [Mixture of Depths paper](https://paper-club.ivanleo.com/papers/mixture-of-depths?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 proposes accelerating transformer training using a modified MoE routing mechanism.
	* **CUDA Optimization Deep Dives** 
	 : CUDA developers dug into kernel profiling with tools like
	 **[NVIDIA Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its#introduction)**
	 , discussed
	 **memory coalescing** 
	 and
	 **burst sizes** 
	 around 128 bytes, and debated the efficiency of
	 **low-bit quantization** 
	 methods. Conversations also covered
	 **flash attention compatibility** 
	 issues with PyTorch 2.3.0, and the implications of PyTorch AO supporting
	 **custom CUDA extensions** 
	 for performance tuning.
3. **Open-Source LLM Ecosystem Expansion** 



	* **Apple's Surprise Entry into Open-Source Models** 
	 :
	 **Apple's release of
	 [OpenELM](https://huggingface.co/apple/OpenELM?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)**
	 , a family of efficient open-source language models ranging from 270M to 3B parameters, caught the AI community by surprise. The move marked a shift from Apple's traditionally proprietary approach, with the 270M model quickly gaining attention on Hugging Face.
	* [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 is introduced, leveraging Mistral for pretrained medical LLMs.
	* Mozilla's
	 [llamafile project](https://hacks.mozilla.org/2024/04/llamafiles-progress-four-months-in/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 enables distributing and running LLMs locally with high performance.
	* Dify emerges as an
	 [open-source LLM app development platform](https://github.com/langgenius/dify?tab=readme-ov-file&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 combining AI workflows and model management.
4. **Evaluating and Benchmarking LLMs** 



	* On the
	 [Judgemark benchmark](https://eqbench.com/judgemark.html?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 ,
	 **Llama-3-70b** 
	 shows promise for fine-tuning
	 **disco-judge** 
	 applications.
	* Discussions around the effectiveness of
	 **validation loss** 
	 as a performance indicator for LLMs.
	* The
	 [Low-Cost Language Models survey](https://arxiv.org/abs/2404.11160?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
	 evaluates CPU-friendly LLMs on Python code generation tasks.
	* Debates on the transparency of
	 **Nightshade's** 
	 autoencoder capabilities and the need for publishing findings openly.




---


PART 1: High level Discord summaries
====================================


[Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Fine-Tuning Tribulations and Triumphs** 
 : Engineers shared their challenges with fine-tuning models, involving 4-bit quantization, VRAM requirements exceeding 48GB, and token index sequence length errors. Solutions included batch size revision, enabling
 [pack](https://github.com/unslothai/unsloth?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its#-finetune-for-free) 
 functionality, and considering alternative testing environments like
 **Aphrodite Engine** 
 or
 **llama.cpp** 
 .


* **Gossip or Gone?** 
 : Speculation is rampant about the dissolution of the
 **WizardLM** 
 team after
 **Qingfeng Sun's staff page** 
 redirect. Contrasting sentiments were shared, from salvaging
 **WizardLM datasets** 
 to
 **showcase** 
 sessions where
 **Meta's LlaMA-3 models** 
 (including an 8B and 70B version) were cited as top performers in their classes.


* **From Cold Storage to Hot Topics** 
 : A member proudly announced an
 [open-source release](https://github.com/oKatanaaa/kolibrify?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 of
 **Kolibrify** 
 , a curriculum training tool for instruction-following LLMs. On a technical note, the community discussed
 **Triton** 
 dependencies, errors with "Quantization failed," and
 **gguf model** 
 testing strategies, reaching a consensus on best practices for
 **fine-tuning** 
 and deployment options.


* **Pragmatic Pruning Progress** 
 : Insights were shared about a project on iterative context length increase for models using a
 **[triton laser merge trainer](https://github.com/l4b4r4b4b4/trl/tree/evol_laser_merge_trainer?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)**
 that operates during evaluation. This method, signaled as innovative due to no reinitialization requirements, could provide a pathway for enhanced model usability without system overhaul.


* **Unsloth's Milestones and Resources** 
 : Unsloth AI marked a significant milestone with 500k monthly downloads of their fine-tuning framework on Hugging Face and promoted the sharing of
 **exact match** 
 GGUF models despite potential redundancy. Emphasis was also on directing users to
 **[Colab notebooks](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)**
 for effective fine-tuning strategies.




---


[Perplexity AI](https://discord.com/channels/1047197230748151888?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Siri Gets a Brainy Buddy** 
 : Perplexity AI Discord chatbot introduces an exclusive auditory feature for
 **iOS users** 
 that reads answers to any posed question.


* **Opus Limit Outcry** 
 : Frustration arises within the community concerning the new 50-query daily limit on Claude 3 Opus interactions, while still,
 **Perplexity chatbot supports Opus** 
 despite these caps.


* **API Adoption Anxieties** 
 : AI Engineers are discussing integration issues with the Perplexity API, such as outdated responses and a lack of GPT-4 support; a user also sought advice on
 **optimal hyperparameters** 
 for the
 `llama-3-70b-instruct` 
 model.


* **A Game of Models** 
 : The community is buzzing with anticipation around Google's Gemini model, and its potential impact on the AI landscape, while noting GPT-5 will have to bring exceptional innovations to keep up with the competition.


* **Crystal Ball for Net Neutrality** 
 : A linked article prompts discussions on the FCC's reestablishment of Net Neutrality, with implications for the
 **AI Boom's** 
 future being pondered by community members.




---


[CUDA MODE](https://discord.com/channels/1189498204333543425?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------



**CUDA Collective Comes Together** 
 : Members focused on honing their skills with
 **CUDA** 
 through optimizing various kernels and algorithms, including matrix multiplication and flash attention. Threads spanned from leveraging the
 [NVIDIA Nsight Compute CLI User Guide](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its#introduction) 
 for kernel profiling to debate on the efficiency of low-bit quantization methods.
 



**PyTorch Tangles with Compatibility and Extensions** 
 : A snag was hit with
 **flash-attn compatibility** 
 in
 **PyTorch 2.3.0** 
 , resulting in an
 `undefined symbol` 
 error, which participants hoped to see rectified promptly. PyTorch AO ignited enthusiasm by
 [supporting custom CUDA extensions](https://github.com/pytorch/ao/pull/135?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , facilitating performance tuning using
 `torch.compile` 
 .
 



**Greener Code with C++** 
 : An announcement about a bonus talk from the
 **NVIDIA C++ team** 
 on converting
 `llm.c` 
 to
 `llm.cpp` 
 teased opportunities for clearer, faster code.
 



**The Matrix of Memory and Models** 
 : Discussions delved deep into finer points of CUDA best practices, contemplating
 **burst sizes** 
 for memory coalescing around
 **128 bytes** 
 as explored in
 **Chapter 6, section 3.d** 
 of the CUDA guide, and toying with the concept of reducing overhead in packed operations.
 



**Recording Rendezvous** 
 : Volunteers stepped up for screen recording with detailed, actionable advice and
 [Existential Audio - BlackHole](https://existential.audio/blackhole/download/?code=681349920&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 for lossless sound capture, highlighting the careful nuances needed for a refined technical setup.
 




---


[LM Studio](https://discord.com/channels/1110598183144399058?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **GPU Offloads to AMD OpenCL** 
 : A technical hiccup with
 **GPU Offloading** 
 was resolved by switching the GPU type to
 **AMD Open CL** 
 , demonstrating a simple fix can sidestep performance issues.
* **Mixed News on Updates and Performance** 
 : Upgrade issues cropped up in LM Studio with
 **version 0.2.21** 
 , causing previous setups running
 **phi-3 mini models** 
 to malfunction, while other users are experimenting with using
 **Version 2.20** 
 and facing GPU usage spikes without successful model loading. Users are actively troubleshooting, including submitting requests for screenshots for better diagnostics.
* **LM Studio Turns Chat into Document Dynamo** 
 : Enthusiastic discussions around improving
 **LM Studio's chat feature** 
 have led to embedding document retrieval using
 **Retriever-Augmented Generation (RAG)** 
 and tweaking GPU settings for better resource utilization.
* **Tackling AI with Graphical Might** 
 : The community is sharing insights into optimal hardware setups and potential performance boosts anticipated from Nvidia Tesla equipment when using AI models, indicating a strong interest in the best equipment for AI model hosting.
* **AMD's ROCm Under the Microscope** 
 : The use of
 **AMD's ROCm tech preview** 
 has shown promise with certain setups, achieving a notable 30t/s on an eGPU system, although compatibility snags underscore the importance of checking GPU support against the ROCm documentation.




---


[Nous Research AI](https://discord.com/channels/1053877538025386074?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Pushing The Envelope on Model Context Limits** 
 : Llama 3 models are breaking context barriers, with one variant reaching a
 **96k context for the 8B model** 
 using PoSE and continued pre-training with 300M tokens. The efficacy of Positional Skip-wisE (PoSE) and
 **RoPE scaling** 
 were key topics, with a
 [paper on PoSE's context window extension](https://openreview.net/forum?id=3Z1gxuAQrA&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and discussions on fine-tuning RoPE base during fine-tuning for lengthier contexts mentioned.
 



**LLM Performance and Cost Discussions Engage Community** 
 : Engineers expressed skepticism about validation loss as a performance indicator and shared a cost comparison of training epochs, highlighting a case where four epochs cost $2,368 versus $41,440 for fifty epochs with minor performance gains. Another engineer is considering combining several 8B models into a mixture of experts based on
 **Gemma MoE** 
 and speculated on potential enhancements using
 **DPO/ORPO techniques** 
 .
 



**The Saga of Repository Archival** 
 : Concerns were voiced about the sudden disappearance of Microsoft’s WizardLM repo, sparking a debate on the importance of archiving, especially in light of Microsoft's investment in OpenAI. Participants underscored the need for backups, drawing from instances such as the recent reveal of
 **WizardLM-2** 
 , accessible on
 [Hugging Face](https://huggingface.co/collections/microsoft/wizardlm-2-661d403f71e6c8257dbd598a?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and
 [GitHub](https://github.com/victorsungo/WizardLM/tree/main/WizardLM-2?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Synthetic Data Generation: A One-Stop Shop** 
 :
 *Argilla’s Distilabel* 
 was recommended for creating
 **diverse synthetic data** 
 , with practical examples and repositories such as the
 [distilabel-workbench](https://github.com/argilla-io/distilabel-workbench?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 illustrating its applications. The conversation spanned single document data synthesis, multi-document challenges, and strategies for extended contexts in language models.
 



**Simulated World Engagements Rouse Curiosity** 
 : Websim’s capabilities to simulate CLI commands and full web pages have captivated users, with example simulations shared, such as the
 **EVA AI interaction profile** 
 on
 [Websim](https://websim.ai/c/p3pZvmAYbsRT2hzBz?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 . Speculations on the revival of World-Sim operated in parallel, and members looked forward to its reintroduction with a "pay-for-tokens" model.
 




---


[OpenAI](https://discord.com/channels/974519864045756446?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Apple's Open Source Pivot with OpenELM** 
 : Apple has released
 **OpenELM** 
 , a family of efficient language models now available on Hugging Face, scaling from 270M to 3B parameters, marking their surprising shift towards open-source initiatives. Details about the models are
 [on Hugging Face](https://huggingface.co/apple/OpenELM?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Conversations Surrounding AI Sentience and Temporal Awareness** 
 : The community engaged in deep discussions emphasizing the difference between
 **sentience** 
 —potentially linked to emotions and motivations—and
 **consciousness** 
 —associated with knowledge acquisition. A parallel discussion pondered if intelligence and temporal awareness in AI are inherently discrete concepts, influencing our understanding of neural network identity and experiential dimension.


* **AI Voice Assistant Tech Talk** 
 : AI enthusiasts compared notes on
 **OpenWakeWords** 
 for homegrown voice assistant development and
 **Gemini** 
 's promise as a Google Assistant rival. Technical challenges highlighted include the intricacies of interrupt AI speech and preferences for push-to-talk versus voice activation.


* **Rate Limit Riddles with Custom GPT Usage** 
 : Users sought clarity on
 **GPT-4's usage caps** 
 especially when recalling large documents and shared tips on navigating the 3-hour rolling cap. The community is exploring the thresholds of rate limiting, particularly when employing custom GPT tools.


* **Prompt Engineering Prowess & LLM Emergent Abilities** 
 : There's a focus on strategic prompt crafting for specific tasks such as developing GPT-based coding for
 **Arma 3's SQF language** 
 . Fascination arises with
 **emergent behaviors** 
 in LLMs, referring to phases of complexity leading to qualitative behavioral changes, exploring parallels to the concept of
 *More Is Different* 
 in prompt engineering contexts.




---


[Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**AI Rollout Must Be Crystal Clear** 
 : Valve's new
 **content policy** 
 requires developers to disclose AI usage on
 **Steam** 
 , particularly highlighting the need for transparency around live-generated AI content and mechanisms that ensure responsible deployment.
 



**Copyright Quandary in Content Creation** 
 : Conversations bubbled up over the legal complexities when generating content with public models such as
 **Stable Diffusion** 
 ; there's a necessity to navigate copyright challenges, especially on platforms with rigorous copyright enforcement like
 **Steam** 
 .
 



**Art Imitates Life or... Itself?** 
 : An inquiry raised by
 **Customluke** 
 on how to create a model or a Lora to replicate their art style using
 **Stable Diffusion** 
 sparked suggestions, with tools like
 **dreambooth** 
 and
 **kohya\_ss** 
 surfaced for model and Lora creation respectively.
 



**Selecting the Better Suited AI Flavor** 
 : A vocal group of users find
 **SD 1.5** 
 superior to
 **SDXL** 
 for their needs, citing sharper results and better training process, evidence that the choice of AI model significantly impacts outcome quality.
 



**Polishing Image Generation** 
 : Tips were shared for improving image generation results, recommending alternatives such as
 **Forge** 
 and
 **epicrealismXL** 
 to enhance the output for those dissatisfied with the image quality from models like
 **ComfyUI** 
 .
 




---


[HuggingFace](https://discord.com/channels/879548962464493619?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **BioMistral Launch for Medical LLMs** 
 :
 [BioMistral](https://huggingface.co/BioMistral/BioMistral-7B?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , a new set of pretrained language models for medical applications, has been introduced, leveraging the capabilities of the foundational Mistral model.


* **Nvidia's Geopolitical Adaptation** 
 : To navigate US export controls, Nvidia has unveiled the RTX 4090D, a China-compliant GPU with reduced power consumption and CUDA cores, detailed in reports from
 [The Verge](https://www.theverge.com/2023/12/29/24018799/nvidia-4090d-china-slower-us-sanctions?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and
 [Videocardz](https://videocardz.com/newz/nvidia-geforce-rtx-4090-with-blower-type-cooler-is-now-on-sale-in-china?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Text to Image Model Fine-Tuning Discussed** 
 : Queries about optimizing text to image models led to suggestions involving the
 [Hugging Face diffusers repository](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Gradio Interface for ConversationalRetrievalChain** 
 : Integration of ConversationalRetrievalChain with Gradio is in the works, with community efforts to include personalized PDFs and discussion regarding interface customization.


* **Improved Image Generation and AI Insights in Portuguese** 
 : New developments include an app at
 [Collate.one](https://collate.one/newsletter?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 for digesting read-later content, advancements in generating high-def images in seconds at
 [this space](https://huggingface.co/spaces/KingNish/Instant-Image?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , and
 [Brazilian Portuguese translations](https://www.youtube.com/watch?v=A9qPlYVeiOs&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 of AI community highlights.


* **Quantization and Efficiency** 
 : There's active exploration on quantization techniques to maximize model efficiency on VRAM-limited systems, with preferences leaning toward Q4 or Q5 levels for a balance between performance and resource management.


* **Table-Vision Models and COCO Dataset Clarification** 
 : There's a request for recommendations on vision models adept at table-based question-answering, and security concerns raised regarding the hosting of the official COCO datasets via an HTTP connection.


* **Call for Code-Centric Resources and TLM v1.0** 
 : The engineering community is seeking more tools with direct code links, as exemplified by
 [awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , and the launch of v1.0 of the Trustworthy Language Model (TLM), introducing a confidence score feature, is celebrated with a
 [playground](https://tlm.cleanlab.ai/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and
 [tutorial](https://help.cleanlab.ai/tutorials/tlm/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .




---


[Eleuther](https://discord.com/channels/729741769192767510?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
---------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Parallel Ponderings Pose No Problems** 
 : Engineers highlighted that some
 **model architectures** 
 , specifically PaLM, employ
 **parallel attention and FFN (feedforward neural networks)** 
 , deviating from the series perception some papers present.


* **Data Digestion Detailing** 
 : The
 **Pile dataset's hash values** 
 were shared, offering a reference for those looking to utilize the dataset in various JSON files, an aid found on
 [EleutherAI's hash list](https://www.eleuther.ai/hashes?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Thinking Inside the Sliding Window** 
 : Dialogue on
 **transformers** 
 considered
 **sliding window attention** 
 and effective receptive fields, analogizing them to convolutional mechanisms and their impact on attention's focus.


* **Layer Learning Ladders Lengthen Leeway** 
 : Discussions about improving transformers' handling of
 **lengthier sequence lengths** 
 touched upon strategies like integrating RNN-type layers or employing dilated windows within the architecture.


* **PyTorch's New Power Player** 
 : A new
 **PyTorch library, torchtitan** 
 , was introduced via a
 [GitHub link](https://github.com/pytorch/torchtitan?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , promising to ease the journey of training larger models.


* **Linear Logic Illuminates Inference** 
 : The mechanics of
 **linear attention** 
 were unpacked, illustrating its sequence-length linearity and constant memory footprint, essential insights for future model optimization.


* **Performance Parity Presumption** 
 : One engineer reported that the
 **phi-3-mini-128k** 
 might match the
 **Llama-3-8B** 
 , triggering a talk on the influences of pre-training data on model benchmarking and baselines.


* **Delta Decision's Dual Nature** 
 : The possibility of
 **delta rule linear attention** 
 enabling more structured yet less parallelizable operations stirred a comparison debate, supported by a
 [MastifestAI blog post](https://manifestai.com/blogposts/faster-after-all/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Testing Through a Tiny Lens** 
 : Members cast doubt on "needle in the haystack" tests for long-context language models, advocating for real-world application as a more robust performance indicator.


* **Prompt Loss Ponderings** 
 : The group questioned the systemic study of masking user prompt loss during supervised fine-tuning (SFT), noting a research gap despite its frequent use in language model training.


* **Five is the GSM8K Magic Number** 
 : There was a consensus suggesting that using
 *5* 
 few-shot examples is the appropriate alignment with the
 **Hugging Face leaderboard** 
 criteria for
 **GSM8K** 
 .


* **VLLM Version Vivisection** 
 : Dialogue identified
 **Data Parallel (DP)** 
 as a stumbling block in updating
 **VLLM** 
 to its latest avatar, while
 **Tensor Parallel (TP)** 
 appeared a smoother path.


* **Calling Coders to Contribute** 
 : The lm-evaluation-harness appeared to be missing a
 `register_filter` 
 function, leading to a call for contributors to submit a PR to bolster the utility.


* **Brier Score Brain Twister** 
 : An anomaly within the
 **ARC evaluation** 
 data led to a suggestion that the Brier score function be refitted to ensure error-free assessments regardless of data inconsistencies.


* **Template Tête-à-Tête** 
 : Interest was piqued regarding the status of a chat templating branch in
 *Hailey's branch* 
 , last updated a while ago, sparking an inquiry into the advancement of this functionality.




---


[OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Mixtral Muddle** 
 : A provider of
 **Mixtral 8x7b** 
 faced an issue of sending blank responses, leading to their temporary removal from OpenRouter. Auto-detection methods for such failures are under consideration.
 



**Soliloquy's Subscription Surprise** 
 : The
 **Soliloquy 8B** 
 model transitioned to a paid service, charging
 **$0.1 per 1M tokens** 
 . Further information and discussions are available at
 [Soliloquy 8B](https://openrouter.ai/models/lynn/soliloquy-l3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**DBRX AI Achieves AI Astonishment** 
 : Fprime-ai announced a significant advancement with their
 **DBRX AI** 
 on LinkedIn, sparking interest and discussions in the community. The LinkedIn announcement can be read
 [here](https://www.linkedin.com/posts/fprime-ai_fprimeailabs-dbrx-ai-activity-7189599191201980417-Te5d?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Creative Model Melee** 
 : Community members argued about the best open-source model for role-play creativity, with
 **WizardLM2 8x22B** 
 and
 **Mixtral 8x22B** 
 emerging as top contenders due to their creative capabilities.
 



**The Great GPT-4 Turbo Debate** 
 : Microsoft's influence on the
 **Wizard LM** 
 project incited a heated debate, leading to a deep dive into the incidence, performance, and sustainability of models like GPT-4, Llama 3, and WizardLM. Resources shared include an
 [incident summary](https://rocky-muscle-755.notion.site/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and a miscellaneous
 [OpenRouter model list](https://openrouter.ai/models?q=free&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 




---


[LlamaIndex](https://discord.com/channels/1059199217496772688?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Create-llama Simplifies RAG Setup** 
 : The
 **create-llama v0.1** 
 release brings new support for
 **@ollama** 
 and vector database integrations, making it easier to deploy RAG applications with llama3 and phi3 models, as detailed in their
 [announcement tweet](https://twitter.com/llama_index/status/1783528887726817653?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**LlamaParse Touted in Hands-on Tutorial and Webinar** 
 : A hands-on tutorial showcases how
 **LlamaParse** 
 ,
 **@JinaAI\_ embeddings** 
 ,
 **@qdrant\_engine vector storage** 
 , and
 **Mixtral 8x7b** 
 can be used to create sophisticated RAG applications, available
 [here](https://twitter.com/llama_index/status/1783601807903863184?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , while KX Systems hosts a webinar to unlock complex document parsing capabilities with
 **LlamaParse** 
 (details in
 [this tweet](https://twitter.com/llama_index/status/1783622871614664990?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ).
 



**AWS Joins Forces with LlamaIndex for Developer Workshop** 
 : AWS collaborates with
 **@llama\_index** 
 to provide a workshop focusing on LLM app development, integrating AWS services and LlamaParse; more details can be found
 [here](https://twitter.com/llama_index/status/1783877951278432733?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Deep Dive into Advanced RAG Systems** 
 : The community engaged in robust discussions on improving RAG systems and shared a video on advanced setup techniques, addressing everything from sentence-window retrieval to integrating structured Pydantic output (
 [Lesson on Advanced RAG](https://learn.deeplearning.ai/courses/building-evaluating-advanced-rag/lesson/5/auto-merging-retrieval?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ).
 



**Local LLM Deployment Strategies Discussed** 
 : There was active dialogue on employing local LLM setups to circumvent reliance on external APIs, with guidance provided in the official
 **LlamaIndex documentation** 
 (
 [Starter Example with Local LLM](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 ), showcasing strategies for resolving import errors and proper package installation.
 




---


[LAION](https://discord.com/channels/823813159592001537?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Llama 3's Mixed Reception** 
 : Community feedback on
 **Llama 3** 
 is divided, with some highlighting its inadequate code recall abilities compared to expectations set by GPT-4, while others speculate the potential for configuration enhancements to bridge the performance gap.
 



**Know Your Customer Cloud Conundrum** 
 : The proposed U.S. "Know Your Customer" policies for cloud services spark concern and discussion, emphasizing the necessity for community input on the
 [Federal Register](https://www.federalregister.gov/documents/2024/01/29/2024-01580/taking-additional-steps-to-address-the-national-emergency-with-respect-to-significant-malicious?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 before the feedback window closes.
 



**Boost in AI Model Training Efficiency** 
 : Innovations in vision model training are making waves with a
 *weakly supervised pre-training method* 
 that races past traditional contrastive learning, achieving
 **2.7 times faster** 
 training as elucidated in this
 [research](https://arxiv.org/abs/2404.15653?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 . The approach shuns contrastive learning's heavy compute costs for a multilabel classification framework, yielding a performance on par with
 **CLIP** 
 models.
 



**The VAST Landscape of Omni-Modality** 
 : Enthusiasm is sighted for finetuning
 **VAST** 
 , a Vision-Audio-Subtitle-Text Omni-Modality Foundation Model. The project indicates a stride towards omni-modality with the resources available at its
 [GitHub repository](https://github.com/txh-mercury/vast?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Nightshade's Transparency Troubles** 
 : The guild debates the effectiveness and transparency of
 **Nightshade** 
 with a critical lens on autoencoder capabilities and reluctances in the publishing of potentially controversial findings.
 




---


[OpenInterpreter](https://discord.com/channels/1146610656779440188?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Mac Muscle Meets Interpreter Might** 
 : Open Interpreter's
 **New Computer Update** 
 has significantly improved local functionality, particularly with
 **native Mac integrations** 
 . The implementation allows users to control Mac's native applications using simple commands such as
 `interpreter --os` 
 , as detailed in their
 [change log](https://changes.openinterpreter.com/log/ncu-ii?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Eyes for AI** 
 : Community members highlighted the
 **Moondream tiny vision language model** 
 , providing resources like the
 [Img2TxtMoondream.py script](https://github.com/CodeAKrome/bootcupboard/blob/main/llm-img/Img2TxtMoondream.py?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 . Discussions also featured
 **LLaVA** 
 , a multimodal model hosted on
 [Hugging Face](https://huggingface.co/liuhaotian/llava-v1.6-34b?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , which is grounded in the powerful
 **NousResearch/Nous-Hermes-2-Yi-34B** 
 model.
 



**Loop Avoidance Lore** 
 : Engineers have been swapping strategies to mitigate looping behavior in local models, considering solutions ranging from tweaking
 *temperature settings* 
 and
 *prompt editing* 
 to more complex
 *architectural changes* 
 . An intriguing concept, the
 *frustration metric* 
 , was introduced to tailor a model's responses when stuck in repetitive loops.
 



**Driving Dogs with Dialogue** 
 : A member inquired about the prospect of leveraging
 **Open Interpreter** 
 for commanding the
 **Unitree GO2 robodog** 
 , sparking interest in possible interdisciplinary applications. Technical challenges, such as setting dummy API keys and resolving namespace conflicts with Pydantic, were also tackled with shared solutions.
 



**Firmware Finality** 
 : The
 **Open Interpreter 0.2.5 New Computer Update** 
 has officially graduated from beta, including the fresh enhancements mentioned earlier. A query about the update's beta status led to an affirmative response after a version check.
 




---


[OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**CEO's Nod to a Member's Tweet** 
 : A participant was excited about the
 *CEO of Hugging Face acknowledging their tweet* 
 ; network and recognition are alive in the community.
 



**Tech Giants Jump Into Fine-tuning** 
 : With examples like
 **[Meditron](https://arxiv.org/abs/2311.16079?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its)**
 , discussion on fine-tuning language models for specific uses is heating up, highlighting the promise for domain-specific improvements and hinting at an
 **upcoming paper** 
 on continual pre-training.
 



**Trouble in Transformer Town** 
 : An 'AttributeError' surfaced in
 **transformers 4.40.0** 
 , tripping up a user, serving as a cautionary tale that even small updates can break workflows.
 



**Mixing Math with Models** 
 : Despite some confusion, inquiries were made about integrating
 **zzero3** 
 with
 **Fast Fourier Transform (fft)** 
 ; keep an eye out for this complex dance of algorithms.
 



**Optimizer Hunt Heats Up** 
 : The
 **FSDP (Fully Sharded Data Parallel)** 
 compatibility with optimizers remains a hot topic, with findings that
 **AdamW** 
 and
 **SGD** 
 are in the clear, while
 `paged_adamw_8bit` 
 is not supporting FSDP offloading, leading to a quest for alternatives within the
 **OpenAccess-AI-Collective/axolotl** 
 resources.
 




---


[Cohere](https://discord.com/channels/954421988141711382?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Upload Hiccups and Typographic Tangles** 
 : Users in the
 **Cohere** 
 guild tackled issues with the
 **Cohere Toolkit** 
 on Azure, pointing to the paper clip icon for uploads; despite this, problems persisted with the upload functionality going undiscovered. The
 **Cohere typeface** 
 's licensing on GitHub provoked discussion; it's not under the MIT license and is slated for replacement.
 



**Model Usage Must-Knows** 
 : Discussion clarified that Cohere's
 [Command+ models](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 are available with open weight access but not for commercial use, and the training data is not shared.
 



**Search API Shift Suggestion** 
 : The guild mulled over the potential switch from
 **Tavily** 
 to the
 **Brave Search API** 
 for integrating with the Cohere-Toolkit, citing potential benefits in speed, cost, and accuracy in retrieval.
 



**Toolkit Deployment Debates** 
 : Deployment complexities of the Cohere Toolkit on Azure were deliberated, where selecting a model deployment option is crucial and the API key is not needed. Conversely, local addition of tools faced issues with PDF uploads and sqlite3 version compatibility.
 



**Critical Recall on 'Hit Piece'** 
 : Heated discussions emerged over the criticism of a "hit piece" against
 *Cohere* 
 , with dialogue focused on the responsibility of AI agents and their real-world actions. A push for critical accountability emerged, with members reinforcing the need to back up critiques with substantial claims.
 




---


[tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Tinygrad Sprints Towards Version 1.0** 
 : Tinygrad is gearing up for its 1.0 version, spotlighting an API that's nearing stability, and has a toolkit that includes
 [installation guidance](https://tinygrad.github.io/tinygrad/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , a
 [MNIST tutorial](https://tinygrad.github.io/tinygrad/mnist/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , and comprehensive
 [developer documentation](https://tinygrad.github.io/tinygrad/developer/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Comma Begins Tinybox Testing with tinygrad** 
 : George Hotz emphasized tinybox by comma as an exemplary testbed for tinygrad, with a focus maintained on software over hardware, while a potential tinybox 2 collaboration looms.


* **Crossing off Tenstorrent** 
 : After evaluation, a partnership with Tenstorrent was eschewed due to inefficiencies in their hardware, leaving the door ajar for future collaboration if the cost-benefit analysis shifts favorably.


* **Sorting Through tinygrad's Quantile Function Challenge** 
 : A dive into tinygrad's development revealed efforts to replicate
 `torch.quantile` 
 for diffusion model sampling, a complex task necessitating a precise sorting algorithm within the framework.


* **AMMD's MES Offers Little to tinygrad** 
 : AMD's Machine Environment Settings (MES) received a nod from Hotz for its detailed breakdown by Felix from AMD, but ultimately assessed as irrelevant to tinygrad's direction, with efforts focused on developing a PM4 backend instead.




---


[Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Strong Performer: Hermes 2.5 Edges Out Hermes 2** 
 : Enhanced with code instruction examples,
 **Hermes 2.5** 
 demonstrates superior performance across various benchmarks when compared to
 **Hermes 2** 
 .
 



**Security in the Limelight** 
 : Amidst sweeping software and feature releases by Modular, addressing security loopholes becomes critical, emphasizing protection against supply chain attacks like the
 **XZ incident** 
 and the trend of open-source code prevalence in software development forecasted to hit
 **96% by 2024** 
 .
 



**Quantum Complexity Through A Geometric Lens** 
 : Members discussed how the geometric concept of the
 **amplituhedron** 
 could simplify quantum particle scattering amplitudes, with machine learning being suggested as a tool to decipher increased complexities in visualizing
 **quantum states** 
 as systems scale.
 



**All About Mojo** 
 : Dialogue around the
 **Mojo Programming Language** 
 covered topics like assured memory cleanup by the OS, the variance between
 `def` 
 and
 `fn` 
 functions with examples found
 [here](https://docs.modular.com/mojo/manual/functions?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , and the handling of mixed data type lists via
 `Variant` 
 that requires improvement.
 



**Moving Forward with Mojo** 
 : ModularBot flagged an issue filed on GitHub about
 **Mojo** 
 , urged members to use issues for better tracking of concerns, for instance, about
 `__copyinit__` 
 semantics
 [via GitHub Gist](https://gist.github.com/modularbot/6aed759930420cd70f38795dbcb874fe?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , and reported a cleaner update in code with more insertions than deletions, achieving better efficiency.
 




---


[LangChain AI](https://discord.com/channels/1038097195422978059?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**A Tricky Query for Anti-Trolling AI Design** 
 : A user proposed designing an
 **anti-trolling AI** 
 and sought suggestions on how the system could effectively target online bullies.
 



**Verbose SQL Headaches** 
 : Participants shared experiences with open-source models like
 **Mistral** 
 and
 **Llama3** 
 generating overly verbose SQL responses and encountered an
 `OutputParserException` 
 , with links to
 [structured output support](https://python.langchain.com/docs/integrations/chat/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and examples of invoking SQL Agents.
 



**RedisStore vs. Chat Memory** 
 : The community clarified the difference between
 **stores** 
 and
 **chat memory** 
 in the context of LangChain integrations, emphasizing the specific use of
 `RedisStore` 
 for key-value storage and
 **Redis Chat Message History** 
 for session-based chat persistence.
 



**Techie Tutorial on Model Invocation** 
 : There was a discussion on the correct syntax when integrating prompts into LangChain models via JavaScript, with recommendations for using
 `ChatPromptTemplate` 
 and
 `pipe` 
 methods for chaining prompts.
 



**Gemini 1.5 Access with a Caveat** 
 : Users discussed the integration of
 **Gemini 1.5 Pro** 
 with LangChain, highlighting that it necessitates
 `ChatVertexAI` 
 instead of
 `ChatGoogleGenerativeAI` 
 and requires configuring the
 `GOOGLE_APPLICATION_CREDENTIALS` 
 environment variable for proper access.
 




---


[Latent Space](https://discord.com/channels/822583790773862470?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Apple Bites the Open Source Apple** 
 : Apple has stepped into the open source realm, releasing a suite of models with parameters ranging from 270M to 3B, with the
 [270M parameter model available on Hugging Face](https://huggingface.co/collections/apple/openelm-instruct-models-6619ad295d7ae9f868b759ca?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Dify Platform Ups and Downs** 
 : The open-source LLM app development platform Dify is gaining traction for combining AI workflows and model management, although concerns have arisen about its lack of
 [loops and context scopes](https://github.com/langgenius/dify?tab=readme-ov-file&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**PyTorch Pumps Up LLM Training** 
 : PyTorch has introduced
 [Torchtitan](https://github.com/pytorch/torchtitan?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , a library dedicated to aiding the training of substantial language models like llama3 from scratch.
 



**Video Gen Innovation with SORA** 
 : OpenAI's SORA, a video generation model that crafts videos up to a minute long, is getting noticed, with user experiences and details explored in an
 [FXGuide article](https://www.fxguide.com/fxfeatured/actually-using-sora/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**MOD Layers for Efficient Transformer Training** 
 : The 'Mixture of Depths' paper was presented, proposing an accelerated training methodology for transformers by alternately using new MOD layers and traditional transformer layers, introduced in the
 [presentation](https://paper-club.ivanleo.com/papers/mixture-of-depths?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 and detailed in the paper's
 [abstract](https://arxiv.org/abs/2402.00841?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 




---


[Mozilla AI](https://discord.com/channels/1089876418936180786?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Phi-3-Mini-4K Instruct Powers Up** 
 : Utilizing
 **Phi-3-Mini-4K-Instruct** 
 with llamafile provides a setup for high-quality and dense reasoning datasets as discussed by members, with
 [integration steps outlined on Hugging Face](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its#how-to-use-with-llamafile) 
 .


* **Model Download Made Easier** 
 : A README update for
 **Mixtral 8x22B Instruct llamafile** 
 includes a download tip: use
 `curl -L` 
 for smooth redirections on CDNs, as seen in the
 [Quickstart guide](https://huggingface.co/jartine/Mixtral-8x22B-Instruct-v0.1-llamafile?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Llamafile and CPUs Need to Talk** 
 : An issue with running llamafile on an
 **Apple M1** 
 Mac surfaced due to AVX CPU feature requirements, with the temporary fix of a system restart and advice for using smaller models on 8GB RAM systems shared in this
 [GitHub issue](https://github.com/Mozilla-Ocho/llamafile/issues/327?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its#issuecomment-2053680659) 
 .


* **Windows Meets Llamafile, Confusion Ensues** 
 : Users reported
 **Windows Defender** 
 mistakenly detecting llamafile as a trojan. Workarounds proposed included using virtual machines or whitelisting, with the reminder that official binaries can be found
 [here](https://www.microsoft.com/en-us/wdsi/filesubmission?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .


* **Resource-Hungry Models Test Limits** 
 : Engaging the 8x22B model requires heavy resources, with references to a recommended 128GB RAM for stable execution of
 [Mistral 8x22B model](https://github.com/Mozilla-Ocho/llamafile/releases/tag/0.8?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , marking the necessity for big memory footprints when running sophisticated AI models.




---


[DiscoResearch](https://discord.com/channels/1178995845727785010?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Llama Beats Judge in Judging** 
 : On the
 **Judgemark** 
 benchmark,
 **Llama-3-70b** 
 showcased impressive performance, demonstrating its potential for fine-tuning purposes in
 **disco-judge** 
 applications, as it supports at least 8k context length. The community also touched on collaborative evaluation efforts, with references to advanced judging prompt design to assess complex rubrics.
 



**Benchmarking Models and Discussing Inference Issues** 
 :
 **Phi-3-mini-4k-instruct** 
 unexpectedly ranked lower on the
 **eq-bench** 
 leaderboard despite promising scores in published evaluations. In model deployment, discussions highlighted issues like slow initialization and inference times for
 **DiscoLM\_German\_7b\_v1** 
 and potential misconfigurations that could be remedied using
 `device_map='auto'` 
 .
 



**Tooling API Evaluation and Hugging Face Inquiries** 
 : Community debates highlighted
 **Tgi** 
 for its API-first, low-latency approach and praised
 **vllm** 
 for being a user-friendly library optimized for cost-efficiency in deployment. Queries on Hugging Face's batch generation capabilities sparked discussion, with community involvement evident in a GitHub issue exchange.
 



**Gratitude and Speculation in Model Development** 
 : Despite deployment issues, members have expressed appreciation for the
 **DiscoLM** 
 model series, while also speculating about the potential of constructing an
 **8 x phi-3 MoE model** 
 to bolster model capabilities.
 **DiscoLM-70b** 
 was also a hot topic, with users troubleshooting errors and sharing usage experiences.
 



**Success and Popularity in Model Adoption** 
 : The adaptation of the
 **Phi-3-mini-4k** 
 model, referred to as llamafication, yielded a respectable EQ-Bench Score of 51.41 for German language outputs. Conversation also pinpointed the swift uptake of the
 **gguf** 
 model, indicated by a notable number of downloads shortly after its release.
 




---


[Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Claude Displays Depth and Structure** 
 : In a rich discussion, the behavior and training of
 **Claude** 
 were considered "mostly orthogonal" to Anthropic's vision, revealing unexpected depth and structural understanding through
 **RLAIF training** 
 . Comparisons were made to concepts like "Jungian individuation" and conversation threads
 [highlighted Claude's capabilities](https://x.com/repligate/status/1783426037210026372?s=46&t=xxWoJxAS_7-BBFC2ro84Zw&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Debating the Merits of RLHF vs. KTO** 
 : A comparison between
 **Reinforcement Learning from Human Feedback (RLHF)** 
 and
 **Knowledge-Targeted Optimization (KTO)** 
 sparked debate, considering their suitability for different commercial deployments.
 



**Training Method Transition Yields Improvements** 
 : An interview was mentioned where a progression in training methods from
 **Supervised Fine Tuning (SFT)** 
 to
 **Data Programming by Demonstration (DPO)** 
 , and then to
 **KTO** 
 , led to improved performance based on user feedback.
 



**Unpacking the Complexity of RLHF** 
 : The community acknowledged the intricacies of
 **RLHF** 
 , especially as they relate to varying data sources and their impact on downstream evaluation metrics.
 



**Probing Grad Norm Spikes** 
 : A request for clarity on the implications of gradient norm spikes during pretraining was made, emphasizing the potential adverse effects but specifics were not delivered in the responses.
 




---


[Skunkworks AI](https://discord.com/channels/1131084849432768614?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



**Moondream Takes On CAPTCHAs** 
 : A
 [video guide](https://www.youtube.com/watch?v=Gwq7smiWLtc&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 showcases fine-tuning the
 **Moondream Vision Language Model** 
 for better performance on a CAPTCHA image dataset, aimed at improving its image recognition capabilities for practical applications.
 



**Low-Cost AI Models Make Cents** 
 : The document "Low-Cost Language Models: Survey and Performance Evaluation on Python Code Generation" was shared, covering evaluations of
 **CPU-friendly language models** 
 and introducing a novel dataset with 60 programming problems. The use of a Chain-of-Thought prompt strategy is highlighted in the
 [survey article](https://arxiv.org/abs/2404.11160?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Meet, Greet, and Compute** 
 : AI developers are invited to a meetup at
 **Cohere space** 
 in Toronto, which promises networking opportunities alongside lightning talks and demos — details available on the
 [event page](https://lu.ma/devs5?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 .
 



**Arctic Winds Blow for Enterprises** 
 :
 **Snowflake Arctic** 
 is introduced via a
 [new video](https://www.youtube.com/watch?v=nV6eIjnHEH0&utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , positioning itself as a cost-effective, enterprise-ready Large Language Model to complement the suite of AI tools tailored for business applications.
 




---


[Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Run Models Locally with Ease** 
 : Engineers explored
 [jan.ai](https://jan.ai/?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 , a GUI commended for its straightforward approach for running GPT models on local machines, potentially simplifying the experimentation process.
* **Apple Enters the Language Model Arena** 
 : The new
 [OpenELM](https://huggingface.co/apple/OpenELM?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 series introduced by Apple provides a spectrum of efficiently scaled language models, including instruction-tuned variations, which could change the game for parameter efficiency in modeling.




---


[Alignment Lab AI](https://discord.com/channels/1087862276448595968?utm_source=ainews&utm_medium=email&utm_campaign=ainews-apples-openelm-beats-olmo-with-50-of-its) 
 Discord
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


* **Llama 3 Steps Up in Topic Complexity** 
 : Venadore has started experimenting with
 **llama 3** 
 for topic complexity classification, reporting promising results.




---



 The
 **LLM Perf Enthusiasts AI Discord** 
 has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
 




---



 The
 **AI21 Labs (Jamba) Discord** 
 has no new messages. If this guild has been quiet for too long, let us know and we will remove it.
 




---






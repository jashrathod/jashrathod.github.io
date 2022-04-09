---
layout: post
title: Zero-Shot Text-to-Image Generation
subtitle: "DALL-E: OpenAI's text-to-image generation model"
thumbnail-img: ""
share-img: /assets/img/openai-dalle.jpg
tags: [Text-to-image, OpenAI, Discrete VAE, Transformer, Zero-shot learning, Machine Learning, Deep Learning]
comments: true
---

OpenAI's DALL-E is a simple approach based on Transformer that autoregressively transforms models the text & image tokens as single stream of date.

{: .box-note}
**Autoregressive:** predicts future behavior based on past behavior, used for forecasting when there is some correlation between values in time series.

### Introduction:

- **History of text to image synthesis:**

    - Various approaches have tried to improve the visual fidelity, but there is still a problem of 
        1. object distortion 
        2. illogical funding object placement 
        3. unnatural placement of foreground & background elements 
    - Eg: extending DRAW generative models, to condition on image captions to generate novel visual scenes, GANs for improved image fidelity, energy-based framework for conditional image generation to improve sample quality

- When compute, model size, and data are scaled carefully, auto regressive transformers achieve great results in several domains like text, images, audio.

- This work demonstrates Training a 12-billion parameter autoregressive transformer on 250-million image-text pairs collected from internet results in flexible, high fidelity generative model of images controllable through natural language.

{: .box-note}
**Zero-shot Learning:** In this, at test time, learner observes samples from classes that were not observed during training & needs to predict the class they belong to. So prediction happens using auxiliary information and on the fly.

### Method:

- To model a transformer to autoregressively model text and image token as single stream of data.

{: .box-note}
**Note:** directly using pixels of actual image proves computationally expensive & requires lot of memory.

#### Stage 1:

- Train DiscreteVAE to compress 256x256 RGB image into 32x32 grid of image tokens, where each element can assume 8192 possible values.
- This reduces context size of transformer by a factor of 192 without significant degradation in visual quality.

![DALL-E Stage 1](/assets/img/dalle-stage1.png)

#### Stage 2:
    
- Concatenate upto 256 BPE-encoded text tokens with 32x32=1024 image tokens.
- Train autoregressive transformer to model joint distribution over the text and image tokens.

![DALL-E Stage 2](/assets/img/dalle-stage2.png)

### Complete DALL-E pipeline

![DALL-E](/assets/img/dalle.png)


Thanks for reading!

***

Got any questions or suggestions? Want to share any thoughts or ideas with me? Feel free to reach out to me on [LinkedIn](https://linkedin.com/in/jash-rathod). Always happy to help!

Also, you can view by other works on [GitHub](https://github.com/jashrathod) and [my blog](https://jashrathod.github.io/).

Till then, see you in my next post!

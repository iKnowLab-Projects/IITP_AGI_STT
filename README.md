# IITP_AGI_STT Repository Description

This repository contains a comprehensive implementation of a domain-specific Speech-to-Text (STT) system tailored for robotic command recognition. The project leverages the Qwen model as its foundation and fine-tunes it using the specialized BridgeDataV2-audio dataset, which focuses on robot command utterances.

## Overview

The primary objective of this repository is to provide an end-to-end solution for training, fine-tuning, and deploying a speech recognition model that excels in understanding robot-specific commands and instructions. By utilizing the Qwen model architecture as a base, the system benefits from strong pre-trained language understanding capabilities, which are then adapted to the specific domain of robotic control through targeted fine-tuning.

## Dataset

The fine-tuning process employs the BridgeDataV2-audio dataset (available at https://huggingface.co/datasets/iknow-lab/BridgeDataV2-audio), which is specifically curated for robot command recognition tasks. This dataset contains audio samples of various robot commands and instructions, making it ideal for training models that need to accurately interpret human speech in robotic interaction scenarios.

## Key Features

- **Model Fine-tuning**: The repository includes complete code and configurations for fine-tuning the Qwen model on the BridgeDataV2-audio dataset, enabling domain-specific adaptation for robot command recognition.

- **API Service Integration**: The trained model is packaged with a FastAPI-based web service, allowing easy integration and deployment of the STT system as a RESTful API. This makes it straightforward to incorporate the speech recognition capabilities into various robotic applications and systems.

- **Production-Ready Deployment**: The FastAPI implementation ensures that the model can be efficiently served in production environments, with support for asynchronous processing and scalable request handling.

## Use Cases

This repository is particularly useful for:
- Developers building voice-controlled robotic systems
- Researchers working on human-robot interaction
- Engineers implementing speech interfaces for industrial robots
- Teams developing assistive robotics with voice command capabilities

The combination of domain-specific fine-tuning and API-based deployment makes this solution both accurate for robot command recognition and practical for real-world implementation.

--------------------------
<a href="https://example.com">Untitled</a> © 1999 by <a href="https://example.com">Jane Doe</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0</a><img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;"><img src="https://mirrors.creativecommons.org/presskit/icons/nc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
--------------------------

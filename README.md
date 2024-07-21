# Unsloth Project - Distributed Training Attempt on 4090

## Introduction
This repository tracks our ongoing attempts at distributed training of the Unsloth Project with the NVIDIA GeForce RTX 4090 GPU.

## Current Status
The project is currently in progress. We are actively working on optimizing the training process and achieving better performance.

## Goals
The ultimate aim of this project is to achieve efficient and scalable distributed training to enhance the model's performance and generalization ability.

## Setup
1. Hardware Requirements
    - NVIDIA GeForce RTX 4090 GPU
    - Sufficient memory and CPU resources
2. Software Installation
    - Install the latest version of CUDA and cuDNN.
    - Install the necessary deep learning frameworks such as TensorFlow or PyTorch.

## Training Process (In Progress)
1. Success in Running the Continued Pretraining of Unsloth
 - Successful environment setup. The required Python environment for this is version 3.9. For simplicity, we directly installed the virtualenv tool on the original environment for version management. The Dockerfile can be found in the code repository at https://github.com/InnoMakerQiu/det_docker_build. After pulling the specified image, activate the Python 3.9 environment and then install as per the official Unsloth tutorial.
 - Then run according to the official notebook, and the result was smooth. We were able to obtain 150 rounds of iterative training in 20 minutes.

## 2. Success in Running the Distributed Training of Unsloth
 - The trainer of Unsloth inherits from Trainer. Therefore, to achieve distributed training of Unsloth, the key is to implement the distributed training of the trainer. For detailed explanations, please refer to the documentation.

## Planned Next Steps
1. Further optimize the model architecture.
2. Explore different training strategies.
3. Conduct more comprehensive performance evaluations.

## Conclusion
This is an evolving project, and we are committed to sharing updates and insights as we make progress. Stay tuned for more!

## Contact
If you have any questions or suggestions during this process, please contact us at [2377767409@qq.com].

Thank you for your interest and support!
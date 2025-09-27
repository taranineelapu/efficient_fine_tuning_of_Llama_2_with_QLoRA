# Fine-Tuning Large Language Model under Resource Constraints: A Case Study
This project demonstrates how large language models (LLMs) can be fine-tuned effectively under limited computational resources. This case study focuses on the Llama-2-7b-chat-hf model, applying parameter-efficient techniques such as QLoRA and 4-bit quantization to achieve meaningful results on a single GPU. Real-time training progress is monitored using TensorBoard, demonstrating effective fine-tuning of large language models under resource constraints.

# Tech Stack
Modeling: Hugging Face Transformers, Llama-2-7b-chat-hf
Fine-Tuning Techniques: LoRA, QLoRA (4-bit precision), BitsAndBytes
Monitoring: TensorBoard
Environment: Python, single GPU

# Usage
Run fine-tuning on consumer-grade GPUs with limited VRAM
Track training metrics using TensorBoard
Save and reuse fine-tuned models for downstream NLP tasks

# Features
Model Setup:
Llama-2-7b-chat-hf loaded from Hugging Face
Fine-tuned using LoRA with QLoRA (4-bit quantization) for memory savings
Configured with efficient hyperparameters (LoRA rank, alpha, dropout)

Data Preparation:
Used guanaco-llama2-1k dataset, pre-aligned with Llama 2 prompt format
Smaller dataset volume to accommodate limited compute

Training & Monitoring:
Single-epoch fine-tuning with gradient checkpointing for memory optimization
Real-time performance tracking via TensorBoard

Efficiency Gains:
Training completed in ~24 minutes with limited hardware
Achieved reasonable convergence while minimizing GPU usage

Note:	However, please note that this code was written and specifically designed for systems with an NVIDIA GPU and driver. Hence, if your system does not have an NVIDIA GPU, this will throw a runtime error.

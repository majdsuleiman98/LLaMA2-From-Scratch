# 🦙 LLaMA2 From Scratch

This repository implements a simplified transformer architecture inspired by Meta's LLaMA2, built completely from scratch using PyTorch.

## 🚀 Project Overview

The goal of this project is to deeply understand the inner workings of modern transformer models by:

- Manually implementing multi-head self-attention
- Adding rotary positional embeddings (RoPE)
- Using RMSNorm instead of LayerNorm
- Managing key/value caches for efficient inference
- Supporting grouped query attention (GQA)

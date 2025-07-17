# Tiny Diffusion Model

This project provides a minimal and readable implementation of diffusion models applied to simple two-dimensional datasets.

The goal is to understand the core logic behind score-based diffusion models: how they add noise to data over time, and how a neural network can learn to reverse this process to generate new samples.

No optimizations, tricks, or production-level code are included. The idea is to focus on clarity and learning.

---

## Datasets

The model is trained on small 2D datasets, including fractals and simple geometric structures. These datasets make it easier to visualize and interpret the training and sampling process.

Example datasets:

- Barnsley Fern
- Sierpinski Triangle
- Spiral
- Koch Snowflake
- Mandelbrot Set
- Swiss Roll

Here is a sample of the datasets used:

<p align="center">
  <img src="test_dataset.png" alt="Example of datasets" width="500"/>
</p>

---

## Project Structure


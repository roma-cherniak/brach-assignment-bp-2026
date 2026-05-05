# Assignment

Wrap a pretrained computer vision model into a Python package with a simple inference interface, then publish it to GitHub and PyPI.

**Estimated time:** 30–60 minutes

---

## Getting started

1. **Fork this repository** to your own GitHub account — this is where you'll start from.
2. Clone your fork locally and build your solution on top of it.
3. When you're done, your fork will contain the final package code.

---

## Requirements

### 1. Package setup
- Use **`uv`** to initialize and manage the project.

### 2. Model
- The model must be implemented in **raw PyTorch**. Write your own `nn.Module` class.
- Topic: **computer vision**.
- The model must load **pre-computed weights** from a `.pth` / `.pt` file. The weights can be bundled in the repo.

### 3. Inference interface

At minimum it must support: loading the model and running prediction on a single image.

### 4. Publishing
- **GitHub:** public repository with all the code (your fork of this repo).
- **PyPI** : the package must be installable via `pip install <package-name>`.

### 5. README.md
Short, clear documentation in your fork containing:
- **Installation** — a single `pip install` command.
- **Quick start** — 5–10 lines of example code.
- **API reference** — brief description of the main functions/classes.
- **Evaluation proposal** — how you would evaluate this model on a task I'll assign during the review. Describe the dataset, metrics, pipeline, optionally pseudocode. One paragraph + bullet points is enough. **You do not need to actually run the evaluation.**

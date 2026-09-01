# Generalizing Tree-to-Tree Neural Networks with a Pointer-Generator Mechanism

Independent research project extending a neural source-code translation model to support arbitrary identifiers and numeric literals, instead of a fixed training vocabulary.

## Background

[Chen et al. (2018), "Tree-to-tree Neural Networks for Program Translation"](https://arxiv.org/abs/1802.03691) (NeurIPS) introduced a tree-structured encoder-decoder model for translating source code between programming languages (in this project, CoffeeScript to TypeScript). The original model represents each variable name and numeric constant as an entry in a fixed output vocabulary learned at training time. In practice, this means the model can only reproduce variable names and constants it has already seen, which does not hold for realistic code.

## What this project does

This project modifies the tree-to-tree architecture to remove that constraint:

- **Pointer-generator mechanism** (adapted from [See et al. (2017)](https://arxiv.org/abs/1704.04368), originally proposed for abstractive summarization): at each decoding step, the model learns to choose between generating a token from a fixed vocabulary or copying an identifier/constant directly from the source tree via attention. Adapting this to translation (rather than summarization) required restricting copying to identifiers and constants only, since source and target are different languages.
- **Bidirectional tree-LSTM encoder**: the original encoder only passes child-node states upward, causing the root node to absorb all subtree information and pushing attention toward degenerate, overfit behavior. This project adds a top-down pass so encoder states carry positional and structural context, not just aggregated content.

## Results

- The modified model translates programs using identifiers and constants outside the original fixed vocabulary — something the baseline could not do at all.
- On the original benchmark (fixed vocabulary), this comes at a modest cost: token-level accuracy drops by at most ~5 percentage points and program-level (exact-match) accuracy by at most ~9 percentage points relative to Chen et al.'s reported numbers.
- Generalization to variable/constant configurations outside the training distribution improves substantially, though the original benchmark wasn't designed to isolate overfitting from vocabulary coverage, so this should be read as a qualitative improvement rather than a precise quantitative one.

## Limitations / future work

- Accuracy on longer programs remains limited, likely because temporary variables introduced during translation are still drawn from a fixed output set rather than generated dynamically.
- The approach still requires paired (source, target) programs for supervision; building such pairs at scale is expensive. Dual-learning-style approaches that don't require 1:1 pairs would be a natural extension.

## Status

This was completed as a high school graduation research project (2019-2020). The code has not been actively maintained since; it's shared here as a record of the approach and findings rather than production-quality software.

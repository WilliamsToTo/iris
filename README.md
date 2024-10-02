# IRIS - Iterative Retrieval and Integrated System for Real-Time Causal Discovery

## Directory Structure
- `./data_woTable/` - Contains all datasets
- `./algs/` - Contains implementation of various algorithms
  - Statistical algorithms (PC, GES, NOTEARS)
  - Value extraction component
  - Google retrieval component

## Running the Code

### 1. Real-time Causal Discovery
This script performs data collection, value extraction, and hybrid causal discovery.

```bash
python run_causal_discovery_realtime.py --dataset cancer --alg ges --llm gpt-4o
```

### 2. Missing Variable Proposal
This script implements the missing variable proposal component.

```bash
python run_new_variable_proposal.py --dataset cancer --llm gpt-4o
```

## Setup Requirements

Before running the code, you need to set up your Google API key:

1. Watch [this tutorial video](https://www.youtube.com/watch?v=D4tWHX2nCzQ) to learn how to get a Google API key
2. Add your API key to `./algs/retrieve/google_retrieve.py`

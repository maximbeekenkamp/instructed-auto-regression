# Research Proposal

**Author**: Qian Zhang, Maxim Beekenkamp

## Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across various natural language processing tasks. However, their application in scientific and engineering domains remains limited. This proposal introduces **Instructed Auto-regression (IAG)**, a novel approach that combines the strengths of LLMs with domain-specific data to enhance performance in these fields. By leveraging the auto-regressive nature of LLMs, IAG aims to model complex dynamic systems effectively.

## Related Work

### Operator Learning for Reconstruction Problems
This approach employs transformers to reconstruct missing information from observed data. Its limitation lies in the ability to obtain reconstruction data only concurrently with the observed data, restricting its applicability in scenarios requiring predictions beyond the observed timeframe.

### Neural Ordinary Differential Equations (Neural ODEs)
Neural ODEs are models that learn to approximate the dynamics of systems and have been successfully applied to tasks such as time-series prediction and control. However, they necessitate complete data for each time step during training, which can be a constraint when dealing with incomplete or irregular datasets.

## Method

Consider a continuous-time process \( X(t, x) \in \mathbb{R}^n \), where \( x \in \mathbb{R}^d \). Assume it satisfies a dynamic system:

\[
\frac{dX(t, x)}{dt} = f(X(t, x), t)
\]

Here, \( f \) is a function that may include differential operators such as \( \nabla \) (gradient) or \( \Delta \) (Laplacian). The objective is to learn the system’s dynamics from observed data. Unlike traditional Neural ODE setups, where directly learning \( f \) with differential operators can be challenging, IAG proposes to learn the system’s dynamics implicitly through auto-regression.

The inference process involves predicting the system’s state at a future time step based on prior observations:

- \( (t_0, x_0, X(t_0, x_0)), (t_1, x_1, \_) \Rightarrow X(t_1, x_1) \)
- \( (t_0, x_0, X(t_0, x_0)), (t_1, x_1, X(t_1, x_1)), (t_2, x_2, \_) \Rightarrow X(t_2, x_2) \)
- ...
- \( (t_0, x_0, X(t_0, x_0)), \dots, (t_n, x_n, \_) \Rightarrow X(t_n, x_n) \)

In this framework:

- \( t_i \) represents the time step.
- \( x_i \) denotes the spatial coordinates.
- \( X(t_i, x_i) \) is the system’s state at the given time and location.

The goal is to learn the system’s dynamics by sequentially predicting the next state in the sequence. A transformer model can be employed to capture these dynamics by training it on observed data, enabling it to generate accurate predictions for future states.

## Implementation

### PyTorch
Implementing the IAG model using vanilla PyTorch involves constructing a transformer-based architecture that processes observed data to predict subsequent states. While straightforward, this approach may lack optimization techniques such as key-value (KV) caching, potentially resulting in slower training and inference times, particularly for large datasets extending beyond simple ODEs.

### Hugging Face Transformers
A more efficient implementation can be achieved by leveraging the Hugging Face Transformers library. This library offers a wide range of pre-trained models and optimization techniques that can enhance the performance of the IAG model. By customizing the model architecture and training procedures, and fine-tuning on domain-specific data, the IAG model’s efficiency and versatility in scientific and engineering applications can be significantly improved.

## Examples

### 1. Ordinary Differential Equations (ODEs)

- **Simple Linear ODE**: Modeling exponential growth or decay processes.
- **Van der Pol Oscillator**: Analyzing nonlinear oscillatory systems with applications in electrical circuits and biological rhythms.
- **Lorenz System**: Studying chaotic systems, pertinent to meteorology and fluid dynamics.

### 2. Fluid Dynamics

- **Vortex Street**: Predicting patterns in fluid flow around bluff bodies, relevant in engineering designs.
- **Turbulent Jet Flow**: Understanding mixing processes in jet propulsion and combustion systems.
- **Weather Prediction**: Forecasting atmospheric conditions by modeling complex meteorological phenomena.

By applying the IAG approach to these examples, the potential of LLMs in capturing and predicting complex dynamic behaviors in scientific and engineering contexts can be effectively explored.


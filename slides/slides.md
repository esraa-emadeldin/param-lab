---
marp: true
title: Parameters vs Hyperparameters
paginate: true
size: 16:9
style: |
  /* Colours: blue = parameters (learned), orange = hyperparameters (chosen) */
  section {
    font-family: 'Segoe UI', 'Inter', sans-serif;
    background: #F4F1EA;
    color: #16181D;
    font-size: 30px;
    padding: 60px 80px;
  }
  h1, h2 { color: #16181D; font-weight: 800; }
  h2 { font-size: 52px; margin-bottom: 24px; }
  strong { color: #2F5FC4; }
  em { color: #B5530F; font-style: normal; font-weight: 700; }
  table { font-size: 26px; width: 100%; }
  th { background: #E6E1D6; }
  blockquote { border-left: 8px solid #B5530F; padding-left: 24px; color: #16181D; font-size: 34px; }
  section.dark { background: #16181D; color: #F4F1EA; }
  section.dark h1, section.dark h2 { color: #F4F1EA; }
  section.dark strong { color: #8FB0F2; }
  section.dark em { color: #F09A52; }
  section.accent { background: #2F5FC4; color: #F4F1EA; }
  section.accent h1 { color: #F4F1EA; }
  section.lead { justify-content: center; }
  section.lead h1 { font-size: 80px; line-height: 1.1; }
  .big { font-size: 60px; font-weight: 800; }
---

<!-- _class: lead dark -->
<!-- _paginate: false -->

PARAMETERS VS HYPERPARAMETERS

# What does a model learn, and what do *we* decide?

[Team names] · [Course name] · FCAI, Cairo University

<!--
Hook (0:00–0:20): "When a machine-learning model 'learns', what exactly is it learning? And which parts do we still have to decide? That's the difference between parameters and hyperparameters, and it's the key to training any model well."
-->

---

## Two kinds of numbers in every model

| | **Parameters** | *Hyperparameters* |
|---|---|---|
| Who sets them? | **The model**, during training | *Us*, before training |
| Learned from data? | ✅ Yes | ❌ No |
| Examples | weights, bias, slope, split rules | learning rate, steps, max depth, k |
| If wrong… | training fixes them | training goes badly |

<!--
(0:20–0:50) "Parameters are the numbers inside the model that it learns from the data during training, like the weights of a neural network, or the slope and intercept of a line. We never set them by hand. Hyperparameters are the settings we choose before training starts, like the learning rate, the number of training steps, or how deep a decision tree may grow. The model can't learn them from the data. We have to pick them, and a bad choice can ruin training."
-->

---

## The one-line test

> If **training changes it**, it's a **parameter**.
> If *you set it before training*, it's a *hyperparameter*.

📚 Think of a student preparing for an exam:
- **What they learn** from the textbook → **parameters**
- *How many hours they study, and how fast they go through each chapter* → *hyperparameters*

<!--
(0:50–1:10) "A simple test: if training changes it, it's a parameter. If you set it, it's a hyperparameter. Like a student: what they learn from the book is the parameters; the study plan they choose is the hyperparameters."
-->

---

## Examples across models

| Model | **Parameters** (learned) | *Hyperparameters* (chosen) |
|---|---|---|
| Linear regression | weights **w**, bias **b** | *learning rate, number of steps* |
| Decision tree | split rules (feature + threshold) | *max_depth, min_samples_leaf* |
| KNN | *(none: it stores the data)* | *k (number of neighbours)* |
| Neural network | millions of weights + biases | *layers, neurons, learning rate, batch size* |

<!--
(1:10–1:40) "Some examples. In linear regression, the parameters are the weights and the bias, and the hyperparameters are the learning rate and the number of steps. In a decision tree, the split rules are learned, but max depth is chosen. In KNN, the number of neighbours, k, is a hyperparameter. In a neural network, the weights are parameters, while the number of layers, the batch size and the learning rate are hyperparameters."
-->

---

## How do we choose hyperparameters?

Try several values and compare them on data the model **hasn't seen**.

| Method | Idea |
|---|---|
| *Grid search* | Try **every** combination from a list *(in our demo)* |
| *Random search* | Try **random** combinations: faster with many hyperparameters |
| *Cross-validation* | Split the training data into k parts and test on each in turn, so the comparison is **fair** |

<!--
(1:40–2:10) "So how do we choose hyperparameters well? We try several values and compare them on data the model hasn't seen. Grid search tries every combination, random search tries random ones, and cross-validation makes the comparison fair. Let's see both ideas live."
-->

---

<!-- _class: lead accent -->
<!-- _paginate: false -->

LIVE DEMO

# 🎛️ Param Lab

A Streamlit app built with numpy and scikit-learn

<!--
Demo (2:10–4:10): cut to the screen recording of the app here.
Part 1 (lr 0.1): steps 0 → 10 (w 4.5, b 35) → 100 (w 5.04, b 39.74; true 5 and 40). lr 0.001 → w only 0.91. lr 1.5 → exploded.
Part 2: max_depth 1 (test 65%) → 15 (train 100%, test 65%) → 4 (75%) → click GridSearchCV → picks 4.
Full narration: docs/video-script.md.
-->

---

<!-- _class: dark -->

## What the demo showed

**Parameters: learned, not typed**
<span class="big">**w = 5.04 · b = 39.74**</span>
Gradient descent found the real rule (5 × hours + 40) by itself in 100 steps.

*Hyperparameters: chosen by us*
<span class="big">*max_depth 4 → 75%*</span>
Depth 1 underfits (65%), depth 15 overfits (100% train, 65% test). GridSearchCV picked 4 on its own.

<!--
Optional summary right after the demo if you have spare seconds. Point at each big number once.
-->

---

## What a bad hyperparameter does

| Hyperparameter | Too small | Too big |
|---|---|---|
| *Learning rate* | 🐢 learns far too slowly | 💥 overshoots and explodes |
| *Training steps* | stops before it has learned | wastes time, may overfit |
| *Tree max_depth* | 📉 underfits: too simple | 📈 overfits: memorises the training data |
| *k in KNN* | overfits to single neighbours | underfits: everything averaged |

<!--
(4:10–4:35) "So, to recap. Parameters are learned from the data, like w and b, which our model found by itself. Hyperparameters are chosen by us before training, like the learning rate and the tree depth. Choose them badly and the model learns too slowly, explodes, underfits or overfits."
-->

---

<!-- _class: lead dark -->
<!-- _paginate: false -->

RECAP

# The model learns the **parameters**. *We tune the hyperparameters.*

Thank you · Questions?

<!--
(4:35–5:00) "Choose them well, with tools like grid search and cross-validation, and the same model performs at its best. The model learns the parameters. We tune the hyperparameters. Thank you."
-->

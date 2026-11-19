# Video script: Parameters vs. Hyperparameters (5:00)

Narration pace is about 130 words per minute. Split the parts between teammates however you like.
Read aloud, the narration takes about 4 minutes. Use the rest for pauses: let each change sit on screen for 2–3 seconds during the demo.
The slide numbers below match `slides/slides.md`.

---

## 1 · Hook (0:00–0:20) · slide 1
> "When a machine-learning model 'learns', what exactly is it learning?
> And which parts do *we* still have to decide?
> That's the difference between parameters and hyperparameters, and it's the key to training any model well."

## 2 · The two definitions (0:20–1:10) · slides 2–3
> "**Parameters** are the numbers inside the model that it learns from the data during training,
> like the weights of a neural network, or the slope and intercept of a line.
> We never set them by hand. The training algorithm finds them.
> **Hyperparameters** are the settings we choose *before* training starts,
> like the learning rate, the number of training steps, or how deep a decision tree may grow.
> The model can't learn them from the data. We have to pick them, and a bad choice can ruin training.
> A simple test: if training changes it, it's a parameter. If you set it, it's a hyperparameter."

## 3 · Examples and tuning (1:10–2:10) · slides 4–5
> "Some examples. In linear regression, the parameters are the weights and the bias, and the hyperparameters are the learning rate and the number of steps.
> In a decision tree, the split rules are learned, but max depth is chosen.
> In KNN, the number of neighbours, k, is a hyperparameter.
> In a neural network, the weights are parameters, while the number of layers, the batch size and the learning rate are hyperparameters.
> So how do we choose hyperparameters well? We try several values and compare them on data the model hasn't seen.
> Grid search tries every combination, random search tries random ones,
> and cross-validation makes the comparison fair. Let's see both ideas live."

## 4 · Demo (2:10–4:10) · slide 6, then the app
### Part 1: Parameters (2:10–3:10)
Start: learning rate **0.1**, training steps **0**.

| Click | Say |
|---|---|
| Point at the chart and the metrics | "100 students, study hours versus exam score. Our model is a line, score equals w times hours plus b. At the start, w and b are zero: the model knows nothing." |
| Steps slider to **10** | "Ten steps of gradient descent: the line moves up. w is 4.5 and b is 35." |
| Steps slider to **100** | "After 100 steps: w is **5.04** and b is **39.74**. We generated this data with 5 and 40, so the model *learned* the real rule by itself. Those are the parameters." |
| Learning rate **0.001** | "But the learning rate is ours to choose. Too small, and after 100 steps the line has barely moved: w is only 0.91." |
| Learning rate **1.5** | "Too big, and every step overshoots. The error 💥 explodes. Same data, same model: only the hyperparameter changed." |

### Part 2: Hyperparameters (3:10–4:10)
| Click | Say |
|---|---|
| Scroll down, max_depth at **1** | "Now a decision tree predicting pass or fail. With depth 1 it's too simple: only **65%** on new students. That's underfitting." |
| Slider to **15** | "Depth 15: **100%** on the training data, but still **65%** on new students. It memorised the training set, including the unlucky students. That's overfitting." |
| Slider to **4** | "Somewhere in the middle is better: depth 4 gives **75%**." |
| Click **Find the best max_depth** | "We don't have to guess. GridSearchCV tries every depth with 5-fold cross-validation and picks **depth 4** on its own, the top of the curve." |

## 5 · Recap (4:10–5:00) · slides 7–9
> "So, to recap.
> Parameters are learned from the data: like w and b, which our model found by itself.
> Hyperparameters are chosen by us before training: like the learning rate and the tree depth.
> Choose them badly and the model learns too slowly, explodes, underfits or overfits.
> Choose them well, with tools like grid search and cross-validation, and the same model performs at its best.
> The model learns the parameters. *We* tune the hyperparameters. Thank you."

---

## Recording checklist
- [ ] Run `streamlit run app.py` and click through everything once before recording, so every result is cached and appears instantly.
- [ ] Browser window at 1280×720 or larger, zoom 125% so the numbers are readable.
- [ ] Starting state: learning rate **0.1**, steps **0**, max_depth **1**. Refresh the page to reset the button.
- [ ] Record the demo with OBS or the Windows Snipping Tool (Win+Shift+R), then place it after slide 6 in your video editor.
- [ ] Time one full read-through. If it runs long, shorten part 3 first.

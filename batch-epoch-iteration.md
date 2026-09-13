Absolutely. The **three terms you're thinking of are most likely:**

1. **Epoch**
2. **Batch**
3. **Iteration** (also called a **step** in many ML frameworks)

These three are tightly connected, especially when you study **Gradient Descent, Neural Networks, and Deep Learning**.

I'll build this from **zero → intuition → mathematics → training loop → Batch GD vs SGD vs Mini-Batch GD → practical examples → common confusion**.

---

# 1. First: Why do we even need Epoch, Batch, and Iteration?

Before these terms, we need to understand what happens when a machine-learning model learns.

Imagine we have:

```text
1000 students' data
```

and we want to predict whether a student will pass.

Each student might have:

```text
Hours studied → 5
Attendance   → 80%
Assignments  → 7
Result       → Pass
```

So we have:

```text
1000 training examples
```

We give these examples to a model.

The model makes predictions.

Then we calculate **how wrong it was** using a **loss function**.

For example:

```text
Actual:     1
Prediction: 0.7

Error/loss → something > 0
```

Then we use **Gradient Descent** to adjust the model's parameters.

The basic learning cycle is:

```text
Training data
     ↓
Model makes predictions
     ↓
Calculate loss
     ↓
Calculate gradients
     ↓
Update parameters
     ↓
Repeat
```

But now a problem appears.

### What if we have 1,000,000 training examples?

Should we give **all 1,000,000 examples to the model at once**?

Or should we give them one at a time?

Or maybe groups of 32/64/128?

That's where:

> **Batch, Iteration, and Epoch**

become important.

---

# 2. First understand "training example"

Let's establish some terminology.

Suppose our dataset has:

```text
10,000 rows
```

Each row represents one training example.

For example:

| Student | Hours | Attendance | Result |
| ------- | ----: | ---------: | ------ |
| 1       |     2 |         60 | Fail   |
| 2       |     5 |         80 | Pass   |
| 3       |     7 |         90 | Pass   |
| ...     |   ... |        ... | ...    |
| 10000   |     4 |         75 | Pass   |

We can call:

```text
10,000 = number of training examples
```

Usually represented as:

$$
N = 10,000
$$

---

# 3. What is a Batch?

A **batch is a group of training examples processed together in one training step.**

Suppose we have:

```text
10,000 examples
```

and choose:

```text
batch size = 100
```

Then we don't necessarily process all 10,000 at once.

Instead:

```text
Batch 1 → examples 1–100
Batch 2 → examples 101–200
Batch 3 → examples 201–300
...
Batch 100 → examples 9901–10000
```

Each group of 100 is a **batch**.

So:

> **Batch = the number of training examples processed together before one parameter update.**

This definition is extremely important.

---

# 4. What is Batch Size?

The number of examples inside one batch is called the:

> **batch size**

Example:

```text
Batch 1:
[100 examples]
```

Then:

```text
batch size = 100
```

If:

```text
batch size = 32
```

then the model processes:

```text
32 examples
```

before performing the corresponding parameter update.

Common batch sizes include:

```text
16
32
64
128
256
512
```

These aren't magical numbers. They're commonly convenient choices, especially because of GPU/memory considerations.

---

# 5. Now: What is an Iteration?

An **iteration** generally means:

> **One batch is processed and the model performs one parameter update.**

For example:

```text
Batch 1
   ↓
Forward pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Update weights
```

That's **one iteration**.

Then:

```text
Batch 2
   ↓
Forward pass
   ↓
Loss
   ↓
Backpropagation
   ↓
Update weights
```

That's another iteration.

So:

```text
1 batch processed
        ↓
1 parameter update
        ↓
1 iteration
```

In modern deep-learning terminology, **step** is also commonly used for this parameter-update event.

---

# 6. What is an Epoch?

Now we reach the most important one.

An **epoch means the model has gone through the entire training dataset once.**

Suppose:

```text
Dataset = 10,000 examples
```

After the model has processed all 10,000 examples once:

```text
1 epoch completed
```

Then it can go through the same training dataset again:

```text
Epoch 1 → all 10,000 examples
Epoch 2 → all 10,000 examples
Epoch 3 → all 10,000 examples
...
```

So:

> **Epoch = one complete pass through the entire training dataset.**

---

# 7. Now connect all three

Suppose:

```text
Dataset = 1000 examples
Batch size = 100
```

We divide the dataset:

```text
1000 examples
       ↓
──────────────────────────
Batch 1 → 100
Batch 2 → 100
Batch 3 → 100
Batch 4 → 100
Batch 5 → 100
Batch 6 → 100
Batch 7 → 100
Batch 8 → 100
Batch 9 → 100
Batch 10 → 100
──────────────────────────
```

The model processes:

```text
Batch 1 → iteration 1
Batch 2 → iteration 2
Batch 3 → iteration 3
...
Batch 10 → iteration 10
```

After iteration 10:

> The entire dataset has been processed once.

Therefore:

> **1 epoch has completed.**

So:

```text
1 epoch
   =
10 iterations
   =
10 batches processed
   =
1000 training examples processed
```

---

# 8. The fundamental relationship

If:

$$
N = \text{number of training examples}
$$

and:

$$
B = \text{batch size}
$$

then approximately:

$$
\text{iterations per epoch} =
\frac{N}{B}
$$

More precisely, when the final batch can be smaller:

$$
\text{iterations per epoch}
=
\left\lceil\frac{N}{B}\right\rceil
$$

where \(\lceil\rceil\) means round up.

---

# 9. Example

Suppose:

```text
N = 1000
Batch size = 100
```

Then:

$$
\frac{1000}{100}=10
$$

So:

```text
10 iterations = 1 epoch
```

If we train for:

```text
5 epochs
```

then:

$$
5 \times 10 = 50
$$

So:

```text
50 iterations
```

and approximately:

```text
50 parameter updates
```

---

# 10. Another example where it doesn't divide perfectly

Suppose:

```text
N = 1050
Batch size = 100
```

Then:

```text
Batch 1 → 100
Batch 2 → 100
...
Batch 10 → 100
Batch 11 → 50
```

Therefore:

$$
\left\lceil\frac{1050}{100}\right\rceil = 11
$$

So:

```text
1 epoch = 11 iterations
```

The last batch contains only:

```text
50 examples
```

Depending on the framework/configuration, you can either process that smaller final batch or configure the loader to drop it.

---

# 11. Why don't we simply use the entire dataset?

Excellent question.

This takes us directly into **Gradient Descent**.

There are three important approaches:

### 1. Batch Gradient Descent

Use the **entire dataset** for one update.

### 2. Stochastic Gradient Descent (SGD)

Use **one training example** for one update.

### 3. Mini-Batch Gradient Descent

Use a **small group of examples** for one update.

This third approach is what you'll encounter most often in modern deep learning.

Let's understand all three properly.

---

# 12. Batch Gradient Descent

Suppose:

```text
1000 training examples
```

Batch Gradient Descent uses:

```text
all 1000 examples
```

to calculate the gradient.

Then:

```text
1000 examples
      ↓
calculate predictions
      ↓
calculate total/average loss
      ↓
calculate gradient
      ↓
UPDATE WEIGHTS
```

One update happens after all 1000 examples.

Therefore:

```text
Batch size = 1000
```

And:

```text
1 epoch = 1 iteration
```

because the entire dataset was one batch.

---

# 13. Stochastic Gradient Descent

Now go to the opposite extreme.

Instead of:

```text
1000 examples
```

we use:

```text
1 example
```

at a time.

For example:

```text
Example 1
   ↓
prediction
   ↓
loss
   ↓
gradient
   ↓
update weights

Example 2
   ↓
prediction
   ↓
loss
   ↓
gradient
   ↓
update weights

Example 3
   ↓
...
```

So:

```text
Batch size = 1
```

For 1000 examples:

```text
1 epoch = 1000 iterations
```

because each example causes an update.

---

# 14. Mini-Batch Gradient Descent

Now we choose a middle ground.

Suppose:

```text
1000 examples
Batch size = 100
```

Then:

```text
100 examples → update
100 examples → update
100 examples → update
...
```

So:

```text
10 iterations = 1 epoch
```

This is called:

> **Mini-Batch Gradient Descent**

And this is extremely common in neural-network/deep-learning training.

---

# 15. Visual comparison

Imagine:

```text
1000 training examples
```

### Batch Gradient Descent

```text
┌─────────────────────────────────────┐
│        ALL 1000 examples            │
└─────────────────────────────────────┘
                  ↓
             ONE UPDATE

1 epoch = 1 iteration
```

### SGD

```text
Example 1 → update
Example 2 → update
Example 3 → update
...
Example 1000 → update

1 epoch = 1000 iterations
```

### Mini-Batch

```text
100 examples → update
100 examples → update
100 examples → update
...
100 examples → update

1 epoch = 10 iterations
```

---

# 16. Why use mini-batches?

This is where the computational reason becomes important.

Imagine training a neural network on:

```text
1,000,000 images
```

Trying to put all 1,000,000 images into memory at once could be extremely expensive.

Instead:

```text
Batch size = 32
```

The model takes:

```text
32 images
 ↓
forward pass
 ↓
loss
 ↓
backpropagation
 ↓
update weights
```

Then another 32:

```text
32 images
 ↓
forward pass
 ↓
loss
 ↓
backpropagation
 ↓
update weights
```

and so on.

This makes training much more manageable and allows efficient use of GPUs.

---

# 17. But there's another important idea: batches are not just about memory

Mini-batches also give us a useful **estimate of the gradient**.

Suppose the true gradient based on the whole dataset is:

$$
\nabla J(w)
$$

Calculating it over every training example can be expensive.

A mini-batch gives us an estimate:

$$
\nabla J_{\text{batch}}(w)
$$

It won't necessarily be exactly the same as the full-dataset gradient.

That's actually okay.

This introduces some randomness/noise into the updates.

That noisy behavior can be useful for optimization.

---

# 18. Why does SGD look "noisy"?

Suppose the ideal gradient says:

```text
      ↓
      ↓
      ↓
      ↓
```

The model would smoothly move toward the minimum.

But with individual examples, each example can suggest a slightly different direction:

```text
        ↗
      ↘
       ↓
     ↙
       ↘
        ↓
```

So the path can look noisy.

Conceptually:

```text
Full Batch:

Start
  \
   \
    \
     \
      Minimum
```

SGD:

```text
Start
  \
   ↘
     ↙
       ↘
      ↙
        ↘
       Minimum
```

It's not necessarily moving directly toward the minimum every single update, but overall it can move toward a good solution.

---

# 19. Where does Gradient Descent fit into this?

Let's connect everything.

Suppose our model has a parameter:

$$
w
$$

We want to minimize the loss:

$$
J(w)
$$

Gradient descent uses:

$$
w_{\text{new}}
=
w_{\text{old}}
-
\eta \nabla J(w)
$$

where:

* \(w\) = model parameter/weight
* \(\eta\) = learning rate
* \(\nabla J(w)\) = gradient

The key question becomes:

> **Which examples do we use to calculate the gradient before updating \(w\)?**

That's exactly where the three methods differ.

---

# 20. Batch Gradient Descent

Use all training examples:

$$
\nabla J(w)
=
\frac{1}{N}
\sum_{i=1}^{N}
\nabla L_i(w)
$$

Then:

$$
w \leftarrow w-\eta\nabla J(w)
$$

One update after the entire dataset.

---

# 21. SGD

Use one example:

$$
w \leftarrow
w-\eta\nabla L_i(w)
$$

So:

```text
1 example
→ gradient
→ update
```

Then another:

```text
1 example
→ gradient
→ update
```

---

# 22. Mini-Batch

Suppose batch \(B\) contains \(m\) examples.

Calculate:

$$
\nabla J_B(w)
=
\frac{1}{m}
\sum_{i\in B}
\nabla L_i(w)
$$

Then:

$$
w \leftarrow w-\eta\nabla J_B(w)
$$

So the batch provides an estimate of the gradient, and the model updates its weights.

---

# 23. Now let's understand Epoch deeply

An epoch has **nothing inherently to do with updating weights**.

An epoch simply describes **how much of the dataset has been processed**.

Think of reading a textbook.

Suppose you have:

```text
100 pages
```

If you read:

```text
pages 1 → 100
```

you completed:

> **one pass through the book**

That's analogous to:

> **one epoch**

If you read it again:

```text
pages 1 → 100
```

that's:

> **second epoch**

So:

```text
Epoch = one complete pass through the training dataset
```

---

# 24. Why train for multiple epochs?

Because one pass through the dataset usually isn't enough for the model to learn good parameters.

Imagine:

```text
Initial weights
     ↓
Epoch 1
     ↓
better weights
     ↓
Epoch 2
     ↓
better weights
     ↓
Epoch 3
     ↓
better weights
```

The model repeatedly sees training data and adjusts its parameters.

For example:

```text
Epoch 1 → loss = 0.90
Epoch 2 → loss = 0.65
Epoch 3 → loss = 0.48
Epoch 4 → loss = 0.35
Epoch 5 → loss = 0.27
```

Ideally, training loss decreases.

But **more epochs are not automatically better**.

Eventually the model can start **overfitting**.

---

# 25. Epoch vs Iteration — the biggest confusion

Students often confuse:

```text
Epoch
```

with:

```text
Iteration
```

Remember this:

### Iteration

> One batch → one update → generally one iteration/step.

### Epoch

> Entire training dataset → one complete pass.

So:

```text
DATASET
│
├── Batch 1 → Iteration 1
├── Batch 2 → Iteration 2
├── Batch 3 → Iteration 3
├── Batch 4 → Iteration 4
└── Batch 5 → Iteration 5
                         ↓
                    1 EPOCH
```

---

# 26. Let's do a complete numerical example

Suppose:

```text
Training examples = 5000
Batch size = 100
Epochs = 20
```

### Step 1: Number of batches

$$
5000 / 100 = 50
$$

So:

```text
50 batches per epoch
```

### Step 2: Iterations per epoch

Each batch produces one update.

Therefore:

```text
50 iterations per epoch
```

### Step 3: Total iterations

$$
50 \times 20 = 1000
$$

Therefore:

```text
1000 total iterations
```

### Step 4: Number of parameter updates

Approximately:

```text
1000 updates
```

So:

```text
5000 examples
       ↓
batch size = 100
       ↓
50 iterations
       ↓
1 epoch
       ↓
20 epochs
       ↓
1000 iterations/updates
```

---

# 27. A very important formula

If:

$$
N = \text{number of training examples}
$$

$$
B = \text{batch size}
$$

$$
E = \text{number of epochs}
$$

then:

### Iterations per epoch

$$
\boxed{
\left\lceil\frac{N}{B}\right\rceil
}
$$

### Total iterations

$$
\boxed{
E \times
\left\lceil\frac{N}{B}\right\rceil
}
$$

assuming each batch is processed and each batch corresponds to one update.

---

# 28. Example with 60,000 images

Suppose you're training a neural network on:

```text
60,000 images
```

and:

```text
batch size = 32
epochs = 10
```

Iterations per epoch:

$$
\left\lceil\frac{60000}{32}\right\rceil
=1875
$$

Therefore:

```text
1 epoch = 1875 iterations
```

For 10 epochs:

$$
1875\times10=18,750
$$

So roughly:

```text
18,750 parameter updates
```

---

# 29. What happens inside one iteration?

This is another critical concept.

Suppose:

```text
Batch size = 32
```

One iteration might look like:

```text
32 training examples
        ↓
   Forward Pass
        ↓
   Predictions
        ↓
    Calculate Loss
        ↓
  Backpropagation
        ↓
 Calculate Gradients
        ↓
 Update Weights
```

Then:

```text
Next 32 examples
```

and repeat.

---

# 30. Forward pass

The model receives the batch.

For example:

```text
32 images
```

The neural network processes them:

```text
Input
 ↓
Layer 1
 ↓
Layer 2
 ↓
Layer 3
 ↓
Output
```

This is the:

> **forward pass**

It produces predictions.

---

# 31. Loss calculation

Suppose actual labels are:

```text
[cat, dog, dog, cat, ...]
```

and model predicts probabilities.

The loss function tells us:

> "How bad were these predictions?"

For example:

```text
Loss = 0.72
```

Lower generally means better for the chosen loss.

---

# 32. Backpropagation

Now we ask:

> "Which weights contributed to the error, and in what direction should they change?"

Backpropagation calculates gradients using the chain rule.

Conceptually:

```text
Loss
 ↓
gradient
 ↓
weight adjustments
```

---

# 33. Weight update

Gradient descent then changes the weights.

Simplified:

$$
w_{\text{new}}
=
w_{\text{old}}
-
\text{learning rate}
\times
\text{gradient}
$$

After that:

```text
Iteration finished
```

Then the next batch comes.

---

# 34. Important: batch ≠ iteration

They are related but technically different.

### Batch

A **group of examples**.

Example:

```text
32 images
```

### Iteration

A **training step/update associated with processing one batch**.

Example:

```text
Process 32 images
→ calculate gradients
→ update weights
→ one iteration
```

So don't say:

> "A batch is an iteration."

Better:

> **One iteration typically processes one batch.**

---

# 35. What does `batch_size` actually control?

It controls:

> **How many training examples are used for each gradient update.**

Small batch:

```text
batch_size = 8
```

Large number of updates per epoch:

```text
N / 8
```

Large batch:

```text
batch_size = 512
```

Fewer updates per epoch:

```text
N / 512
```

---

# 36. Small batch vs large batch

### Small batch

Example:

```text
batch size = 16
```

Advantages:

* less memory required
* more frequent updates
* gradient estimates have more noise
* can sometimes generalize well

Disadvantages:

* less efficient hardware utilization in some situations
* training can be noisier
* potentially more steps needed

---

### Large batch

Example:

```text
batch size = 512
```

Advantages:

* fewer updates per epoch
* can make efficient use of parallel hardware
* gradient estimate can be more stable

Disadvantages:

* more memory required
* each update can be computationally larger
* very large batches can have optimization/generalization trade-offs

There isn't one universally best batch size.

---

# 37. Now here's an important subtlety

You may hear:

> "SGD uses batch size 1."

That's the classical definition.

But in modern ML libraries, people sometimes casually use "SGD" to refer to the **general stochastic/mini-batch optimization approach**, even when `batch_size` is greater than 1.

So don't get trapped by terminology.

Strictly:

```text
Batch GD
→ entire dataset

SGD
→ one example

Mini-Batch GD
→ subset of examples
```

In practical deep learning, optimizers such as `SGD` are very commonly used with mini-batches.

For example:

```python
optimizer = SGD(...)
batch_size = 32
```

The optimizer being named SGD doesn't mean the batch size must be 1.

---

# 38. What does shuffling have to do with epochs?

Usually, training data is **shuffled between epochs**.

Suppose:

```text
Epoch 1:

1 2 3 4 5 6 7 8
```

Next epoch could become:

```text
5 1 8 3 6 2 7 4
```

Why?

Because if your data always appears in the exact same order, especially if the data is sorted/grouped in some meaningful way, training can be less effective.

Shuffling helps provide varied mini-batches.

---

# 39. Important distinction: training data vs validation/test data

When we say:

> "One epoch means the entire dataset has been processed"

we normally mean the **training dataset** in the context of parameter learning.

For example:

```text
Training set
     ↓
Used to update weights

Validation set
     ↓
Used to evaluate/tune during development

Test set
     ↓
Used for final evaluation
```

Validation/test data should not be used to update the model's weights during ordinary supervised training.

---

# 40. Epoch and validation

Suppose:

```text
Epoch 1
   ↓
Training
   ↓
Validation
   ↓

Epoch 2
   ↓
Training
   ↓
Validation
```

You might see:

```text
Epoch 1:
training loss   = 0.60
validation loss = 0.65

Epoch 2:
training loss   = 0.45
validation loss = 0.50

Epoch 3:
training loss   = 0.32
validation loss = 0.38

...

Epoch 10:
training loss   = 0.05
validation loss = 0.72
```

Now we might suspect:

> **Overfitting**

because training performance keeps improving while validation performance gets worse.

This is one reason we monitor epochs.

---

# 41. Early stopping

Instead of saying:

```text
Train for 100 epochs no matter what
```

we can monitor validation performance.

For example:

```text
Epoch 1 → validation loss 0.60
Epoch 2 → 0.50
Epoch 3 → 0.42
Epoch 4 → 0.38
Epoch 5 → 0.35
Epoch 6 → 0.36
Epoch 7 → 0.40
```

The model was best around epoch 5.

We might stop training rather than continuing indefinitely.

This is called:

> **Early stopping**

---

# 42. Epoch is not a measure of "learning"

This is subtle but important.

Don't think:

> "1 epoch = model learned once."

That's not technically correct.

An epoch simply means:

> **The training algorithm has processed the training dataset once.**

Whether the model learned well depends on:

* learning rate
* model architecture
* batch size
* optimizer
* data quality
* loss function
* initialization
* regularization
* etc.

---

# 43. Another mental model: classroom

Imagine you are a teacher teaching a student.

You have:

```text
100 questions
```

### Batch size = 10

You give the student:

```text
10 questions
```

Then check mistakes and adjust their understanding.

That's one:

> **iteration**

Then another 10:

```text
10 questions
```

Another iteration.

After all 100 questions:

> **1 epoch**

If you repeat all 100 questions five times:

> **5 epochs**

So:

```text
100 questions
│
├── 10 → iteration 1
├── 10 → iteration 2
├── 10 → iteration 3
├── ...
└── 10 → iteration 10
                     ↓
                  1 epoch

Repeat × 5

= 5 epochs
```

---

# 44. One more term you may encounter: Step

You were probably thinking of **iteration**, but there is another closely related word:

> **Step**

In many deep-learning frameworks:

```text
iteration ≈ training step
```

For example:

```text
Batch 1 → Step 1
Batch 2 → Step 2
Batch 3 → Step 3
```

Each step normally corresponds to a parameter update.

So you'll often see:

```text
global step
training step
iteration
```

used in closely related ways.

---

# 45. One more term: sample

You'll also hear:

> **sample**

A sample generally means one individual example/data point.

For example:

```text
Dataset:
1000 samples
```

If:

```text
batch size = 32
```

then:

```text
32 samples = one batch
```

---

# 46. One more term: batch size

Let's put all the vocabulary together:

```text
Sample
   ↓
one training example

Batch
   ↓
group of samples

Batch size
   ↓
number of samples in the batch

Iteration / Step
   ↓
one batch is processed → usually one update

Epoch
   ↓
entire training dataset processed once
```

This hierarchy is worth memorizing.

---

# 47. Complete picture

Suppose:

```text
Dataset = 10,000 samples
Batch size = 100
Epochs = 5
```

Then:

```text
                 TRAINING DATA
                10,000 samples
                      │
                      ↓
              divide into batches
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
    Batch 1        Batch 2        Batch 3 ...
     100             100            100
       ↓              ↓              ↓
 Iteration 1      Iteration 2    Iteration 3
       ↓              ↓              ↓
    Update          Update         Update
       │              │              │
       └──────────────┴──────────────┘
                      ↓
                 100 iterations
                      ↓
                   1 EPOCH
                      ↓
              Repeat 4 more times
                      ↓
                  5 EPOCHS
```

Total:

$$
100 \times 5 = 500
$$

So:

```text
500 iterations
≈ 500 parameter updates
```

---

# 48. Now connect this to neural networks

Suppose you're training a neural network:

```text
Input layer
     ↓
Hidden layer
     ↓
Hidden layer
     ↓
Output layer
```

You have:

```text
50,000 training images
```

Choose:

```text
batch size = 64
epochs = 20
```

Training roughly works like:

```text
                    Epoch 1
                       ↓
       ┌─────────────────────────────┐
       │                             │
Batch 1 → forward → loss → backward → update
Batch 2 → forward → loss → backward → update
Batch 3 → forward → loss → backward → update
...
Batch 782 → forward → loss → backward → update
       │                             │
       └─────────────────────────────┘
                       ↓
                  Epoch complete
                       ↓
                    Epoch 2
                       ↓
                      ...
                       ↓
                   Epoch 20
```

Because:

$$
\lceil50000/64\rceil = 782
$$

So approximately:

```text
782 iterations per epoch
```

and:

$$
782 \times 20 = 15,640
$$

approximately:

```text
15,640 training steps/updates
```

---

# 49. A common misconception

### ❌ Wrong:

> "Batch is how many times the model sees the dataset."

No.

### Correct:

> **Batch is a group of training examples processed together.**

---

### ❌ Wrong:

> "Iteration means one complete pass through the dataset."

No.

### Correct:

> **An iteration/step is typically one batch's processing followed by one parameter update.**

---

### ❌ Wrong:

> "Epoch means one weight update."

No.

### Correct:

> **Epoch means one complete pass through the training dataset.**

---

# 50. The cheat sheet

| Term           | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| **Sample**     | One training example                                   |
| **Batch**      | Group of training examples                             |
| **Batch size** | Number of examples in one batch                        |
| **Iteration**  | One batch processed and typically one parameter update |
| **Step**       | Common synonym for an iteration/update                 |
| **Epoch**      | One complete pass through the training dataset         |

---

# 51. The relationship you should remember

The easiest way to remember everything is:

$$
\boxed{\text{Samples} \rightarrow \text{Batches} \rightarrow \text{Iterations} \rightarrow \text{Epochs}}
$$

More specifically:

```text
Training dataset
      ↓
Divide into batches
      ↓
Process one batch
      ↓
One iteration / step
      ↓
Process all batches
      ↓
One epoch
      ↓
Repeat for multiple epochs
```

And the key formula:

$$
\boxed{
\text{Iterations per epoch}
=
\left\lceil
\frac{\text{Number of training samples}}
{\text{Batch size}}
\right\rceil
}
$$

---

## Finally, connect this with your Gradient Descent knowledge

This is probably the most useful big-picture picture:

```text
                 GRADIENT DESCENT
                       │
           How should weights change?
                       │
                       ↓
              Calculate gradients
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
     Batch GD         SGD       Mini-Batch GD
     all data       1 sample      some data
          │            │            │
          ↓            ↓            ↓
       1 update      1 update     1 update
          │            │            │
          ↓            ↓            ↓
      iteration      iteration     iteration
          │            │            │
          └────────────┼────────────┘
                       ↓
             Eventually process
             entire training set
                       ↓
                    1 EPOCH
                       ↓
                 repeat multiple
                    epochs
```

So **epoch, batch, and iteration aren't three separate machine-learning algorithms**. They are terms that describe **how training data moves through the learning process**.

And when you later study **CNNs, RNNs, backpropagation, optimizers like SGD/Adam, learning-rate schedules, and PyTorch/TensorFlow training loops**, these three terms will keep appearing.

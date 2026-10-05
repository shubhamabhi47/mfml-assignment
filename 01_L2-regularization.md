## Building from the Ground Up

### Step 0: What is a neural network doing?

A neural network is a function with **knobs**. The knobs are called **weights** (and biases). The network takes an input (say, a house description) and produces an output (predicted price).

$$\text{output} = f(\text{input};\; w_1, w_2, \dots, w_n)$$

where $w_1, \dots, w_n$ are the weights (the knobs). There might be millions of them.

### Step 1: How does it learn?

You have a dataset: (house A → price $300k), (house B → price $500k), etc.

The network makes a prediction for each house. You measure how wrong it is with a **loss function**:

$$L(w_1, \dots, w_n) = \frac{1}{N}\sum_{i=1}^{N} \big(f(\text{house}_i;\; w) - \text{actual price}_i\big)^2$$

This is just a **number** that tells you "how bad the current knobs are."

**Training = turning the knobs to make $L$ as small as possible.**

### Step 2: The problem — memorization (overfitting)

You have 1000 training houses. The network has 1,000,000 knobs.

With 1,000,000 knobs and only 1000 data points, the network can make $L = 0$ **exactly** — it can fit every single quirk. It learns:

- "House 37 had a scratch on the wall → price was $295k"
- "House 82 had a blue door → price was $410k"

These are **noise**, not real patterns. On a new house, the network will fail because the new house doesn't have that exact scratch.

This is **overfitting**: the knobs are set to memorize the training data rather than learn the actual rules (square footage matters, number of bedrooms matters, the color of the door does NOT matter).

### Step 3: The fix — L2 Regularization (the "penalty tax")

We change the loss function to:

$$L_{\text{new}}(w) = L(w) + \alpha \sum_{i=1}^{n} w_i^2$$

The new term $\alpha \sum w_i^2$ is the **penalty tax**. It says:

> "For every weight you keep large, you pay a tax. The tax grows with the **square** of the weight."

- If $w_i = 0$: tax = 0 (free)
- If $w_i = 1$: tax = $\alpha$
- If $w_i = 10$: tax = $100\alpha$ (expensive!)

So the network now has a **tradeoff**:

- "Should I keep this weight large to fit the training data better?"
- "But if I keep it large, I pay a big tax."

The network will only keep a weight large if the **benefit** (reduced error on real patterns) outweighs the **cost** (the tax). Useless weights get shrunk toward zero.

**That's all L2 regularization is.** Add $\alpha \sum w_i^2$ to the loss. Done.

### Step 4: Now, what does the Hessian have to do with this?

Recall from our earlier conversation: the **Taylor expansion** of the loss near a minimum looks like:

$$L(w) \approx L(w^*) + \underbrace{\nabla L^T (w - w^*)}_{\text{linear term (zero at minimum)}} + \frac{1}{2}(w-w^*)^T H\,(w-w^*)$$

At a minimum, the linear term vanishes (gradient = 0). So near the minimum:

$$L(w) \approx L(w^*) + \frac{1}{2}(w-w^*)^T H\,(w-w^*)$$

The Hessian $H$ is the matrix of second derivatives. It tells you **how steep the landscape is in each direction**.

### Step 5: What do the eigenvalues of $H$ mean?

Every symmetric matrix $H$ can be decomposed into **eigenvectors** (directions) and **eigenvalues** (steepness in that direction):

$$H = Q \Lambda Q^T$$

where $\Lambda = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$.

Each eigenvalue $\lambda_i$ tells you: **if you move in the direction of eigenvector $q_i$, the loss curves upward with curvature $\lambda_i$.**

- **Large $\lambda_i$** → the loss shoots up fast in that direction → **steep canyon wall** → the weight in that direction is **important** (changing it hurts a lot)
- **Small $\lambda_i$** → the loss barely changes in that direction → **flat plain** → the weight in that direction is **unimportant** (you can change it freely without hurting)

**Concrete example:**

$$H = \begin{pmatrix} 1000 & 0 \\ 0 & 0.001 \end{pmatrix}$$

- Direction $q_1 = (1, 0)$: curvature = 1000 → **canyon** → weight $w_1$ is critical
- Direction $q_2 = (0, 1)$: curvature = 0.001 → **flat plain** → weight $w_2$ is useless

### Step 6: Why does L2 regularization produce the scaling factor $\frac{\lambda_i}{\lambda_i + \alpha}$?

Here's the key derivation. Near the minimum, the **regularized** loss in the eigen-direction $q_i$ looks like:

$$L_i(w_i) = \frac{1}{2}\lambda_i\, w_i^2 + \alpha\, w_i^2$$

Wait, that's not quite right. Let me be more careful.

The **unregularized** loss near the minimum, in the $i$-th eigen-direction, is:

$$L_i = \frac{1}{2}\lambda_i\, w_i^2$$

(with the minimum at $w_i = 0$ for simplicity).

Now add the L2 penalty:

$$L_i^{\text{reg}} = \frac{1}{2}\lambda_i\, w_i^2 + \alpha\, w_i^2 = \frac{1}{2}(\lambda_i + 2\alpha)\, w_i^2$$

Hmm, but that just changes the curvature — it doesn't shift the minimum. The minimum is still at $w_i = 0$.

**The real story is slightly different.** The minimum of the *unregularized* loss is NOT at $w_i = 0$ in general. Let's say the unregularized minimum is at $w_i^*$. Then near that point:

$$L_i \approx L(w^*) + \frac{1}{2}\lambda_i (w_i - w_i^*)^2$$

Now add the L2 penalty (measuring from zero, not from the minimum):

$$L_i^{\text{reg}} = \frac{1}{2}\lambda_i (w_i - w_i^*)^2 + \alpha\, w_i^2$$

To find the new minimum, take the derivative and set to zero:

$$\frac{d}{dw_i}\left[\frac{1}{2}\lambda_i (w_i - w_i^*)^2 + \alpha\, w_i^2\right] = 0$$

$$\lambda_i (w_i - w_i^*) + 2\alpha\, w_i = 0$$

$$\lambda_i\, w_i - \lambda_i\, w_i^* + 2\alpha\, w_i = 0$$

$$(\lambda_i + 2\alpha)\, w_i = \lambda_i\, w_i^*$$

$$\boxed{w_i^{\text{new}} = \frac{\lambda_i}{\lambda_i + 2\alpha}\; w_i^*}$$

(If we absorb the factor of 2 into $\alpha$, this becomes $\frac{\lambda_i}{\lambda_i + \alpha} w_i^*$, which is the form you saw.)

### Step 7: Reading the result

The new weight is the old weight **multiplied by a shrinkage factor**:

$$w_i^{\text{new}} = \underbrace{\frac{\lambda_i}{\lambda_i + \alpha}}_{\text{shrinkage factor}} \cdot w_i^*$$

| Situation | $\lambda_i$ | Factor $\frac{\lambda_i}{\lambda_i+\alpha}$ | What happens |
|---|---|---|---|
| Steep canyon (important weight) | 1000 | $\frac{1000}{1000.1} \approx 0.9999$ | Weight barely changes |
| Flat plain (useless weight) | 0.001 | $\frac{0.001}{0.101} \approx 0.0099$ | Weight crushed to ~1% of original |

**The Hessian automatically tells the regularization which weights to keep and which to shrink.** Important directions (high curvature) are protected; unimportant directions (low curvature) get squashed.

### Step 8: The full picture in one flow

```
Neural network has weights w
        ↓
Loss L(w) measures error on training data
        ↓
Without regularization: network memorizes noise (overfitting)
        ↓
Add L2 penalty: L_reg = L(w) + α||w||²
        ↓
Near the minimum, L looks quadratic: ½wᵀHw
        ↓
H has eigenvalues λ₁, ..., λₙ (curvatures in principal directions)
        ↓
Regularization shrinks each direction by factor λᵢ/(λᵢ+α)
        ↓
Steep directions (large λᵢ) → factor ≈ 1 → weight preserved
Flat directions (small λᵢ)   → factor ≈ 0 → weight crushed
        ↓
Result: network keeps only the weights that matter
```

### Step 9: Why "L2" specifically?

The penalty is $\alpha \sum w_i^2 = \alpha \|w\|_2^2$. The "2" refers to the **L2 norm** (Euclidean norm) of the weight vector. You could also use:

- **L1**: $\alpha \sum |w_i|$ → pushes weights **exactly** to zero (sparsity)
- **L2**: $\alpha \sum w_i^2$ → pushes weights **toward** zero but rarely exactly zero (shrinkage)

L2 is smoother and easier to optimize, which is why it's the default.

---

**One-sentence summary:** L2 regularization adds a tax on large weights; the Hessian tells you which directions are steep (important) and which are flat (unimportant); the result is that important weights are preserved and unimportant weights are automatically shrunk toward zero.

# L2 Regularization Shrinkage — From the Ground Up

I'll build this up from zero. By the end, you'll see *why* the factor $\frac{\lambda_i}{\lambda_i + 2\alpha}$ appears, not just *that* it does.

---

## Part 1: What is a loss function?

You have data. You want to find a weight $w$ that makes your predictions accurate. The **loss function** $L(w)$ measures "how bad" a given $w$ is:

$$L(w) = \text{some measure of error}$$

**Minimizing** $L(w)$ means: find the $w$ where the error is smallest.

### Concrete 1D example

Suppose your data says the best weight is $w^* = 5$. Near that point, the loss looks like a **bowl** (a parabola):

$$L(w) \approx L(w^*) + \frac{1}{2}\lambda\,(w - w^*)^2$$

- $L(w^*)$ is the minimum error (a constant, doesn't matter for finding the minimum).
- $\frac{1}{2}\lambda\,(w - w^*)^2$ is a parabola centered at $w = w^*$.
- $\lambda > 0$ is the **curvature** — how "steep" or "narrow" the bowl is.

**Key insight:** A **large** $\lambda$ means a **narrow, steep** bowl (the data is very confident about $w^*$). A **small** $\lambda$ means a **wide, flat** bowl (the data is uncertain — many values of $w$ give similar loss).

```
Steep bowl (large λ):        Flat bowl (small λ):

  L                              L
  |  /\                          |
  | /  \                         |___________
  |/    \                         /         \
  |      \                       /           \
  |_______\____ w              _______________ w
       w*                           w*
```

---

## Part 2: What does $\lambda$ (the eigenvalue) actually mean?

In 1D, $\lambda$ is just the second derivative $L''(w)$. In **multiple dimensions**, the loss is a **bowl in 3D, 4D, ...** and different directions have different curvatures.

The **Hessian matrix** $H$ (the matrix of second derivatives) captures all the curvatures. Its **eigenvalues** $\lambda_1, \lambda_2, \ldots$ tell you:

- Along the **$i$-th eigen-direction** $q_i$, the loss curves with curvature $\lambda_i$.
- Large $\lambda_i$ → the bowl is **narrow** in that direction (data is confident).
- Small $\lambda_i$ → the bowl is **flat** in that direction (data is uncertain).

**Why eigen-directions matter:** By rotating our coordinate system to align with the eigenvectors, the loss **decouples** into independent 1D problems. In the $i$-th eigen-direction, the problem is exactly like the 1D bowl above.

So the entire multi-dimensional problem reduces to: **solve a 1D problem in each eigen-direction independently.**

---

## Part 3: The unregularized problem (no penalty)

In the $i$-th eigen-direction, the loss near the minimum is:

$$L_i(w_i) = L(w^*) + \frac{1}{2}\lambda_i\,(w_i - w_i^*)^2$$

The minimum is at $w_i = w_i^*$ (just take derivative, set to zero):

$$\frac{dL_i}{dw_i} = \lambda_i\,(w_i - w_i^*) = 0 \implies w_i = w_i^*$$

This is the "normal" answer. No surprises.

---

## Part 4: What does L2 regularization ADD?

L2 regularization says: **"I don't trust large weights. Penalize them."**

The penalty is:

$$\text{Penalty} = \alpha\, w_i^2$$

where $\alpha > 0$ is a hyperparameter you choose (the **regularization strength**).

**Critical detail:** This penalty is measured from **zero**, not from $w_i^*$. It says "I want $w_i$ to be close to **0**." This is what creates the tension.

### The regularized loss

Now the total loss in the $i$-th direction is:

$$L_i^{\text{reg}}(w_i) = \underbrace{\frac{1}{2}\lambda_i\,(w_i - w_i^*)^2}_{\text{data: wants } w_i = w_i^*} + \underbrace{\alpha\, w_i^2}_{\text{penalty: wants } w_i = 0}$$

**Two competing forces:**
- The data term is a bowl centered at $w_i^*$.
- The penalty is a bowl centered at $0$.

The regularized minimum is where these two bowls **compromise**.

---

## Part 5: The derivation, step by step

### Step 5a: Write out the full expression

$$L_i^{\text{reg}}(w_i) = \frac{1}{2}\lambda_i\,(w_i - w_i^*)^2 + \alpha\, w_i^2$$

Let me expand the first term so you can see what's happening:

$$= \frac{1}{2}\lambda_i\,(w_i^2 - 2w_i\,w_i^* + (w_i^*)^2) + \alpha\, w_i^2$$

$$= \frac{1}{2}\lambda_i\, w_i^2 - \lambda_i\, w_i\, w_i^* + \frac{1}{2}\lambda_i\,(w_i^*)^2 + \alpha\, w_i^2$$

Group the $w_i^2$ terms:

$$= \left(\frac{1}{2}\lambda_i + \alpha\right) w_i^2 - \lambda_i\, w_i^*\, w_i + \frac{1}{2}\lambda_i\,(w_i^*)^2$$

### Step 5b: Take the derivative

$$\frac{dL_i^{\text{reg}}}{dw_i} = (\lambda_i + 2\alpha)\,w_i - \lambda_i\, w_i^*$$

Let me show you where each piece comes from:
- Derivative of $\left(\frac{1}{2}\lambda_i + \alpha\right) w_i^2$ is $(\lambda_i + 2\alpha)\,w_i$
- Derivative of $-\lambda_i\, w_i^*\, w_i$ is $-\lambda_i\, w_i^*$
- Derivative of the constant $\frac{1}{2}\lambda_i\,(w_i^*)^2$ is $0$

### Step 5c: Set derivative to zero (find the minimum)

$$(\lambda_i + 2\alpha)\,w_i - \lambda_i\, w_i^* = 0$$

### Step 5d: Solve for $w_i$

$$(\lambda_i + 2\alpha)\,w_i = \lambda_i\, w_i^*$$

$$\boxed{w_i^{\text{reg}} = \frac{\lambda_i}{\lambda_i + 2\alpha}\; w_i^*}$$

**That's the shrinkage factor.** The regularized answer is the unregularized answer, multiplied by a number less than 1.

---

## Part 6: WHY is the factor what it is? (The deep "why")

Let me re-derive this in a way that makes the *meaning* obvious.

Think of it as a **tug-of-war** between two springs:

| Spring | Centered at | Stiffness | Pulls $w_i$ toward |
|---|---|---|---|
| Data (loss) | $w_i^*$ | $\lambda_i$ | $w_i^*$ |
| Penalty (L2) | $0$ | $2\alpha$ | $0$ |

When you have two springs pulling a mass, the equilibrium position is the **stiffness-weighted average**:

$$w_i^{\text{reg}} = \frac{\lambda_i \cdot w_i^* + 2\alpha \cdot 0}{\lambda_i + 2\alpha} = \frac{\lambda_i}{\lambda_i + 2\alpha}\, w_i^*$$

**This is the same formula as a weighted average where the weights are the curvatures.**

- If the data spring is much stiffer ($\lambda_i \gg 2\alpha$): the answer is close to $w_i^*$ (barely shrunk).
- If the penalty spring is much stiffer ($2\alpha \gg \lambda_i$): the answer is close to $0$ (heavily shrunk).

---

## Part 7: Concrete numerical example

Suppose $w_i^* = 4$, $\lambda_i = 2$, $\alpha = 1$.

**Unregularized:** $w_i = 4$

**Regularized:**
$$w_i^{\text{reg}} = \frac{2}{2 + 2(1)} \times 4 = \frac{2}{4} \times 4 = 2$$

The weight shrank from 4 to 2 — a factor of $\frac{1}{2}$.

Now try a **flat** direction: $\lambda_i = 0.1$, same $\alpha = 1$:

$$w_i^{\text{reg}} = \frac{0.1}{0.1 + 2} \times 4 = \frac{0.1}{2.1} \times 4 \approx 0.19$$

The weight shrank from 4 to **0.19** — almost zero!

### Why the difference?

- In the first case, the data is **confident** (steep bowl, $\lambda_i = 2$). The data's "pull" toward 4 is strong, so the penalty can only push it partway.
- In the second case, the data is **uncertain** (flat bowl, $\lambda_i = 0.1$). The data barely cares whether $w_i$ is 4 or 0.1 or 3. The penalty wins easily and drags the weight to near zero.

---

## Part 8: Summary table

| $\lambda_i$ vs $2\alpha$ | Factor $\frac{\lambda_i}{\lambda_i+2\alpha}$ | What happens | Why |
|---|---|---|---|
| $\lambda_i \gg 2\alpha$ | $\approx 1$ | Almost no shrinkage | Data is very confident in this direction |
| $\lambda_i \approx 2\alpha$ | $\approx \frac{1}{2}$ | 50% shrinkage | Data and penalty are equally strong |
| $\lambda_i \ll 2\alpha$ | $\approx \frac{\lambda_i}{2\alpha} \to 0$ | Heavy shrinkage toward 0 | Data is uncertain; penalty dominates |

---

## Part 9: The big picture (why this matters)

**Overfitting** happens when the model memorizes noise. Noise lives in the **flat directions** of the loss (small $\lambda_i$) — those are directions where the training data doesn't strongly constrain the answer, so the model can take large values there without hurting training loss.

L2 regularization **selectively crushes the flat directions** while leaving the steep directions mostly alone. This is exactly what you want: keep the signal (steep directions), remove the noise (flat directions).

That's the entire story in one sentence: **L2 adds a uniform curvature $2\alpha$ to every direction, and the shrinkage factor is just the ratio of original curvature to total curvature.**



It's **not an assumption** — it's a mathematical fact that follows from **Taylor's theorem**. Let me show you exactly why.

## Taylor expansion around the minimum

For **any** smooth function $L(w)$, you can write:

$$L(w) = L(w^*) + L'(w^*)(w - w^*) + \frac{1}{2}L''(w^*)(w - w^*)^2 + \cdots$$

This is just the Taylor series of $L$ evaluated at $w^*$. It's always true (for sufficiently smooth functions).

Now, $w^*$ is the **minimum**, so:

1. $L'(w^*) = 0$ ← (first derivative is zero at a minimum)
2. $L''(w^*) = \lambda > 0$ ← (second derivative is positive at a minimum — that's what makes it a bowl, not a hill)

Plug those in:

$$L(w) = L(w^*) + 0 \cdot (w - w^*) + \frac{1}{2}\lambda\,(w - w^*)^2 + \cdots$$

$$\boxed{L(w) \approx L(w^*) + \frac{1}{2}\lambda\,(w - w^*)^2}$$

The $\cdots$ are higher-order terms ($(w-w^*)^3$, $(w-w^*)^4$, etc.) which are **negligible when $w$ is close to $w^*$**.

## So it's an *approximation*, not an assumption

| Aspect | Detail |
|---|---|
| **Exact?** | No — only the first two terms are kept |
| **When is it good?** | When $w$ is **close** to $w^*$ (the "near the minimum" part) |
| **Why is it valid?** | Taylor's theorem — works for *any* smooth function, not just quadratic ones |
| **What is $\lambda$?** | It's literally $L''(w^*)$ — the second derivative at the minimum |

## In multiple dimensions (the eigen-direction version)

Same idea, but now $L$ is a function of many variables. The Taylor expansion around $w^*$ gives:

$$L(\mathbf{w}) \approx L(\mathbf{w}^*) + \frac{1}{2}(\mathbf{w} - \mathbf{w}^*)^T H\,(\mathbf{w} - \mathbf{w}^*)$$

where $H$ is the **Hessian** (matrix of second derivatives). When you rotate to eigen-coordinates, $H$ becomes diagonal with entries $\lambda_1, \lambda_2, \ldots$, and the problem decouples into independent 1D bowls — each with curvature $\lambda_i$.

## The one case where it's *exactly* true

If the loss is **already quadratic** (e.g., linear regression with MSE loss), then the Taylor expansion has no higher-order terms and the formula is **exact**, not an approximation. For neural networks the loss is non-quadratic, so it's an approximation — but it's a *very good* one near the minimum, which is all we care about.

Great question. The Hessian $H$ **didn't disappear** — it got **diagonalized** into the $\lambda_i$'s. Here's the exact step:

## The missing step: eigen-decomposition

$H$ is a symmetric matrix. By the **spectral theorem**, you can always write:

$$H = Q\,\Lambda\,Q^T$$

where:
- $Q$ is an **orthogonal** matrix (columns = eigenvectors $q_1, q_2, \ldots$)
- $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots)$ is a **diagonal** matrix (entries = eigenvalues)

## Now substitute into the quadratic form

The loss near the minimum is:

$$L \approx L(w^*) + \frac{1}{2}(\mathbf{w} - \mathbf{w}^*)^T H\,(\mathbf{w} - \mathbf{w}^*)$$

**Define new coordinates:** let $\mathbf{z} = Q^T(\mathbf{w} - \mathbf{w}^*)$, i.e., $z_i$ is the component of $(\mathbf{w} - \mathbf{w}^*)$ along the $i$-th eigenvector.

Then $\mathbf{w} - \mathbf{w}^* = Q\mathbf{z}$, and:

$$(\mathbf{w}-\mathbf{w}^*)^T H\,(\mathbf{w}-\mathbf{w}^*) = (Q\mathbf{z})^T (Q\Lambda Q^T)(Q\mathbf{z})$$

Since $Q^T Q = I$ (orthogonal matrix):

$$= \mathbf{z}^T Q^T Q \Lambda Q^T Q \mathbf{z} = \mathbf{z}^T \Lambda\, \mathbf{z} = \sum_i \lambda_i\, z_i^2$$

**The cross terms vanish.** That's the whole point of rotating to eigen-coordinates.

## So the full picture is:

$$L \approx L(w^*) + \frac{1}{2}\sum_i \lambda_i\, z_i^2$$

This is a **sum of independent 1D parabolas** — one per eigen-direction. Each has curvature $\lambda_i$.

## In short

| Before rotation | After rotation |
|---|---|
| $\frac{1}{2}(\mathbf{w}-\mathbf{w}^*)^T H\,(\mathbf{w}-\mathbf{w}^*)$ | $\frac{1}{2}\sum_i \lambda_i\, z_i^2$ |
| One coupled multi-D bowl | Independent 1D bowls |
| $H$ (full matrix) | $\lambda_1, \lambda_2, \ldots$ (diagonal entries) |

So $H$ **became** the $\lambda_i$'s. The $\lambda_i$ in the 1D formula is just the $i$-th diagonal entry of $\Lambda$, which is the $i$-th eigenvalue of $H$. Nothing was lost — it was just a change of coordinates that decoupled the problem.





Okay. Forget everything I said before. Let's start from **zero** with just numbers.

---

## Part 1: The problem (overfitting)

You want to predict house prices. Your model is:

$$\text{price} = w_1 \times \text{sqft} + w_2 \times \text{door color} + w_3 \times \text{bedrooms}$$

After training, the model finds:

- $w_1 = 500$ (sqft matters a lot → good)
- $w_2 = 3000$ (door color → **this is nonsense**, it just memorized the training data)
- $w_3 = 200$ (bedrooms matter → good)

The model is **overfitting**: it learned "blue door → expensive" because in your 100 training houses, 3 happened to have blue doors and were expensive. That's coincidence, not a real rule.

---

## Part 2: The fix (L2 regularization)

We add a rule: **"You pay a tax on every weight you use."**

The tax on weight $w$ is: $\alpha \times w^2$

Let's say $\alpha = 10$ (the tax rate). Now the model is minimizing:

$$\text{error on data} + 10(w_1^2 + w_2^2 + w_3^2)$$

The model now has to **balance** two things:
- Make the error small (fit the data)
- Keep the weights small (pay less tax)

**What happens with actual numbers:**

| Weight | Without tax | With tax ($\alpha=10$) | Why? |
|---|---|---|---|
| $w_1$ (sqft) | 500 | ~480 | Big weight, but the data **demands** it. Error would explode if you shrink it. So it mostly survives. |
| $w_2$ (door) | 3000 | ~50 | Huge weight, but the data **doesn't demand** it. The tax is way cheaper than the tiny error reduction it gives. So it gets crushed. |
| $w_3$ (beds) | 200 | ~180 | Medium weight, data needs it somewhat. Slight shrinkage. |

**That's L2 regularization. That's the whole thing.** Big weights that the data actually needs survive. Big weights that are just noise get crushed.

---

## Part 3: Why does the data "demand" some weights but not others?

This is where the **Hessian** comes in. But forget that word for now.

The question is: *"If I shrink $w_1$ from 500 to 480, how much does the error go up?"*

- For $w_1$ (sqft): error goes up **a lot**. The data really needs this weight. → The landscape is **steep** here.
- For $w_2$ (door): error goes up **almost nothing**. The data doesn't care. → The landscape is **flat** here.

The Hessian is just a **table of "how much does the error go up if I nudge each weight?"** for every pair of weights.

The **eigenvalues** $\lambda_i$ are the "steepness" in the most important directions:

- Large $\lambda_i$ → steep → the weight matters → error explodes if you change it
- Small $\lambda_i$ → flat → the weight doesn't matter → you can change it freely

---

## Part 4: The final formula (now you can read it)

After L2 regularization, each weight gets multiplied by:

$$\frac{\lambda_i}{\lambda_i + \alpha}$$

where:
- $\lambda_i$ = how steep the landscape is in direction $i$ (how much the data needs that weight)
- $\alpha$ = your tax rate

**Two cases:**

**Steep (important weight):** $\lambda_i = 1000$, $\alpha = 10$

$$\frac{1000}{1000 + 10} = \frac{1000}{1010} \approx 0.99$$

Weight keeps 99% of its value. Barely affected. ✓

**Flat (useless weight):** $\lambda_i = 0.1$, $\alpha = 10$

$$\frac{0.1}{0.1 + 10} = \frac{0.1}{10.1} \approx 0.01$$

Weight keeps 1% of its value. Crushed. ✓

---

## The whole story in 3 sentences

1. **L2 adds a tax** on large weights so the model stops memorizing noise.
2. **The Hessian tells you** which weights the data actually needs (steep) vs. which it doesn't (flat).
3. **The result:** needed weights survive (~99%), unneeded weights get crushed (~1%). The model automatically keeps only what matters.


# Three Deep-Dive Questions

---

## 1. How do eigenvalues affect gradient descent convergence?

### Setup

Recall: near the minimum, the loss is $\frac{1}{2}\sum_i \lambda_i\, z_i^2$ in eigen-coordinates. Gradient descent updates each coordinate independently:

$$z_i^{(k+1)} = z_i^{(k)} - \eta\,\lambda_i\, z_i^{(k)} = (1 - \eta\,\lambda_i)\, z_i^{(k)}$$

where $\eta$ is the learning rate.

### The convergence rate per direction

After $k$ steps, the error in direction $i$ is:

$$z_i^{(k)} = (1 - \eta\,\lambda_i)^k\, z_i^{(0)}$$

For this to **decay** (not blow up), we need:

$$|1 - \eta\,\lambda_i| < 1 \quad \forall\, i$$

This gives the constraint:

$$0 < \eta < \frac{2}{\lambda_{\max}}$$

### The bottleneck: the smallest eigenvalue

The **slowest** direction to converge is the one with the **smallest** $\lambda_i$, because $(1 - \eta\,\lambda_{\min})$ is closest to 1 (least decay per step).

With the optimal learning rate $\eta = \frac{2}{\lambda_{\max} + \lambda_{\min}}$, the worst-case convergence rate per step is:

$$\boxed{\text{rate} = \frac{\kappa - 1}{\kappa + 1}}$$

where $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ is the **condition number**.

### What this means concretely

| $\kappa$ | Rate per step | Steps to reduce error by $10^{-6}$ |
|---|---|---|
| 1 (perfectly round bowl) | 0 (one step!) | 1 |
| 10 | 0.818 | ~72 |
| 100 | 0.980 | ~700 |
| 1000 | 0.998 | ~7000 |
| 10000 | 0.9998 | ~70 000 |

**Geometric picture — the "zigzag":**

```
   Steep direction (λ_max):  fast oscillation
   Flat direction (λ_min):   slow progress

   Start
     \
      \  ← big step in steep direction
       \
        \  ← overshoots, comes back
         \
          \  ← only tiny progress in flat direction
           \
            \  ← ...repeats thousands of times
             \
              → minimum
```

The gradient always points "downhill" — mostly along the steep direction. Each step makes big progress in the steep direction but **tiny** progress in the flat direction. The result is a zigzag that takes forever to slide along the narrow valley.

### Why L2 regularization helps here

L2 adds $\alpha$ to every eigenvalue: $\lambda_i \to \lambda_i + 2\alpha$. This:
- Doesn't change $\lambda_{\max}$ much (if $\lambda_{\max}$ is already large)
- **Lifts** $\lambda_{\min}$ significantly (if it was near zero)
- **Reduces** $\kappa$ dramatically

So regularization literally makes the valley wider, reducing the zigzag.

---

## 2. What if the Hessian is not diagonalizable?

### Short answer: It can't happen (for standard loss functions).

The Hessian is the matrix of second partial derivatives:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i\,\partial x_j}$$

By **Clairaut's theorem** (equality of mixed partials), if $f$ is twice continuously differentiable ($C^2$):

$$\frac{\partial^2 f}{\partial x_i\,\partial x_j} = \frac{\partial^2 f}{\partial x_j\,\partial x_i}$$

This means $H$ is **symmetric**: $H = H^T$.

By the **spectral theorem**, every real symmetric matrix is **always** diagonalizable by an orthogonal matrix. There are no exceptions.

### So when *could* you have a non-diagonalizable matrix?

| Scenario | Why it's not a problem |
|---|---|
| $f$ is not $C^2$ (e.g., ReLU at 0) | The Hessian doesn't exist at that point. You use subgradients or smoothing. |
| You're looking at a **Jacobian** (not Hessian) of a system | Jacobians can be non-symmetric and non-diagonalizable. Use Arnoldi iteration instead of Lanczos. |
| You approximate the Hessian (e.g., Gauss-Newton $J^T J$) | $J^T J$ is always symmetric PSD, hence diagonalizable. |
| Numerical errors make a symmetric matrix *slightly* non-symmetric | Use $(H + H^T)/2$ to symmetrize. The eigenvalue problem is well-conditioned for symmetric matrices. |

### What if the Hessian is singular (zero eigenvalues)?

This *can* happen (e.g., at a saddle point or a flat direction). It's not "non-diagonalizable" — it's still diagonalizable, just with some $\lambda_i = 0$. Consequences:

- $\kappa = \infty$ → gradient descent never converges in that direction
- Newton's method fails (can't invert $H$)
- **Fix:** Add L2 regularization ($\lambda_i \to \lambda_i + 2\alpha > 0$), or use quasi-Newton methods (L-BFGS) that build an invertible approximation

---

## 3. How are eigenvalues computed for large matrices?

### The problem

For a neural network with $n = 10^8$ parameters, the Hessian is $10^8 \times 10^8$. You can't even **store** it (that's $10^{16}$ numbers ≈ 80 exabytes). And you definitely can't run $O(n^3)$ eigensolvers.

### Key insight: you don't need all eigenvalues

For understanding convergence, regularization effects, etc., you mainly care about:
- The **extremal** eigenvalues ($\lambda_{\max}$, $\lambda_{\min}$) → for condition number
- Maybe a few eigenvalues near a specific value → for spectral analysis

### The Lanczos algorithm (for symmetric matrices)

This is the standard tool. It's a **Krylov subspace** method:

**Idea:** Instead of working with the full $n \times n$ matrix, build up a small $m \times m$ tridiagonal matrix $T$ (where $m \ll n$) whose eigenvalues approximate the extreme eigenvalues of $H$.

**Algorithm (simplified):**

1. Start with a random unit vector $v_1$.
2. For $k = 1, 2, \ldots, m$:
   - Compute $w_k = H\, v_k$ (a **matrix-vector product** — no need to form $H$ explicitly!)
   - Orthogonalize against previous vectors (three-term recurrence for symmetric case)
   - Get new vector $v_{k+1}$ and a tridiagonal entry
3. After $m$ steps, you have a tridiagonal matrix $T_m$ (size $m \times m$, with $m \ll n$).
4. The eigenvalues of $T_m$ approximate the extreme eigenvalues of $H$.

**Why it works:** The Krylov subspace $\text{span}\{v_1, Hv_1, H^2v_1, \ldots, H^{m-1}v_1\}$ captures the "most important" directions — those corresponding to extreme eigenvalues.

**Cost:** Each step costs $O(nnz(H))$ (number of non-zeros in $H$) for the matrix-vector product. Total: $O(m \cdot nnz(H))$. For sparse $H$ (e.g., from a CNN), $nnz(H) \ll n^2$.

### For neural networks specifically

You can't even form $H$ explicitly. But you **can** compute $Hv$ (Hessian-vector product) using the **Hessian-free** trick:

$$Hv = \lim_{\epsilon \to 0} \frac{\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)}{2\epsilon}$$

This requires only **two forward + backward passes** (same cost as computing a gradient). So Lanczos can run "Hessian-free":

```
Lanczos iteration (Hessian-free):
  for k in 1..m:
      w = Hessian_vector_product(v_k)   # 2 backprop passes
      # three-term recurrence...
```

### Summary of methods by matrix size

| Matrix size $n$ | Method | Cost |
|---|---|---|
| $n < 1000$ | Direct (QR, divide & conquer) | $O(n^3)$ |
| $n = 10^4 - 10^6$, sparse | **Lanczos** (symmetric) / Arnoldi (non-symmetric) | $O(m \cdot nnz)$, $m \ll n$ |
| $n = 10^8$ (neural nets) | **Hessian-free Lanczos** + random probing | $O(m)$ backprop passes |
| $n = 10^{10}+$ (physics, PDEs) | Lanczos with shift-invert, multigrid preconditioning | Problem-dependent |

### One more practical trick: Hutchinson's estimator

If you just need **traces** or **eigenvalue statistics** (not individual eigenvalues), you can use random vectors:

$$\text{tr}(H) \approx \frac{1}{K}\sum_{k=1}^{K} v_k^T H\, v_k, \quad v_k \sim \mathcal{N}(0, I)$$

This gives you the **sum of all eigenvalues** (and similarly, $\text{tr}(H^2)$ gives the sum of squared eigenvalues, etc.) using only $K$ Hessian-vector products. This is how tools like `hessian-free` estimators in PyTorch/JAX work.



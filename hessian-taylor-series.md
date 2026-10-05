# Taylor Series & the Hessian: From the Ground Up

---

## 1. The Core Problem

You have a function $f(x)$ — say $e^x$, $\sin x$, or some complicated physical law — and you want to **approximate it with something simpler** (a polynomial) near a point $a$. Why? Because polynomials are the only functions you can compute with just addition, multiplication, and constants. Every numerical method on a computer ultimately reduces to polynomial evaluation.

The question: **what is the "best" polynomial of degree $n$ that approximates $f$ near $a$?**

---

## 2. Order 0: The Constant Approximation

The simplest possible approximation is a constant. The **best** constant to use near $a$ is clearly the value of the function itself:

$$f(x) \approx f(a)$$

This is a horizontal line at height $f(a)$. It's only good if you're *very* close to $a$.

---

## 3. Order 1: The Tangent Line (Linear Approximation)

A constant can't capture the **slope** of the function. So we add a linear term:

$$f(x) \approx f(a) + c_1(x - a)$$

What should $c_1$ be? We want this polynomial to not just match the *value* of $f$ at $a$, but also its **rate of change** at $a$. That means:

$$\frac{d}{dx}\bigg[f(a) + c_1(x-a)\bigg]_{x=a} = c_1 = f'(a)$$

So:

$$\boxed{f(x) \approx f(a) + f'(a)(x - a)}$$

This is the **tangent line**. It matches $f(a)$ and $f'(a)$. The error near $a$ is $O((x-a)^2)$ — it vanishes quadratically.

**Where does this come from?** It's literally the definition of the derivative. The derivative $f'(a)$ is defined as:

$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

Rearranging: $f(a+h) = f(a) + f'(a)h + \text{something smaller than } h$. That "something smaller" is the error, and it's $O(h^2)$.

---

## 4. Order 2: Adding Curvature

The tangent line is straight — it can't capture **curvature** (concavity). The second derivative $f''(a)$ measures exactly that. So we add a quadratic term:

$$f(x) \approx f(a) + f'(a)(x-a) + c_2(x-a)^2$$

Now we demand that the **second derivative** of our polynomial also matches $f''(a)$:

$$\frac{d^2}{dx^2}\bigg[f(a) + f'(a)(x-a) + c_2(x-a)^2\bigg]_{x=a} = 2c_2 = f''(a)$$

So $c_2 = \dfrac{f''(a)}{2!}$.

$$\boxed{f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2}$$

The error is now $O((x-a)^3)$.

**Why the $2!$?** Because when you differentiate $c_2(x-a)^2$ twice, you get $2c_2$. The factorial in the denominator is the "un-differentiation" of the power.

---

## 5. Order 3, 4, ..., $n$: The Pattern

Continuing this logic — match the $k$-th derivative at $a$ — you get:

$$\boxed{f(x) \approx \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!}(x-a)^k}$$

This is the **$n$-th degree Taylor polynomial** $T_n(x)$ centered at $a$.

**The recipe is mechanical:**
1. Compute $f(a), f'(a), f''(a), \ldots, f^{(n)}(a)$
2. Divide the $k$-th derivative by $k!$
3. Multiply by $(x-a)^k$
4. Sum from $k=0$ to $n$

If $a = 0$, it's called a **Maclaurin series** (a special case).

---

## 6. Rigorous Derivation (Two Paths)

### Path A: "Matching Derivatives" (Constructive)

Suppose $f$ is a power series: $f(x) = \sum_{k=0}^{\infty} c_k (x-a)^k$. Then:

- $f(a) = c_0$
- $f'(a) = c_1$
- $f''(a) = 2c_2 \Rightarrow c_2 = f''(a)/2!$
- $f^{(k)}(a) = k! \cdot c_k \Rightarrow c_k = f^{(k)}(a)/k!$

So **if** $f$ is representable as a power series, the coefficients are *forced* to be $f^{(k)}(a)/k!$. Taylor's theorem then proves that for sufficiently smooth functions, this polynomial approximation actually works (the error goes to zero as $n \to \infty$ under appropriate conditions).

### Path B: Iterated Fundamental Theorem of Calculus (Analytic)

Start with the FTC:

$$f(x) = f(a) + \int_a^x f'(t_1)\, dt_1$$

Now apply the FTC to $f'$:

$$f'(t_1) = f'(a) + \int_a^{t_1} f''(t_2)\, dt_2$$

Substitute back:

$$f(x) = f(a) + f'(a)(x-a) + \int_a^x \int_a^{t_1} f''(t_2)\, dt_2\, dt_1$$

Repeat for $f''$:

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \int_a^x \int_a^{t_1} \int_a^{t_2} f'''(t_3)\, dt_3\, dt_2\, dt_1$$

After $n$ steps, the remainder is an $(n+1)$-fold integral of $f^{(n+1)}$. Each nested integral contributes a factor of $(x-a)/k$, giving the factorial in the denominator.

---

## 7. The Remainder: How Good Is the Approximation?

The **remainder** (error) after $n$ terms is:

$$R_n(x) = f(x) - T_n(x)$$

**Lagrange form:** There exists some $c$ between $a$ and $x$ such that:

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$$

**Key bound:** If $|f^{(n+1)}(t)| \leq M$ for all $t$ between $a$ and $x$, then:

$$|R_n(x)| \leq \frac{M}{(n+1)!}|x-a|^{n+1}$$

**Convergence:** The Taylor series converges to $f(x)$ if and only if $\lim_{n\to\infty} R_n(x) = 0$.

- For $e^x$, $M = e^{|x|}$ is bounded on any finite interval, and $(n+1)!$ grows faster than any exponential → converges everywhere.
- For $\sin x$, $M = 1$ always → converges everywhere.
- For $\frac{1}{1-x}$ at $a=0$: converges only for $|x| < 1$ (radius of convergence = distance to the nearest singularity).
- **Pathological case:** $f(x) = e^{-1/x^2}$ for $x \neq 0$, $f(0) = 0$. All derivatives at $0$ are zero, so the Taylor series is identically $0$, but $f$ is not zero. The function is $C^\infty$ but **not analytic** at $0$.

---

## 8. Historical Context

- **Brook Taylor** (1715) published the general formula in *Methodus Incrementorum Directa*.
- **Maclaurin** (1742) studied the special case $a=0$ in detail.
- **Newton** had been using such expansions for decades before Taylor (to compute roots, etc.), but never published the general theorem.
- The connection to **analytic functions** (functions equal to their Taylor series in a neighborhood) was developed by Cauchy and others in the 19th century.

---

## 9. Going Multivariable: The Setup

Now let $f: \mathbb{R}^n \to \mathbb{R}$ be a scalar-valued function of $n$ variables, and we want to approximate $f$ near a point $\mathbf{a} \in \mathbb{R}^n$.

Let $\mathbf{x} - \mathbf{a} = \mathbf{h}$ (the displacement vector).

### Order 0:
$$f(\mathbf{x}) \approx f(\mathbf{a})$$

### Order 1: The Gradient

The first derivative of a multivariable function is the **gradient**:

$$\nabla f(\mathbf{a}) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}\bigg|_{\mathbf{x}=\mathbf{a}}$$

The first-order Taylor approximation is:

$$\boxed{f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T (\mathbf{x} - \mathbf{a})}$$

In 2D: $f(x,y) \approx f(a,b) + f_x(a,b)(x-a) + f_y(a,b)(y-b)$.

This is the **tangent plane** — the multivariable analog of the tangent line.

---

## 10. Order 2: The Hessian Appears

To capture **curvature** in multiple variables, we need all second partial derivatives. There are $n^2$ of them (though by **Clairaut's theorem**, if $f$ is $C^2$, then $f_{x_i x_j} = f_{x_j x_i}$, so the matrix is symmetric).

### The Hessian Matrix

$$\boxed{H_f(\mathbf{a}) = \nabla^2 f(\mathbf{a}) = \begin{pmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{pmatrix}\bigg|_{\mathbf{x}=\mathbf{a}}}$$

The $(i,j)$-th entry is $\dfrac{\partial^2 f}{\partial x_i \partial x_j}\bigg|_{\mathbf{a}}$.

It's **symmetric** ($H = H^T$) when $f \in C^2$ (Clairaut/Schwarz theorem: mixed partials commute).

### The Second-Order Taylor Formula

$$\boxed{f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^T(\mathbf{x}-\mathbf{a}) + \frac{1}{2}(\mathbf{x}-\mathbf{a})^T H_f(\mathbf{a})\,(\mathbf{x}-\mathbf{a})}$$

The last term is a **quadratic form**. In 2D it expands to:

$$\frac{1}{2}\bigg[f_{xx}(a,b)(x-a)^2 + 2f_{xy}(a,b)(x-a)(y-b) + f_{yy}(a,b)(y-b)^2\bigg]$$

Notice the factor of $2$ on the cross term — it comes from the fact that both $H_{12}$ and $H_{21}$ contribute (and they're equal).

---

## 11. Derivation of the Multivariable Formula (The "Restriction" Trick)

The cleanest derivation: **reduce to one variable by restricting to a line**.

Define a one-variable function along the line from $\mathbf{a}$ to $\mathbf{x}$:

$$g(t) = f\big(\mathbf{a} + t(\mathbf{x} - \mathbf{a})\big), \quad t \in [0, 1]$$

Now $g(0) = f(\mathbf{a})$ and $g(1) = f(\mathbf{x})$. Apply the **one-variable** Taylor theorem to $g$ at $t = 0$:

$$g(1) = g(0) + g'(0) + \frac{1}{2}g''(0) + \cdots$$

Now compute the derivatives using the **chain rule**:

**First derivative:**
$$g'(t) = \sum_{i=1}^n \frac{\partial f}{\partial x_i}\big(\mathbf{a}+t\mathbf{h}\big) \cdot h_i = \nabla f(\mathbf{a}+t\mathbf{h})^T \mathbf{h}$$

At $t=0$: $g'(0) = \nabla f(\mathbf{a})^T \mathbf{h}$.

**Second derivative:**
$$g''(t) = \sum_{i,j=1}^n h_i \frac{\partial^2 f}{\partial x_i \partial x_j}\big(\mathbf{a}+t\mathbf{h}\big) h_j = \mathbf{h}^T H_f(\mathbf{a}+t\mathbf{h})\, \mathbf{h}$$

At $t=0$: $g''(0) = \mathbf{h}^T H_f(\mathbf{a})\, \mathbf{h}$.

Substituting back:

$$f(\mathbf{x}) = f(\mathbf{a}) + \nabla f(\mathbf{a})^T \mathbf{h} + \frac{1}{2}\mathbf{h}^T H_f(\mathbf{a})\,\mathbf{h} + O(\|\mathbf{h}\|^3)$$

This is the full second-order multivariable Taylor expansion. **The Hessian appears naturally as the second derivative of the restricted one-variable function.**

---

## 12. What the Hessian Actually Tells You

The quadratic form $\mathbf{h}^T H \mathbf{h}$ encodes the **local curvature** of $f$ in all directions simultaneously.

### Eigenvalue Interpretation

Since $H$ is real and symmetric, it has real eigenvalues $\lambda_1, \ldots, \lambda_n$ and an orthonormal eigenbasis. In the eigenbasis, the quadratic term becomes:

$$\frac{1}{2}(\lambda_1 u_1^2 + \lambda_2 u_2^2 + \cdots + \lambda_n u_n^2)$$

where $u_i$ are coordinates along the eigenvectors. So:

| Eigenvalue condition | Shape near $\mathbf{a}$ | Classification |
|---|---|---|
| All $\lambda_i > 0$ | Bowl (upward) | Local **minimum** |
| All $\lambda_i < 0$ | Dome (downward) | Local **maximum** |
| Mixed signs | Saddle | **Saddle point** |
| Some $\lambda_i = 0$ | Flat in some direction | Inconclusive (degenerate) |

This is the **second-derivative test** generalized to $n$ variables.

### The 2D Special Case

For $f: \mathbb{R}^2 \to \mathbb{R}$:

$$H = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{xy} & f_{yy} \end{pmatrix}$$

The determinant $D = f_{xx}f_{yy} - f_{xy}^2$ gives:
- $D > 0, f_{xx} > 0$ → local min
- $D > 0, f_{xx} < 0$ → local max
- $D < 0$ → saddle
- $D = 0$ → test fails

---

## 13. Higher-Order Terms (Beyond the Hessian)

The third-order term involves all **third partial derivatives** — a $3$-way tensor (not a matrix). In $n$ variables, the $k$-th order term involves a **rank-$k$ symmetric tensor** of partial derivatives:

$$T_k = \frac{1}{k!} \sum_{i_1, \ldots, i_k} \frac{\partial^k f}{\partial x_{i_1} \cdots \partial x_{i_k}}(\mathbf{a}) \, h_{i_1} \cdots h_{i_k}$$

There is no "third-order Hessian matrix" — it's a tensor. The Hessian is special because order 2 is the highest order that can be represented as a matrix.

### The Exponential Operator Form

There's a beautiful compact way to write the full infinite series:

$$f(\mathbf{a} + \mathbf{h}) = e^{\mathbf{h} \cdot \nabla} f(\mathbf{a}) = \sum_{k=0}^{\infty} \frac{1}{k!}\big(\mathbf{h} \cdot \nabla\big)^k f(\mathbf{a})$$

where $\mathbf{h} \cdot \nabla = h_1 \frac{\partial}{\partial x_1} + h_2 \frac{\partial}{\partial x_2} + \cdots$ is the **directional derivative operator**. This looks like the Taylor series of $e^t$ applied to the operator $\mathbf{h}\cdot\nabla$.

---

## 14. Applications

### Newton's Method (Optimization)

To find a minimum of $f$, set $\nabla f = 0$. Newton's method uses the second-order Taylor approximation and sets its gradient to zero:

$$\nabla f(\mathbf{x}_k) + H_f(\mathbf{x}_k)(\mathbf{x}_{k+1} - \mathbf{x}_k) = 0$$

$$\boxed{\mathbf{x}_{k+1} = \mathbf{x}_k - H_f(\mathbf{x}_k)^{-1} \nabla f(\mathbf{x}_k)}$$

This is the multivariable analog of $x_{k+1} = x_k - f(x_k)/f'(x_k)$.

**Why it's powerful:** If $H$ is positive definite, the quadratic model has a unique minimum, and Newton's method converges **quadratically** (the number of correct digits roughly doubles each step).

**Why it's expensive:** Computing $H$ requires $O(n^2)$ second derivatives, and inverting it costs $O(n^3)$. For large $n$ (e.g., deep learning with millions of parameters), this is infeasible → leads to approximations like **L-BFGS**, **Gauss-Newton**, **natural gradient**, etc.

### Machine Learning

- The Hessian of the loss function tells you the **curvature of the loss landscape**.
- **Hessian-free methods** (e.g., conjugate gradient with Hessian-vector products) avoid forming $H$ explicitly.
- **Second-order optimization** (e.g., K-FAC, SHALOM) uses approximate Hessians for better convergence.

### Physics & Mechanics

- In Lagrangian mechanics, the Hessian of the potential energy $V(\mathbf{q})$ at an equilibrium gives the **normal mode frequencies**: $\omega_i^2 = \lambda_i / m_i$ (eigenvalues of the mass-weighted Hessian).
- In general relativity, the Hessian of the metric appears in the Riemann curvature tensor.

### Statistics

- The **Fisher information matrix** is the negative expected Hessian of the log-likelihood.
- In Bayesian inference, the Hessian of the negative log-posterior at the MAP estimate gives the **covariance** of the Laplace approximation.

---

## 15. Summary: The Hierarchy

| Order | 1D object | Multivariable object | Role in Taylor |
|-------|-----------|---------------------|----------------|
| 0 | $f(a)$ (scalar) | $f(\mathbf{a})$ (scalar) | Value |
| 1 | $f'(a)$ (scalar) | $\nabla f$ (vector, $n \times 1$) | Slope / direction of steepest ascent |
| 2 | $f''(a)$ (scalar) | $H_f$ (matrix, $n \times n$) | Curvature in all directions |
| 3 | $f'''(a)$ (scalar) | Rank-3 tensor ($n \times n \times n$) | Rate of change of curvature |
| $k$ | $f^{(k)}(a)$ (scalar) | Rank-$k$ symmetric tensor | $k$-th order shape |

Each order adds one more "layer of shape information." The Hessian is the first object that is genuinely **multidimensional** in its structure — it's not just a number, it's a matrix that encodes how the function curves in every possible direction simultaneously.

---

## 16. Common Pitfalls & Subtleties

1. **The Hessian is symmetric only if $f \in C^2$** (continuous second partials). If $f$ is merely twice differentiable, mixed partials may not commute and $H$ need not be symmetric.

2. **Positive semi-definite ≠ minimum.** If $H$ is positive semi-definite (some eigenvalues zero), the second-order test is **inconclusive** — you need higher-order terms. Example: $f(x,y) = x^2 + y^4$ has $H = \text{diag}(2, 0)$ at the origin (a minimum), but $f(x,y) = x^2 - y^4$ has the same Hessian (a saddle).

3. **The Taylor series may not converge to the function.** As noted, $C^\infty$ does not imply analytic. You need the remainder to actually go to zero.

4. **The "1/2" in the quadratic term is not optional.** It comes from the $2!$ in the Taylor formula. Forgetting it is a very common error.

5. **In optimization, you often only need Hessian-vector products**, not the full matrix. This is because Newton's step $H^{-1}g$ can be computed via conjugate gradient using only the ability to compute $H\mathbf{v}$ for any $\mathbf{v}$, which costs $O(n)$ extra work per iteration (via reverse-mode autodiff) instead of $O(n^2)$ to form $H$.


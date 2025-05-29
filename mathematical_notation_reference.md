# Mathematical Notation Reference for Code Documentation

*A comprehensive guide to typing elegant mathematical symbols in your code comments and documentation*

## Why Use Mathematical Notation?

Mathematical notation in code comments creates:
- **Clarity**: Direct connection between implementation and theory
- **Professionalism**: Academic-quality documentation
- **Readability**: Concise expression of complex concepts
- **Maintainability**: Future developers understand the math immediately

## Essential Greek Letters

| Symbol | Name | Alt Code | Usage Example |
|--------|------|----------|---------------|
| ∇ | Nabla (Gradient) | Alt + 8711 | `# Compute ∇J(θ) for policy gradient` |
| π | Pi | Alt + 227 | `# Sample action from π(a\|s)` |
| θ | Theta | Alt + 952 | `# Update parameters θ ← θ + α∇J(θ)` |
| δ | Delta | Alt + 235 | `# One-hot vector δₐ for action a` |
| λ | Lambda | Alt + 955 | `# Eligibility traces with λ decay` |
| α | Alpha | Alt + 945 | `# Learning rate α = 0.001` |
| γ | Gamma | Alt + 947 | `# Discount factor γ = 0.99` |
| β | Beta | Alt + 946 | `# Momentum parameter β` |
| ε | Epsilon | Alt + 949 | `# Exploration rate ε-greedy` |
| μ | Mu | Alt + 956 | `# Mean μ of distribution` |
| σ | Sigma | Alt + 963 | `# Standard deviation σ` |
| τ | Tau | Alt + 964 | `# Time constant τ` |
| ω | Omega | Alt + 969 | `# Angular frequency ω` |

## Mathematical Operators

| Symbol | Name | Alt Code | Usage Example |
|--------|------|----------|---------------|
| ≈ | Approximately | Alt + 8776 | `# Q(s,a) ≈ r + γ max Q(s',a')` |
| ≤ | Less than or equal | Alt + 8804 | `# Constraint: 0 ≤ probability ≤ 1` |
| ≥ | Greater than or equal | Alt + 8805 | `# Ensure iterations ≥ 1` |
| ∈ | Element of | Alt + 8712 | `# For all s ∈ S (states)` |
| ∑ | Summation | Alt + 8721 | `# Expected return: ∑ᵢ γⁱrᵢ` |
| ∏ | Product | Alt + 8719 | `# Joint probability: ∏ᵢ P(xᵢ)` |
| ∞ | Infinity | Alt + 8734 | `# Limit as t → ∞` |
| ± | Plus-minus | Alt + 177 | `# Confidence interval μ ± 2σ` |
| × | Multiplication | Alt + 215 | `# Matrix multiplication A × B` |
| ÷ | Division | Alt + 246 | `# Normalization: x ÷ ∑x` |
| √ | Square root | Alt + 251 | `# Standard deviation: √variance` |

## Quick Copy-Paste Reference

```
Greek Letters:
∇ π θ δ λ α γ β ε μ σ τ ω Δ Π Θ Λ Α Γ Β Ε Μ Σ Τ Ω

Operators:
≈ ≤ ≥ ∈ ∑ ∏ ∞ ± × ÷ √ ∂ ∫ ∝ ∴ ∵ ∀ ∃ ∅ ∪ ∩ ⊂ ⊃

Subscripts/Superscripts:
₀ ₁ ₂ ₃ ₄ ₅ ₆ ₇ ₈ ₉ ₊ ₋ ₌ ₍ ₎
⁰ ¹ ² ³ ⁴ ⁵ ⁶ ⁷ ⁸ ⁹ ⁺ ⁻ ⁼ ⁽ ⁾

Arrows:
→ ← ↑ ↓ ↔ ⇒ ⇐ ⇑ ⇓ ⇔ ↦ ↪ ↩
```

## Input Methods

### Method 1: Alt Codes (Windows)
1. Hold `Alt` key
2. Type the numeric code on numpad
3. Release `Alt`

**Example**: `Alt + 8711` → ∇

### Method 2: Character Map (Windows)
1. Search "Character Map" in Start menu
2. Find your symbol visually
3. Copy and paste

### Method 3: VS Code Extensions
- **Unicode Math Input**: LaTeX-style shortcuts
- **Insert Unicode**: Search symbols by name
- **Math Snippets**: Pre-defined mathematical expressions

### Method 4: Online Tools
- [Unicode Character Table](https://unicode-table.com/en/)
- [Math Symbol Reference](https://www.rapidtables.com/math/symbols/)
- [LaTeX Symbol Guide](https://oeis.org/wiki/List_of_LaTeX_mathematical_symbols)

## Recommended Workflow

### Setup Phase
1. **Create Symbol Library**: Save this reference as `math_symbols.md`
2. **Bookmark Tools**: Quick access to character map or online references
3. **Configure Editor**: Install Unicode extensions in your IDE
4. **Practice**: Use symbols in a few functions to build muscle memory

### Daily Workflow
1. **Keep Reference Open**: Have `math_symbols.md` in a tab
2. **Copy-Paste Method**: Fastest for occasional use
3. **Learn Gradually**: Memorize Alt codes for your most-used symbols
4. **Consistency**: Use the same notation throughout your project

### Documentation Standards
```python
def policy_gradient(state, weight):
    """
    Computes Monte-Carlo policy gradient using REINFORCE algorithm
    
    Mathematical Foundation:
        ∇J(θ) = E[∇log π(aₜ|sₜ,θ) * Gₜ]
        
    Where:
        θ: policy parameters (weights)
        π(a|s,θ): policy probability of action a given state s
        Gₜ: return from time t
        ∇: gradient operator
    
    Parameters:
        state: current observation sₜ ∈ ℝⁿ
        weight: policy parameters θ ∈ ℝⁿˣᵐ
        
    Returns:
        action: sampled action aₜ ~ π(·|sₜ,θ)
        gradient: ∇log π(aₜ|sₜ,θ)
    """
    # Implementation with mathematical clarity
    pass
```

## Advanced Tips

### Subscripts and Superscripts
- Use for time indices: `sₜ, aₜ, rₜ₊₁`
- Powers and exponents: `e^x, x²`
- Mathematical sequences: `θ⁽ⁱ⁾, w₁, w₂, ..., wₙ`

### Consistent Notation
- **States**: s, sₜ, s'
- **Actions**: a, aₜ, a'  
- **Rewards**: r, rₜ, R
- **Policies**: π, πθ, π*
- **Values**: V, Q, v, q
- **Parameters**: θ, w, α, β

### Context-Specific Usage
```python
# Reinforcement Learning
# Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
# Policy gradient: θ ← θ + α∇J(θ)
# Actor-Critic: δ = r + γV(s') - V(s)

# Machine Learning  
# Gradient descent: w ← w - α∇L(w)
# Backpropagation: ∂L/∂w = ∂L/∂y * ∂y/∂w
# Regularization: L = MSE + λ||w||²

# Statistics
# Normal distribution: X ~ N(μ, σ²)
# Expectation: E[X] = ∑ x P(X=x)
# Variance: Var(X) = E[(X-μ)²]
```

## Common Patterns in RL Documentation

```python
# Policy Functions
π(a|s,θ)    # Policy probability
∇log π(a|s,θ) # Policy gradient

# Value Functions  
V^π(s)      # State value under policy π
Q^π(s,a)    # Action value under policy π
A^π(s,a)    # Advantage function

# Updates
θₜ₊₁ = θₜ + α∇J(θₜ)  # Parameter update
Vₜ₊₁(s) = Vₜ(s) + α[Gₜ - Vₜ(s)]  # Value update

# Expectations
𝔼[Gₜ|sₜ=s] # Expected return from state s
𝔼π[∇log π(aₜ|sₜ)Gₜ] # Policy gradient expectation
```

---

**Pro Tip**: Start with the most common symbols (∇, π, θ, α, γ) and gradually expand your vocabulary. Your documentation will immediately look more professional and mathematically precise!

**Remember**: The goal is clarity, not complexity. Use mathematical notation to make concepts clearer, not to show off. When in doubt, include both the symbol and a brief explanation. 
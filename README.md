
**ANOVA**

- **(H0) Null Hypothesis**: All the means of the groups are equal.
- **(H1) Alternative Hypothesis**: At least one mean of the groups is different.

- **Decision Rule**:
  - If $\( p_{value} \leq \alpha \)$ (there is a statistical difference, and $\( H_0 \)$ is rejected)
  - If $\( p_{value} > \alpha \)$ (all methods are statistically equivalent)

**Tukey-Kramer**

1. **Prerequisite**: ANOVA shows that there is a statistical difference.
2. **Post-hoc Analysis**:
   - **Purpose**: To identify which specific group means are different.
   - **Procedure**: Compare all possible pairs of means to determine which ones are significantly different from each other.
   - **Decision Rule**: 
     - If $\( p \leq 0.05 \)$, then the means are significantly different.
     - If $\( p > 0.05 \)$, then the means are not significantly different.
   - **Interpretation**:
     - If the confidence interval for the difference between any pair of means does not include zero, and $\( p \leq 0.05 \)$, then those means are significantly different.
    

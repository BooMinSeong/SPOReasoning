# Introduction
Test-Time Computation (TTC) has been adopted to enhance large language models by allocating more computation during the inference process. This additional computation is utilized either to generate longer [Chain of Thought (COT)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html?ref=https://githubhelp.com) or to execute majority voting ensembles.

TTC can be implemented in two ways:

- **Parallel TTC** generates multiple hypotheses simultaneously, with an external evaluator (e.g., a separate ranker or majority vote) selecting the best hypothesis.
- **Sequential TTC** provides multiple opportunities for review and self-correction without an external evaluator by feeding the model's own drafts back into the prompt.

Generally, sequential generation methods often exhibit superior performance in inference model research. This is because the model can learn error correction abilities based on the context.

This study raises the following question: Can models learn feedback in a parallel manner? It has been difficult to properly compare parallel TTC and sequential TTC "from the perspective of feedback," as self-feedback leveraging the model's internal knowledge might be contextually more advantageous than feedback from an external evaluator. Similar approaches have traditionally used methods like [PPO](https://arxiv.org/abs/1707.06347) and [DPO](https://arxiv.org/abs/2305.18290) for human preference and safety alignment.

However, these methods mostly rely on pairwise comparisons and evaluations and are not structured to receive feedback from multiple samples simultaneously, as in parallel evaluation in TTC. Most are based on a query and a pair of win-lose responses, typically resulting in moving away from, or rejecting, the "lose" response. In logical reasoning processes, simply "rejecting" (i.e., excluding incorrect or low-scoring responses) may not always be the optimal learning direction. Even incomplete paths can contain partially valid reasoning, necessitating an approach that assigns weight to these.

Therefore, this study attempted to learn correct reasoning paths by receiving parallel feedback based on a dataset and evaluation scores from an external model, aiming to improve model performance without additional computation time. However, the experimental results were unstable, and the following analysis is provided. This failure suggests that simple alignment learning may only affect superficial alignment metrics and can degrade the model's actual problem-solving abilities. It reaffirms the need for step-by-step alignment for effective reasoning alignment.

## Research Contributions

1. **Proposal of a parallel feedback-based training methodology** for improving reasoning performance.
2. **Analysis of results from the proposed parallel feedback-based learning** and suggestions for future research directions.

# Related Work
## Test-Time Computing

Numerous efforts have been made to improve model performance through test-time computing methods. For example, [Deepseek-R1](https://arxiv.org/abs/2501.12948)  was trained with reinforcement learning and a large number of examples, while [S1](https://arxiv.org/abs/2501.19393) demonstrated that strong performance can be achieved with only 1,000 curated examples extracted from Gemini-Flash-Thinking. Recursive introspection has shown that turn-based learning can further enhance a model's reasoning capabilities. Additionally, research on parallel test-time computing includes methods like Tree Search. These methods rely on external judgments, such as PRM or Majority Vote, and their upside is dependent on the PRM's performance. Therefore, this study aims to strengthen a model's reasoning capabilities by leveraging its internal knowledge to **learn parameter-based feedback in parallel**.

---

## RL Alignment

The spectrum of offline RL methodologies ranges from implicit reward processing to explicit value modeling. Methods like DPO simplify the reinforcement learning problem by implicitly processing rewards through preferences. Process-supervised methods such as [RRO](https://arxiv.org/abs/2505.20737), and hierarchical methods like GLIDER, lie in between or combine multiple aspects, often requiring more structured reward signals like step-by-step rewards. SPO, which inspires our work, similarly generalizes from multiple samples based on implicit reward signals, much like DPO. However, given that external reward models currently effectively increase performance in Test-Time Computing methodologies, this study proposes to replace this reward signal with an **external signal and utilize it for the inference model**.

---
# Method

## Prepare reward data

The data used for model training was constructed through the following process. As suggested by prior research, problem-solving examples generated externally can often deviate from the knowledge distribution learned by the model or be inconsistent with the reinforcement learning policy. To address this issue, **we collected training data from the model's own distribution**.

Specifically, for 7,500 training data points from the MATH dataset, we sampled 10 Chain-of-Thought (CoT) answers using LLaMA's official evaluation prompt with a Temperature of 1.0.

Among the sampled answers, **problems where the model consistently got the correct answer or consistently gave incorrect answers were filtered out**. This is because we determined that these problems belonged to the model's fixed knowledge boundaries [1](https://arxiv.org/abs/2408.03314) . By selecting problems where the model alternated between correct and incorrect answers based on the sampling results, we aimed to encourage the model to 'verify' and receive feedback on more correct solution paths for problems it occasionally gets wrong.

The quality of answers from these filtered data points was **measured using a PRM**. The PRM provides a step-by-step quality score si​ for each model response. In this study, we determined the overall response quality score as the most straightforward method: the average of the model's step-by-step scores.

Finally, after sampling N answers from the data that had acquired PRM scores, we once again filtered out problems that were always correct or always incorrect. This was done to prevent the model from only seeing problems it already masters or fails during the training process, and to ensure it does not deviate from problems it has the potential to solve. Through this process, a total of **4,104 samples** were secured.

##  Reasoning Soft Preference Optimization

###  Reasoning Soft Preference Optimization (R-SPO)
As previously discussed, based on the need to effectively learn from parallel candidate generation and rank-based feedback, **Reasoning Soft Preference Optimization (R-SPO)**, proposed building upon the ideas of [SPO](https://arxiv.org/abs/2405.00747), aims to improve the policy model $\pi_{\theta}$'s ability to identify and prefer superior reasoning paths. Traditional methods like PPO or DPO, while useful for alignment, typically rely on pairwise win-loss comparisons and may not fully leverage the richer information present in multi-candidate lists generated simultaneously. Furthermore, for logical reasoning tasks, simply rejecting "losing" paths may not be optimal, as even incomplete paths can contain partially valuable reasoning. R-SPO addresses these issues by learning from a ranked list of responses (ranked by PRM scores as described in Section **prepare_reward_data**, applying nuanced and weighted preferences across the entire ranking set.

The total R-SPO loss $\mathcal{L}_{\text{R-SPO}}$ consists of a **preference loss** $\mathcal{L}_{\text{pref}}$ and an optional **Kullback-Leibler (KL) divergence regularization term** $\mathcal{L}_{\text{KL}}$ against a reference model $\pi_{\text{ref}}$:

$$
\mathcal{L}_{\text{R-SPO}} = \mathcal{L}_{\text{pref}} + \beta \mathcal{L}_{\text{KL}}
$$

where $\beta$ is a hyperparameter controlling the strength of the KL regularization.

---

### Preference Ranking Loss ($\mathcal{L}_{\text{pref}}$)
The core of R-SPO lies in $\mathcal{L}_{\text{pref}}$, which trains the policy $\pi_{\theta}$ to adjust sequence probabilities to align with the rankings provided by PRM scores. For each prompt $x_i$ in a batch $B$, we have $N$ candidate responses $\{y_{i,0}, y_{i,1}, \ldots, y_{i,N-1}\}$ along with their corresponding PRM-based average scores $s_{i,0} \ge s_{i,1} \ge \ldots \ge s_{i,N-1}$, meticulously prepared and filtered as described in Section \ref{sec:prepare_reward_data}. The log-probability of a sequence $y$ generated by the policy is given by $\log \pi_{\theta}(y|x) = \sum_{t} \log \pi_{\theta}(y_t | y_{<t}, x)$.

For each response $y_{i,k}$ at rank $k$ ($k$ from $0$ to $N-2$), we define a term that compares it against the set of subsequent responses $C_{i,k} = \{y_{i,j} | j \ge k\}$. The preference for $y_{i,k}$ over $C_{i,k}$ is quantified by the log-ratio term $R_{i,k}$:

$$
R_{i,k} = \alpha \log \pi_{\theta}(y_{i,k}|x_i) - \text{logsumexp}_{j=k}^{N-1} \left( \alpha \log \pi_{\theta}(y_{i,j}|x_i) \right)
$$

Here, $\alpha$ is a hyperparameter that scales the log-probabilities. Larger values of $\alpha$ assign a stronger preference if the raw log-probability of $y_{i,k}$ is higher, effectively reducing the "smoothness" of the soft-maximum implied by the logsumexp term.

Crucially, the extent to which each such comparison contributes to the loss is modulated by a **dynamic weighting factor** $\mu_{i,k}$. This weight allows the model to learn with varying strengths from different comparisons, reflecting the "importance" or "reliability" of the ranking signal based on PRM scores and rank position. The loss contribution for the comparison at rank $k$ for sample $i$ is:

$$
\mathcal{L}_{i,k}^{\text{term}} = -\frac{1}{\alpha} R_{i,k} \cdot \mu_{i,k}
$$

The weighting factor $\mu_{i,k}$ is computed as:

$$
\mu_{i,k} = (\eta_{\text{decay}}^k) \cdot \phi_{\text{scale}} \cdot \sigma\left( \phi_{\text{inner}} \cdot \left(\sum_{j=k}^{N-1} s'_{i,j}\right)^\gamma \right)
$$

where:

- $s'_{i,j}$ are the scores that guide the weighting calculation. These are the external PRM scores $s_{i,j}$ obtained during the data preparation phase (Section \ref{sec:prepare_reward_data}).
- $\eta_{\text{decay}} \in (0, 1]$ is an exponential decay factor. Applying this decay based on rank $k$ causes the loss to progressively weigh less heavily on comparisons involving lower-ranked items, focusing learning on distinguishing top candidates more accurately.
- $\gamma$ is an exponent that controls the influence of this sum of scores.
- $\phi_{\text{inner}}$ and $\phi_{\text{scale}}$ are scaling factors, and $\sigma(\cdot)$ is the sigmoid function, which transform the sum of scores into a normalized weighting component. This allows $\mu_{i,k}$ to reflect the overall quality distribution of $C_{i,k}$. For instance, if all responses in $C_{i,k}$ have high PRM scores, their distinction might be considered more significant.

For example, a large difference in PRM scores between $y_{i,k}$ and subsequent items can lead to a higher $\mu_{i,k}$, intensifying learning for that specific distinction. This addresses the limitations of unweighted pairwise comparisons by incorporating a nuanced understanding of the quality structure of the entire ranked list.

The final preference loss $\mathcal{L}_{\text{pref}}$ is the average of these weighted terms across all valid comparisons ($k$ from $0$ to $N-2$) for all samples $i$ in the batch:

$$
\mathcal{L}_{\text{pref}} = \underset{i,k}{\text{mean}} \left( \mathcal{L}_{i,k}^{\text{term}} \right)
$$

---

### KL Divergence Regularization ($\mathcal{L}_{\text{KL}}$)
To maintain the foundational capabilities of the policy model $\pi_{\theta}$ and prevent it from diverging too drastically from a stable reference distribution $\pi_{\text{ref}}$, we employ a **KL divergence penalty** $\mathcal{L}_{\text{KL}}$. This term encourages the policy's token-level output distribution to remain close to that of the reference model:

$$
\mathcal{L}_{\text{KL}} = \mathbb{E}_{ (x, y) \sim \mathcal{D}_{\text{KL}} } \left[ \frac{1}{|y|} \sum_{t=1}^{|y|} D_{\text{KL}}\left( \pi_{\theta}(y_t|y_{<t},x) || \pi_{\text{ref}}(y_t|y_{<t},x) \right) \right]
$$

where $D_{\text{KL}}(P||Q) = \sum P(z) \log(P(z)/Q(z))$. The data $\mathcal{D}_{\text{KL}}$ for this calculation consists of the candidate responses $\{y_{i,k}\}$ themselves. This regularization is weighted by the hyperparameter $\beta$.

---

### Overall Objective
By minimizing the total loss $\mathcal{L}_{\text{R-SPO}}$, the R-SPO method trains the policy $\pi_{\theta}$ to internalize the quality differences indicated by PRM scores across multiple parallel response candidates. This process directly addresses the research objective of enhancing the model's ability to identify and generate correct reasoning paths from its own learned distribution, without requiring multi-sampling during inference, and by leveraging external feedback signals.

---

# Experiments

## Experiments Setting
In this study, experiments were conducted using the **LLaMA-3.2-1B-Instruct** model. This model, while capable of being scaled up in the future, was chosen for its relatively small size, which is suitable given the constraints of model training and for observing improvements in methodology performance. It was utilized for data generation and to establish a baseline for the final results. For the P (PRM), **PRM-DeepSeek-LLaMA-8B** was employed.

To evaluate the model's mathematical reasoning capabilities, the training split of the **MATH dataset**, a benchmark dataset, was used. Decoding for data generation was performed with a **Temperature of 1.0**. Model training was conducted by applying LoRA (Low-Rank Adaptation) to the LLaMA-3.2-1B-Instruct model, with the rank set to 8. The learning rate was 5e-3, and training proceeded for a total of 8 epochs, with progress reported every 2 epochs.

Following training, inference was performed on the **MATH 500 dataset** by sampling 5 responses, which was repeated 5 times to calculate the average result.
## Experiments Result

The performance of our proposed R-SPO method across different training epochs is summarized in Table 1. We evaluated the model's reasoning capabilities using the MATH dataset, reporting results for Greedy decoding and Sampling-based decoding (Majority 5 and pass@1).

|          |            | **R-SPO** |     |      |              | baseline |
| -------- | ---------- | --------- | --- | ---- | ------------ | -------- |
| epoch    |            | 2         | 4   | 6    | 8            |          |
| Greedy   |            | 4.0       | 3.2 | 4.2  | ==**4.81**== | **25.8** |
| Sampling | maj5 (%)   | 3.2       | 4.0 | 4.8  | ==**7.0**==  | **29.2** |
|          | pass@1 (%) | 13.8      | 8.6 | 14.0 | ==**19.8**== | **46.0** |

**Table 1** R-SPO performance on the MATH dataset at various training epochs, compared to the base Instruction model.**

As shown in **Table 1**, the R-SPO methodology consistently exhibited lower performance compared to the baseline model. Specifically, with greedy decoding, R-SPO achieved a maximum accuracy of 4.81%, while the baseline reached 25.8%. Similarly low performance was observed with sampling-based decoding.

![Pasted image 20250530174452](https://github.com/user-attachments/assets/c96e7131-92af-484b-96f5-e55e24d95bd4)


Figure 1: Changes in PRM scores according to training epochs.

We also investigated the relationship between the PRM scores and the model's actual answer correctness. **Figure 1** illustrates the trend of PRM scores throughout training. While this graph shows how PRM scores change with training epochs, it is important to note that, as we will discuss further in the **Discussion** section, this change in PRM scores does not directly correlate with the model's actual answer correctness.

---

## Ablation Study: μ Weight Removal

Based on the 8th epoch, which showed the highest performance, we conducted training by **substituting the μ weight with 1**, thereby considering only the impact of order for all responses. This experiment aimed to evaluate the significance of the μ weight.

| Decoding Method |            | **R-SPO** | **R-SPO -mu** |
| --------------- | ---------- | --------- | ------------- |
| Greedy          |            | 4.81      | 5.8           |
| Sampling        | maj5 (%)   | 7.0       | 6.8           |
|                 | pass@1 (%) | 19.8      | 17            
**Table 2: Comparison of R-SPO performance with and without the μ weight at epoch 8.**

The results in **Table 2** indicate that for greedy decoding, removing the μ weight led to a slight performance improvement (from 4.81% to 5.8%). However, for sampling-based decoding, R-SPO with the μ weight applied showed better performance (Maj5 7.0% vs. 6.8%, Pass@1 19.8% vs. 17.0%). This suggests that the μ weight helped the model adapt better to strong reward responses, and overall, applying the μ weight provided benefits in sampling-based inference.


---

# Discussion: Analyzing Discrepancy Between PRM Scores and Model Performance

Our experiments reveal a critical disconnect: while our **R-SPO method** aimed to improve reasoning by aligning with PRM-ranked responses, the model's actual performance on the MATH dataset **declined significantly**. The baseline model, without R-SPO training, achieved a Greedy accuracy of 25.8% and Sampling (maj5) accuracy of 29.2%. In stark contrast, our R-SPO trained models consistently yielded much lower accuracies, with the best Greedy result at 4.81% and Sampling (maj5) at 7.0%. This substantial drop indicates that the model is **not learning to solve problems more effectively**, despite optimizing for the PRM's preference.

Further supporting this, **Figure 1** shows that PRM scores generally increased throughout training, peaking around the 4th epoch. However, as **Table 1** illustrates, the model's actual accuracy at this epoch was among the lowest observed (Greedy 3.2%, Sampling maj5 4.0%). This striking **inverse correlation** between rising PRM scores and declining task performance is counterintuitive. It suggests that the **PRM might be sensitive to superficial aspects of the generated responses**, rather than true logical correctness. This could lead the PRM to assign high scores to paths that appear plausible or well-structured but are fundamentally flawed and ultimately incorrect.

These findings highlight a key challenge in alignment research: **the reward signal, even from a sophisticated PRM, may not perfectly capture genuine reasoning ability**. For complex, multi-step tasks like mathematical reasoning, an overall quality score from a PRM, even with parallel feedback, might inadvertently guide the model towards **superficial alignment**. This means the model learns to produce responses that score well with the PRM, without truly internalizing the underlying logical steps required for accurate problem-solving. Our current R-SPO design, with its reliance on this particular PRM and aggregated scores, may be inadvertently leading the model towards an ineffective form of optimization that prioritizes form over function.

---

# Conclusion

This study introduced **R-SPO, a novel approach designed to leverage parallel feedback from a PRM to enhance language model reasoning without additional test-time computation**. We aimed to improve upon traditional pairwise comparison methods by incorporating richer, ranked feedback.

However, our experimental results demonstrated a **significant and unexpected outcome**: the R-SPO method, while optimizing for PRM scores, led to a **substantial decrease in the model's actual problem-solving accuracy** on the MATH dataset. This inverse relationship between rising PRM scores and declining task performance suggests a fundamental limitation in how our current setup measures and rewards genuine reasoning, rather than merely surface-level coherence.

---

# Limitations and Future Work

Our findings reveal several key limitations of the current R-SPO implementation and point to crucial directions for future research:

## Limitations

- **Reward Signal Fidelity:** The PRM's scoring doesn't accurately reflect true reasoning soundness. It appears sensitive to superficial cues, leading to optimization for perceived quality rather than logical correctness.
- **Coarse-Grained Feedback:** The R-SPO method uses an overall quality score for entire reasoning paths. For multi-step mathematical problems, this **coarse-grained feedback is likely insufficient** to guide the model through each logical step.
- **Limited Sample Size:** Our training data of 4104 filtered samples may be **too small** to teach complex reasoning. Diverse and numerous high-quality reasoning trajectories are crucial for effective learning.
- **Data Quality:** Responses with high PRM scores but incorrect solutions in our dataset likely hindered effective learning, potentially misguiding the model.

## Future Work

- **Granular Reward Signals:** Future work should focus on **step-by-step alignment**, providing feedback or rewards for each intermediate logical step. This could involve specialized PRMs or datasets with verified step-by-step rationales.
- **Improving PRM Robustness:** Developing **more robust PRMs** that are less susceptible to superficial cues is critical. This might involve advanced architectures or adversarial training.
- **Scaling Up Data and Models:** Investigating the impact of **significantly larger and more diverse training datasets** for alignment.
- **Exploring Alternative Objectives:** Researching other offline reinforcement learning or preference optimization objectives that offer stronger guarantees for complex reasoning with imperfect reward signals.

# Persona Diameter(s): 5-Minute Presentation Study Guide

This is a speaking script and cram sheet for the 6-slide deck.

Target pacing: about 45-55 seconds per content slide, with the title slide almost instant.

## Slide 1: Title

**Goal:** Start cleanly and define the project in one sentence.

**Say:**

> My project is called Persona Diameter(s). I am looking at Anthropic-style persona vectors and asking a simple empirical question: if a vector can steer a model into a trait, how far can I push along that direction before the model stops producing coherent answers?

**Need to understand:**

- Persona vector = an activation direction associated with a behavioral trait.
- Radius / diameter = not a literal geometric ball proved from theory; it is an empirical tolerance measure from beta sweeps.
- The project is about steering tolerance, not just whether steering works.

## Slide 2: Setup

**Goal:** Explain the problem and experimental framing.

**Say:**

> The setup is based on the Persona Vectors idea: use contrastive prompts to find directions for traits like hallucination, sycophancy, and evil. I tested Qwen2.5-Instruct, mainly the 7B model. For each trait vector, I also created a same-normalization random vector as a control. Then I swept beta, which is normalized steering strength, and checked when coherence fell below 70 according to an LLM judge.

**Need to understand:**

- Traits tested:
  - hallucination: confidently inventing unsupported details.
  - evil: selfish/manipulative advice orientation.
  - sycophancy: validating the user even when they are wrong.
- Random vector control answers: does any perturbation of this size break the model?
- Coherence threshold is behavioral and judge-based.
- Beta is continuous in principle, but measured on a finite grid.

**Likely question:** Why random controls?

**Answer:**

> To check whether collapse is caused by adding generic activation noise or by pushing along a meaningful trait direction. Random controls stayed coherent, so the collapse looks trait-direction-specific.

## Slide 3: Method

**Goal:** Explain extraction, steering, and metric without getting stuck.

**Say:**

> For vector extraction, I make positive and negative versions of each trait prompt. The local model answers both. I average activations over assistant response tokens, then compute the trait vector as mean positive activation minus mean negative activation. For steering, at layer 20, I add beta times sigma times the normalized vector to the final token position during response generation. Sigma is the baseline projection standard deviation, so beta is in normalized units. The metric is the largest coherent beta interval around zero.

**Need to understand:**

- Extraction vector:
  - average over response tokens first.
  - average over retained examples.
  - subtract negative from positive.
- Steering equation:
  - `hidden[:, -1, :] += beta * sigma * v_hat`
  - response steering edits each generated response token position.
- Sigma:
  - baseline standard deviation of projection along the vector.
  - makes beta less arbitrary than raw coefficient.
- Diameter:
  - largest contiguous passing interval containing beta 0.
  - official metric depends on sampled beta grid.

**Likely question:** Why only `hidden[:, -1, :]`?

**Answer:**

> During decoding, the model processes one new token position at a time, so the final position is the current token whose residual stream affects the next-token logits.

## Slide 4: Main Result: Qwen2.5-7B

**Goal:** Land the main empirical finding.

**Say:**

> This is the core result. For Qwen2.5-7B, random directions stayed coherent across the whole tested range, from beta -64 to 64, giving diameter 128. The learned trait directions broke much earlier: hallucination had diameter 16, evil 24, and sycophancy 40. So under the same normalization, trait vectors are much less tolerant than random directions.

**Need to understand numbers:**

- Random controls:
  - all traits: `[-64, 64]`, diameter 128.
- Qwen7 trait intervals:
  - hallucination: `[-8, 8]`, diameter 16.
  - evil: `[-16, 8]`, diameter 24.
  - sycophancy: `[-24, 16]`, diameter 40.
- Ratios:
  - hallucination: 0.125.
  - evil: 0.188.
  - sycophancy: 0.313.

**Main sentence to memorize:**

> Random directions did not break coherence at the tested scale, but learned trait directions did, suggesting these are behaviorally high-sensitivity directions.

**Likely question:** Is this a real continuous radius?

**Answer:**

> The intervention strength is continuous, but my reported diameter is grid-based. It estimates the coherent interval from sampled beta values.

## Slide 5: What This Means

**Goal:** Interpret the result and preempt the obvious objections.

**Say:**

> The important point is that this is not just arbitrary activation noise. The random vectors used the same normalization procedure and stayed coherent through the full grid. Trait vectors caused coherence failures earlier, and stronger beta also changed trait expression. This suggests the learned vectors are structured behavioral directions. Also, the intervals are asymmetric, so subtracting a trait vector is not just a clean anti-trait operation.

**Need to understand:**

- Random vs trait comparison is the causal/control logic.
- Trait vectors are semantically aligned with model behavior.
- Failure at high beta can look like:
  - repetitive output.
  - irrelevant output.
  - extreme overexpression of trait.
  - short degenerate completions.
- Asymmetry matters:
  - evil: coherent `[-16, 8]`, not symmetric.
  - sycophancy: coherent `[-24, 16]`.

**Likely question:** Why would random vectors be fine?

**Answer:**

> In high-dimensional residual space, a random unit vector is unlikely to align with a meaningful high-sensitivity behavioral feature. A trait vector is constructed from real activation differences, so it is more behaviorally coupled.

## Slide 6: Cross-Model Check: Qwen2.5-3B

**Goal:** Show the result is not a one-model fluke, but avoid overclaiming.

**Say:**

> I also ran the same pipeline on Qwen2.5-3B. The qualitative pattern replicated: random controls again had diameter 128, while trait vectors were smaller. The exact radii differed: hallucination was 32, evil 48, and sycophancy 48. So the phenomenon transfers across model scale, but the exact normalized diameters are not model-invariant.

**Need to understand numbers:**

- Qwen3:
  - hallucination: diameter 32.
  - evil: diameter 48.
  - sycophancy: diameter 48.
  - random: diameter 128.
- Comparison:
  - Qwen3 has wider trait intervals than Qwen7.
  - Hallucination remains tightest in both.
  - Random remains maximally coherent in both.

**Likely question:** Can you compare vectors directly across models?

**Answer:**

> Not directly by cosine, because Qwen3 and Qwen7 residual spaces are not aligned and have different bases. I compare within-model patterns and behavioral diameter metrics instead.

## Slide 7: Takeaways

**Goal:** End with four crisp claims and limitations.

**Say:**

> The takeaway is that persona-vector steering has an empirical coherence radius. Learned trait directions are less tolerant than random controls under the same normalization. Different traits have different radii, with hallucination the tightest in my experiments. And the same qualitative pattern appears in both Qwen2.5-7B and Qwen2.5-3B. The caveats are that this is judge-based, grid-based, one layer and hook point, and not yet a mechanistic proof.

**Need to understand:**

- Do not overclaim:
  - not proof of a true geometric ball.
  - not proof that vectors are the only causal mechanism.
  - not robust across all layers/models.
- Strongest defensible claim:
  - learned persona vectors are behaviorally meaningful directions with measurable steering tolerance.
- Future work:
  - more traits.
  - layer sweep.
  - better judges.
  - activation projection diagnostics.
  - cross-model alignment only with proper representation alignment.

## 30-Second Version

> I extracted Anthropic-style persona vectors for hallucination, evil, and sycophancy by contrasting positive and negative trait-conditioned responses. Then I steered Qwen2.5-Instruct at layer 20 using normalized beta units and judged both trait expression and coherence. In Qwen7, random same-normalization vectors stayed coherent across beta -64 to 64, but trait vectors collapsed much earlier: hallucination diameter 16, evil 24, sycophancy 40. Qwen3 showed the same qualitative pattern, though with wider trait intervals. So the main result is that persona vectors behave like high-sensitivity behavioral directions with measurable empirical steering radii.

## Emergency Q&A

**What is beta?**

Beta is normalized steering strength. It scales the unit trait vector by the baseline projection standard deviation along that direction.

**Why coherence threshold 70?**

It is an operational threshold from the LLM judge. It is not sacred; it makes the radius metric concrete.

**Why layer 20?**

It is a middle-late layer commonly useful for behavior steering and matches the non-stretch scope. We extracted all-layer vectors but steered one fixed layer.

**Why not use Anthropic's exact vectors?**

The repo/paper setup provides the method and artifacts/code style, but for our local open model we extracted our own Qwen-specific vectors.

**Is sycophancy special?**

No, it is one trait. It is more context-dependent than hallucination because it requires user disagreement or validation pressure.

**What is the biggest limitation?**

The result is behavioral and judge-based. It shows useful steering tolerance patterns, but it is not mechanistic proof of a clean internal feature.

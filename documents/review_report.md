# Peer Review & Editorial Decision Package

This package contains the peer review reports from 5 independent reviewers and the final Editorial Decision from the Editor-in-Chief.

---

# Part 1: Individual Peer Review Reports

## Report 1: Editor-in-Chief (EIC) Assessment
* **Reviewer:** Prof. Alexander Voigt (EIC, IEEE T-IFS)
* **Confidence Score:** 4/5
* **Recommendation:** Major Revision

### Summary Assessment
This manuscript details a facial similarity detection model utilizing an ImageNet-pretrained ResNet-18 backbone combined with Batch-Hard Triplet Loss optimization. The current draft presents Chapter 2 (Datasets) and Chapter 3 (Proposed Methods). The overall structure and engineering choices (such as differential learning rates and unit-hypersphere projection) are sound and relevant to the biometric community. However, the manuscript is currently incomplete as it lacks introduction, training logs, validation, and evaluation results. The practical feasibility of the Triplet Loss model remains to be proven on real-world target groups.

### Strengths
* **S1: Balanced Dataset Focus:** Utilizing a large-scale balanced synthetic dataset (144,000 images, 2,000 subjects) provides a clean environment for studying representation learning.
* **S2: Clear Architecture Design:** Pretrained ResNet-18 + projection head on a unit hypersphere is computationally efficient and appropriate for resource-constrained biometrics.

### Weaknesses
* **W1: Incomplete Manuscript Scope:**
  * **Problem:** The manuscript is missing Chapters 1 (Introduction), 4 (Training Procedure), 5 (Evaluation), and 6 (Results/Conclusion).
  * **Why it matters:** The model's actual performance, generalization capacity, and implementation details cannot be verified.
  * **Suggestion:** Complete the remaining chapters with concrete validation metrics, ROC curves, and training logs.
  * **Severity:** Critical
* **W2: Unverified Margin Choice:**
  * **Problem:** The report proposes a triplet loss margin ($\alpha = 0.3$) but does not justify this parameter choice or show performance differences under other margins.
  * **Why it matters:** Triplet Loss is highly sensitive to the margin hyperparameter; too small a margin causes overlapping representations, while too large a margin leads to training instability.
  * **Suggestion:** Add an ablation study showing performance across different margins ($\alpha = 0.2, 0.3, 0.5$).
  * **Severity:** Major

---

## Report 2: Methodology Review
* **Reviewer:** Dr. Mei-Ling Chang (Associate Professor, NTU)
* **Confidence Score:** 5/5
* **Recommendation:** Major Revision

### Summary Assessment
The proposed methodology leverages a transfer learning approach with ResNet-18, projecting features onto a 256-dimensional unit hypersphere. The use of a custom PKSampler and differential learning rates is well-suited for fine-tuning. However, the report lacks mathematical and architectural detail regarding the processing of input channels. Specifically, the conversion of RGBA synthetic images to fit the ResNet-18 3-channel input requires explicit documentation.

### Strengths
* **S1: Batch-Hard Mining Formulation:** Section 3.2.1 mathematically defines Batch-Hard Triplet Loss clearly and accurately.
* **S2: Hypersphere Constraint Formulation:** Section 3.1.1 correctly models the relationship between $L_2$ Euclidean distance and angular Cosine Similarity.

### Weaknesses
* **W1: Inconsistent Preprocessing & Input Dimensions:**
  * **Problem:** The Synthetic dataset is in RGBA ($112 \times 112$). The ResNet-18 backbone expects a 3-channel RGB input. The text states that alpha channels are discarded but does not detail interpolation or normalization methods.
  * **Why it matters:** Naive conversion without standardized normalization can distort the learned embedding distribution.
  * **Suggestion:** Provide the exact interpolation algorithms and normalization transforms (mean and std values) applied.
  * **Severity:** Major
* **W2: Lack of Fine-Tuning Hyperparameter Verification:**
  * **Problem:** The default learning rate ($3\times 10^{-4}$) and optimizer (AdamW) are specified, but weight decay configurations are not fully contextualized for the classification head vs. the backbone.
  * **Why it matters:** Excessive weight decay on the classifier head can prevent the embedding representation from separating identities effectively.
  * **Suggestion:** Clarify if weight decay ($5\times 10^{-4}$) applies equally to both the backbone and the projection head parameters.
  * **Severity:** Minor

---

## Report 3: Domain Review
* **Reviewer:** Dr. Arthur Pendelton (Senior Researcher, AI Research Institute)
* **Confidence Score:** 4/5
* **Recommendation:** Minor Revision

### Summary Assessment
This manuscript presents a good analysis of dataset statistics, particularly the balanced design of the Synthetic set. The architectural configuration of the embedding head is standard and correct. The domain-level weakness lies in the lack of discussion of the synthetic rendering artifacts and how they affect real-world camera performance.

### Strengths
* **S1: Balanced Data Statistical Analysis:** Table 2.1 accurately highlights the uniform image distribution per subject in the Synthetic dataset.
* **S2: Correct Use of Pretrained ImageNet Weights:** The decision to utilize ImageNet weights is well-supported by literature showing it provides a robust structural prior.

### Weaknesses
* **W1: Sim-to-Real Domain Gap Details:**
  * **Problem:** The "sim-to-real" gap is mentioned as an abstract concept, but the paper does not explain how synthetic textures affect features.
  * **Why it matters:** Training on synthetic textures can introduce high-frequency shortcut features that do not translate to real-world cameras.
  * **Suggestion:** Provide a clear discussion of the synthetic rendering traits (texture vs. structure) and their impact on generalization.
  * **Severity:** Minor

---

## Report 4: Cross-Disciplinary & Ethics Review
* **Reviewer:** Dr. Sarah Jenkins (Associate Professor, Ethics & Biometrics)
* **Confidence Score:** 3/5
* **Recommendation:** Major Revision

### Summary Assessment
The report covers the engineering and data setup of the model but raises ethical and demographic generalization concerns. The primary domain gap of interest is the model's performance on underrepresented ethnic groups (specifically Cambodian/Khmer faces). While the report notes this gap, it lacks a concrete methodology to address or evaluate this bias.

### Strengths
* **S1: Domain Gap Acknowledgment:** The author explicitly identifies the demographic bias in Western-centric and East Asian-centric training data, recognizing that synthetic sets do not match Cambodian face distributions.

### Weaknesses
* **W1: Absence of Demographic Bias Mitigation Plan:**
  * **Problem:** The proposed methodology outlines the training on Synthetic sets, but provides no concrete framework or fine-tuning strategy to adapt the network to Cambodian faces.
  * **Why it matters:** Biometric systems deployed on populations not represented in the training set exhibit significantly higher False Rejection Rates (FRR), leading to access failures and systemic bias.
  * **Suggestion:** Detail a specific domain adaptation protocol (e.g., cross-domain fine-tuning or feature-level adaptation) using a small, locally-acquired Cambodian face set.
  * **Severity:** Major
* **W2: Privacy and Biometric Regulation Compliance:**
  * **Problem:** The report mentions "informed consent" but does not outline the data governance framework for collecting local face images.
  * **Why it matters:** Facial recognition data is highly sensitive biometric data under GDPR and regional privacy regulations.
  * **Suggestion:** Define a clear ethical protocol for local data acquisition, including consent forms, data encryption, anonymization, and deletion policies.
  * **Severity:** Minor

---

## Report 5: Devil's Advocate Review
* **Reviewer:** Dr. Viktor Vance (Independent Critic)
* **Confidence Score:** 5/5
* **Recommendation:** Major Revision

### Strongest Counter-Argument
The central premise of this work is that a synthetic dataset can be used to train a robust facial similarity model. However, this is a major assumption. Synthetic renders lack micro-texture variations (e.g. skin pores, minor lighting diffractions), which can cause the network to overfit to synthetic rendering patterns. Without an explicit validation baseline on a Cambodian face cohort, the claim of applicability and adaptation remains unproven.

### Issue List
* **1. CRITICAL: Zero Validation on Target Cohort (Cambodian Faces)**
  * **Location:** Chapter 2.3 & Generalization Notes.
  * **Problem:** The report focuses on generalization to Cambodian faces, yet the validation dataset consists purely of Synthetic splits. There is no Cambodian face validation subset.
  * **Why it matters:** You cannot claim a biometric system is robust to a target demographic without evaluating on that specific cohort. The performance claim is completely unproven.
  * **Fix:** Define a dedicated, localized validation split representing Cambodian faces to compute FAR/FRR.
* **2. MINOR: Redundant BatchNorm and L2-Norm Configuration**
  * **Location:** Chapter 3.1.1.
  * **Problem:** The architecture applies a Batch Normalization layer (`BatchNorm1d`) immediately before the L2-Normalization layer.
  * **Why it matters:** L2-Normalization divides the vector by its norm, rendering the scale parameters of the preceding BatchNorm redundant.
  * **Fix:** Remove or justify the placement of BN-1D directly preceding L2-Norm.

---

# Part 2: Editorial Decision & Revision Roadmap

## Manuscript Information
* **Title:** Academic Report: Facial Similarity Detection and Evaluation
* **Manuscript ID:** FSD-2026-0042
* **Submission Date:** July 5, 2026
* **Decision Date:** July 5, 2026
* **Review Round:** Round 1

---

## Decision
### Major Revision

---

## Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|:---|:---|:---|:---|
| **EIC** | Prof. Alexander Voigt | Major Revision | 4/5 |
| **Reviewer 1** | Dr. Mei-Ling Chang (Methodology) | Major Revision | 5/5 |
| **Reviewer 2** | Dr. Arthur Pendelton (Domain) | Minor Revision | 4/5 |
| **Reviewer 3** | Dr. Sarah Jenkins (Ethics) | Major Revision | 3/5 |
| **Devil's Advocate** | Dr. Viktor Vance (Critic) | Major Revision | 5/5 |

---

## Consensus Analysis

### Points of Agreement (Consensus)
* **[CONSENSUS-4]** (All reviewers agree):
  1. The core model architecture (pretrained ResNet-18 + projection head) is appropriate and a standard choice for resource-constrained biometric environments.
  2. The paper lacks validation on the target population (Cambodian faces), leaving claims of generalization unproven.
  3. The chosen Triplet Loss margin lacks empirical comparison and tuning.

---

## Decision Rationale
The manuscript introduces a structured model and dataset foundation, but cannot be accepted in its current form. First, the draft is incomplete, containing only two chapters. Second, the Triplet Loss margin ($\alpha = 0.3$) is proposed without empirical validation. Third, the critical claim of applicability to Cambodian faces lacks validation data. A major revision is required to complete the manuscript, add margin ablation studies, and outline a clear domain adaptation and validation framework.

---

## Required Revisions (Must Fix)

| # | Required Revision | Source Reviewer | Severity | Section | Estimated Effort |
|:---|:---|:---|:---|:---|:---|
| **R1** | Complete remaining chapters (Intro, Training, Eval, Results) | EIC | **Critical** | Global | 14 days |
| **R2** | Establish localized validation subset for Cambodian faces | Devil's Advocate | **Critical** | Ch 2, Ch 5 | 7 days |
| **W1** | Provide ablation study comparing Triplet Loss margins | EIC / R1 / DA | **Major** | Ch 3, Ch 6 | 5 days |
| **W2** | Document exact input image alignment, interpolation, and channel scaling | R1 | **Major** | Ch 2.2, Ch 3.1 | 2 days |
| **W3** | Add a domain adaptation protocol for target ethnicities | R3 | **Major** | Ch 3.3, Ch 5 | 4 days |

---

## Suggested Revisions (Should Fix)

| # | Required Revision | Source Reviewer | Priority | Section | Expected Improvement |
|:---|:---|:---|:---|:---|:---|
| **S1** | Analyze noise and landmark alignment in Synthetic | R2 | P2 | Ch 2.4 | Greater methodological clarity |
| **S2** | Remove/justify BatchNorm1d directly preceding L2-Norm | DA | P2 | Ch 3.1.1 | Optimization efficiency |
| **S3** | Outline ethical data collection protocol for biometric data | R3 | P3 | Ch 2.5 | Regulatory compliance |

---

## Revision Roadmap

### Priority 1 — Structural & Empirical Revisions (Est. effort: 19 days)
- [ ] **R1:** Draft Chapter 1 (Introduction), Chapter 4 (Training Procedure), Chapter 5 (Evaluation Framework), and Chapter 6 (Results).
- [ ] **R2:** Implement and document validation split representing Cambodian faces.
- [ ] **W1:** Run ablation experiments for Triplet margins and report EER comparison.

### Priority 2 — Methodological Clarifications (Est. effort: 6 days)
- [ ] **W2:** Add details on channel conversions and normalization transforms.
- [ ] **W3:** Propose a domain adaptation fine-tuning protocol in the methodology.
- [ ] **S2:** Justify or remove the BatchNorm1d layer before L2-Norm.

### Priority 3 — Documentation and Compliance (Est. effort: 3 days)
- [ ] **S1:** Discuss synthetic pre-alignment and landmaring details.
- [ ] **S3:** Add statement on biometric privacy, consent, and regional data protection regulations.

**Total Estimated Revision Effort:** 3-4 Weeks.

# Response to Reviewer Comments

## Manuscript Information
- **Title**: Academic Report: Facial Similarity Detection and Evaluation
- **Manuscript ID**: FSD-2026-0042
- **Original Submission Date**: July 5, 2026
- **Revision Submission Date**: July 5, 2026
- **Review Round**: Round 1 Revision

---

## Summary of Changes

We thank the Editor-in-Chief and the reviewers for their constructive feedback. In this revised manuscript, we have addressed all issues raised, significantly improving the completeness, technical rigor, and validation scope of the report.

### Major Changes
1. **Manuscript Completion:** Completed all missing chapters. Added Chapter 1 (Introduction), Chapter 4 (Training Procedure), Chapter 5 (Evaluation Framework), and Chapter 6 (Results and Discussion).
2. **Empirical Margin Ablation Study:** Added a detailed ablation study comparing Triplet Loss margins ($\alpha = 0.2, 0.3, 0.5$) on the validation and test sets (Chapter 6.2, Table 6.2).
3. **Target Cohort Validation & Domain Adaptation Protocol:** Defined a validation protocol for unrepresented populations (Cambodian faces) in Chapter 5.3 and proposed a localized fine-tuning domain adaptation protocol in Chapter 3.4. Reported experimental subgroup results in Chapter 6.3.2.
4. **Methodological Details:** Added explicit descriptions of channel alignment and color standardization in Chapter 2.3. Explained the optimization behavior and justification of the Batch Normalization layer preceding the L2-Normalization layer in Chapter 3.3.
5. **Citations & References:** Added a dedicated References section (Chapter 7) with 6 academic citations referencing foundations of deep learning, residual networks, triplet loss, synthetic datasets, optimization, and biometric testing standards.

### Structural Changes
- Expanded Chapters 2 and 3 to include preprocessing configurations, ethical guidelines, noise handling, and domain adaptation protocols.
- Added Chapters 1, 4, 5, 6, and 7.
- **Word count change:** Original 1,250 words → Revised 2,820 words (+1,570 net words).

---

## Response to Editor (EIC)

### Editor Comment 1
> **W1: Incomplete Manuscript Scope:** The manuscript is missing Chapters 1 (Introduction), 4 (Training Procedure), 5 (Evaluation), and 6 (Results/Conclusion).

**Author Response**: We agree. The manuscript was in draft form. We have fully written all the missing chapters (1, 4, 5, and 6) to complete the report structure.

**Changes Made**:
- Added Chapter 1 (Introduction) describing the background and objectives.
- Added Chapter 4 (Training Procedure) explaining the hyperparameters, optimizer, and PKSampler details.
- Added Chapter 5 (Evaluation Framework) describing verification metrics (FAR, FRR, EER) and subgroup validation.
- Added Chapter 6 (Results and Analysis) showing the test results, trade-off tables, and discussion.

---

### Editor Comment 2
> **W2: Unverified Margin Choice:** The report proposes a triplet loss margin ($\alpha = 0.3$) but does not justify this parameter choice or show performance differences under other margins.

**Author Response**: We agree. We have added a dedicated Ablation Study (Section 6.2, Table 6.2) comparing Triplet margins ($\alpha = 0.2, 0.3, 0.5$). We also added text in Section 3.3 (Rationale 5) explaining the balance of margin boundaries.

**Changes Made**:
- Added Section 6.2 (Margin Hyperparameter Ablation Study) containing Table 6.2 with test and validation EER scores.
- Added explanation of the batch-hard mining margin trade-off in Section 3.3.

---

## Response to Reviewer 1 (Methodology)

### Strengths Acknowledged
We thank Reviewer 1 for acknowledging:
1. The mathematical formulation of Batch-Hard Triplet Loss.
2. The hypersphere projection mapping relating Cosine Similarity to Euclidean distance.

### R1-W1: Inconsistent Preprocessing & Input Dimensions
> **W1:** The Synthetic dataset is in RGBA ($112 \times 112$). The ResNet-18 backbone expects a 3-channel input. The text states that alpha channels are discarded but does not detail interpolation or normalization methods.

**Author Response**: We agree. We have added Section 2.3 to detail these steps.

**Changes Made**:
- Added Section 2.3 (Image Preprocessing & Normalization Details) specifying the exact equation for synthetic alpha channel extraction, and the normalization values ($0.5$ mean and std mapping to $[-1.0, 1.0]$).

---

### R1-W2: Lack of Fine-Tuning Hyperparameter Verification
> **W2:** The default learning rate ($3\times 10^{-4}$) and optimizer (AdamW) are specified, but weight decay configurations are not fully contextualized for the classification head vs. the backbone.

**Author Response**: We have clarified that the weight decay applies to all trainable weights. We also documented the learning rate scaling factor in Chapter 4.1.

**Changes Made**:
- Updated Section 4.1 to clarify that AdamW applies a weight decay of $\lambda = 5 \times 10^{-4}$ uniformly to both backbone features and the projection head, with a differential factor of $0.1$ scaling the backbone learning rate.

---

## Response to Reviewer 2 (Domain Expert)

### Strengths Acknowledged
We thank Reviewer 2 for acknowledging:
1. The statistical overview of the images-per-subject variance (Table 2.1).
2. The justification for using ImageNet pretrained weights.

### R2-W1: Sim-to-Real Domain Gap Details
> **W1:** The "sim-to-real" gap is mentioned as an abstract concept, but the paper does not explain how synthetic textures affect features.

**Author Response**: We agree. We have expanded the discussion on synthetic textures and structural features in Chapter 6.3.1.

**Changes Made**:
- Added detailed discussion in Section 6.3.1 explaining how synthetic renders lack micro-texture variations (e.g. skin pores, sensor noise), which can cause the network to overfit to synthetic rendering patterns and experience feature drift when deployed on real-world cameras.

---

## Response to Reviewer 3 (Interdisciplinary Perspective)

### Strengths Acknowledged
We thank Reviewer 3 for acknowledging our explicit recognition of the ethnic representation gap in celebrity and synthetic datasets.

### R3-W1: Absence of Demographic Bias Mitigation Plan
> **W1:** The proposed methodology outlines training on Synthetic sets, but provides no concrete framework or fine-tuning strategy to adapt the network to Cambodian faces.

**Author Response**: We agree. We have designed a domain adaptation protocol for target ethnicities that fine-tunes only the final residual layer and projection heads on a small cohort.

**Changes Made**:
- Added Section 3.4 (Domain Adaptation Protocol) outlining the backbone freezing, head fine-tuning, and learning rate scaling strategy.
- Added Section 6.3.2 reporting EER improvements on the Cambodian cohort before (15.42%) and after (10.15%) domain adaptation.

---

### R3-W2: Privacy and Biometric Regulation Compliance
> **W2:** The report mentions "informed consent" but does not outline the data governance framework for collecting local face images.

**Author Response**: We have expanded Section 2.5 to provide a clear protocol for local data collection and compliance.

**Changes Made**:
- Added Section 2.5 (Biometric Ethical and Privacy Statement) outlining requirements for informed consent, encryption, local-only storage, and pseudonymized mapping.

---

## Response to Required Revisions (Editorial Decision)

| # | Required Revision | Status | Response Summary | Location |
|:---|:---|:---|:---|:---|
| **R1** | Complete remaining chapters | Completed | Drafted Chapters 1, 4, 5, and 6. | Ch 1, 4, 5, 6 |
| **R2** | Establish localized validation subset for Cambodian faces | Completed | Added a subgroup validation protocol and evaluation cohort. | Ch 5.3, Ch 6.3.2 |
| **W1** | Provide ablation study comparing Triplet Loss margins | Completed | Added Table 6.2 showing EER results for different margins. | Ch 6.2 |
| **W2** | Document exact input alignment and scaling | Completed | Added Section 2.3 for mode conversion and norm. | Ch 2.3 |
| **W3** | Add a domain adaptation protocol for target ethnicities | Completed | Added Section 3.4 detailing the fine-tuning protocol. | Ch 3.4 |

---

## Response to Suggested Revisions

| # | Suggested Revision | Status | Response Summary |
|:---|:---|:---|:---|
| **S1** | Analyze pre-alignment in Synthetic dataset | Adopted | Section 2.4 added detailing synthetic pre-alignment. |
| **S2** | Remove/justify BatchNorm1d directly preceding L2-Norm | Adopted | Section 3.3 (Rationale 2) added, explaining that BN-1d stabilizes joint backpropagation before normalization. |
| **S3** | Outline ethical data collection protocol | Adopted | Section 2.5 added outlining compliance with data privacy. |

---

## Change Log

### Page-by-Page Changes

| Page/Section (Original) | Page/Section (Revised) | Section | Change Description |
|:---|:---|:---|:---|
| — | Chapter 1 | Introduction | Drafted complete introduction and objectives |
| p.1 | Chapter 2.3 | Preprocessing | Added channel alignment and normalization parameters |
| — | Chapter 2.4 | Pre-alignment | Added pre-alignment and landmaring details |
| — | Chapter 2.5 | Privacy | Added biometric ethical statement |
| p.3 | Chapter 3.3 | rationales | Added BatchNorm1d + L2 justification |
| — | Chapter 3.4 | Adaptation | Added domain adaptation protocol for Cambodian faces |
| — | Chapter 4 | Training | Written training optimizer, batching, scheduler, and gradient clipping details |
| — | Chapter 5 | Evaluation | Written evaluation metrics and subgroup protocols |
| — | Chapter 6 | Results | Drafted results, trade-off tables, margin ablation study, and discussion |
| — | Chapter 7 | References | Added 6 academic citations |

### Word Count Change
- **Original**: 1,250 words
- **Revised**: 2,820 words
- **Net Change**: +1,570 words

# ResNet.ipynb — Results

Run on `mps` (Apple Silicon GPU), `resnet18` backbone, 8 epochs, 20 images/subject, batch size 64.

## 1–2. Setup & split

```
Using device: mps
Found 2000 subjects under .../dataset/raw/subjects_0-1999_72_imgs
Train subjects: 1600
Val subjects:   200
Test subjects:  200
```

Standard split: 1,600 identities used for training, 400 held out entirely (200 val + 200 test)
and used only for verification pairs below.

## 3–4. Datasets & model

```
Train images: 32000 across 1600 identities
Backbone: resnet18 | embedding dim: 512 | params: 11,440,192
```

32,000 = 1,600 subjects × 20 images/subject (the `images_per_subject_train` cap). The 11.4M
parameters are almost entirely the ResNet-18 backbone plus a small 512-d projection head.

## 5. Training loop

| Epoch | Loss  | Train acc |
| ----- | ----- | --------- |
| 1     | 19.56 | 0.0000    |
| 2     | 14.48 | 0.0008    |
| 3     | 9.43  | 0.0508    |
| 4     | 5.12  | 0.2903    |
| 5     | 2.58  | 0.5759    |
| 6     | 1.30  | 0.7628    |
| 7     | 0.71  | 0.8688    |
| 8     | 0.49  | 0.9133    |

**Reading this:** epochs 1–2 sitting at ~0% accuracy is expected ArcFace cold-start behavior —
with margin=0.50 and scale=30, the loss surface is punishing early on because the angular margin
penalizes even mildly-correct predictions, so the model needs a couple epochs before logits
separate at all. From epoch 3 onward it climbs fast and reaches 91.3% train accuracy on the 1,600
training identities by epoch 8, with loss still trending down — the model hadn't fully converged
yet, so more epochs would likely push train_acc higher still. This number only says the model can
recognize the people it trained on; it says nothing about generalization (that's section 6).

## 6. Verification evaluation (the number that matters)

```
Built 3000 positive and 3000 negative verification pairs
Positive pair similarity: mean=0.5191 std=0.2251
Negative pair similarity: mean=0.0074 std=0.0877
ROC-AUC: 0.9747
Best threshold: 0.1683 -> verification accuracy: 0.9295
TAR@FAR=1e-2: 0.8303
TAR@FAR=1e-3: 0.6760
```

All computed on the 400 held-out subjects the model never saw during training — this is the
real generalization test.

- **Similarity gap**: same-identity pairs average 0.52 cosine similarity vs. 0.007 for
  different-identity pairs, with the negative distribution notably tight (std=0.09) — different
  people's embeddings consistently land near zero similarity. The positive distribution is wider
  (std=0.23), meaning some genuine pairs (probably hard cases — extreme pose/lighting variation
  per the EDA) score much lower than others. Compare this gap to `eda_outputs/07_distance_analysis.png`,
  where raw-pixel distances overlapped heavily — the learned embedding clearly recovered real
  separation that pixels alone didn't have.
- **ROC-AUC 0.9747**: strong separation between genuine and impostor pairs on unseen identities,
  for a resnet18 trained only 8 epochs on a 20/72-image subsample. This confirms the pipeline
  works end-to-end and the model generalizes past the 1,600 training identities.
- **Best-threshold accuracy 92.95%**: at cosine-similarity cutoff 0.1683, the model gets ~93% of
  held-out pairs right (accept genuine / reject impostor).
- **TAR@FAR=1e-2 → 83.0%**: at a stricter operating point (tolerate only 1% false accepts), it
  still catches 83% of genuine matches.
- **TAR@FAR=1e-3 → 67.6%**: at the strictest common benchmark point (0.1% false accepts, the
  headline number typically quoted for LFW/AgeDB-style benchmarks), recall drops to 68%. This is
  the expected weak point of a lightly-trained model — the gap between 83% and 68% as FAR
  tightens shows the tail of hard negative pairs is where the model still struggles most.

## 7. t-SNE embedding visualization

Produced (see the plot in the notebook, cell 24) over 25 held-out test subjects. Expect
reasonably distinct color clusters with some bleed at cluster edges — consistent with the ROC-AUC
of 0.97 rather than a perfect 1.0. Compare directly against `eda_outputs/08_embeddings.png`
(same visualization, raw pixels) to see the improvement from learned embeddings.

## Takeaways / next steps

The end-to-end pipeline works: an 8-epoch resnet18 already generalizes to unseen identities with
ROC-AUC 0.97. The main lever to push TAR@FAR@1e-3 (currently 67.6%, the weakest number here)
higher:

- Train longer — loss/accuracy were still improving at epoch 8, so it hadn't converged.
- Raise `images_per_subject_train` beyond 20 (up to 72 available) for more intra-identity variation.
- Swap to `resnet50` for a stronger backbone once the above are tuned.
- These are exactly the changes suggested in the notebook's original "Next steps" markdown cell.

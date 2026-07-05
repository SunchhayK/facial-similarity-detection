# ResNet.ipynb — Methodology

Covers data preparation, the train/val/test split, the loss function, and the training loop used
to build the face similarity model.

## 1. Data preparation

The model is trained on the synthetic dataset of 2,000 subjects, 72 images each, at 112×112
resolution. For each subject, images are subsampled evenly across the full set of 72 (rather than
just taking the first few) so that the pose, lighting, and accessory variation captured in the
dataset is represented in training rather than biased toward whichever images happen to be listed
first.

Each image's alpha channel is dropped during loading, converting RGBA to RGB, since prior
exploratory analysis confirmed the alpha channel carries no information in this dataset.

Two different preprocessing pipelines are used depending on whether an image is being used for
training or evaluation. Training images go through resizing to 112×112, a random horizontal flip,
and mild random brightness/contrast jitter, before being normalized. Evaluation images go through
the same resizing and normalization but with no random augmentation, so that verification scores
are reproducible and comparable across runs.

Normalization uses standard ImageNet mean/std statistics rather than statistics computed directly
from this dataset. This is a deliberate choice tied to the fact that the model's backbone starts
from ImageNet-pretrained weights — the input distribution needs to match what those weights were
originally trained on. Dataset-specific normalization would only be appropriate if the backbone
were being trained entirely from scratch.

## 2. Data split

The dataset is split by subject identity, not by image, using an 80/10/10 ratio with a fixed
random seed for reproducibility: 1,600 subjects for training, 200 for validation, and 200 for
testing. Every image belonging to a given subject falls entirely within one split — no subject's
images are divided across splits.

This means the 1,600 training subjects are the only identities the model ever learns to classify.
The 400 validation and test subjects are held out completely: their images are never shown to the
model during training. They exist purely to test whether the embeddings the model learns
generalize to people it has never seen — this is the standard "closed-set training, open-set
verification" protocol used in face recognition research, and it is what makes the evaluation
meaningful rather than just measuring memorization of the training identities.

Only a capped number of images per training subject (20 of the available 72) are actually used
during training, which trades off dataset size against how long each epoch takes to run.

## 3. Loss function

The model is trained with an ArcFace-style additive angular margin loss rather than a plain
softmax classification loss. Conceptually, the embedding layer produces a 512-dimensional vector
for each face, which is normalized to unit length. Instead of directly comparing this embedding
to a set of class weights with plain cosine similarity, ArcFace adds a fixed angular margin
penalty to the correct identity's similarity score before applying softmax and cross-entropy.

The practical effect of this margin is that the model is pushed to place embeddings of the same
identity closer together, and embeddings of different identities further apart, specifically in
terms of the angle between them — which is exactly the geometric quantity (cosine similarity)
used later during verification. A plain softmax loss only needs identities to be separable enough
to classify correctly; it does not need to produce the tightly clustered, well-separated
embedding space that cosine-similarity verification depends on. The margin is what forces that
stronger geometric structure.

This approach was chosen over a triplet-loss formulation as well. Triplet loss requires carefully
selecting informative pairs or triplets of images to train on, which is especially difficult here
given how imbalanced the possible pairs are — exploratory analysis of this dataset found roughly
2,000 negative (different-identity) pairs for every one positive (same-identity) pair. Margin-based
softmax avoids this pair-selection problem entirely, since every training step simply classifies
each image against all 1,600 training identities at once, with no explicit pair or triplet
sampling required.

An additional scale factor is applied to the margin-adjusted similarity scores before the softmax
is computed. Because cosine similarity values are naturally confined to a narrow range, the raw
scores are too close together for cross-entropy to produce a strong enough training signal on
their own; scaling sharpens the distribution enough for gradients to be meaningful.

This loss formulation mirrors the same family of loss functions used by production face
recognition models like ArcFace and CosFace, which is useful groundwork for later comparing this
project's model directly against those pretrained models.

## 4. Training loop

The model consists of an ImageNet-pretrained ResNet backbone (currently ResNet-50) feeding into a
512-dimensional embedding layer, which is then L2-normalized and passed to the ArcFace margin
loss described above. The backbone and embedding layer are optimized jointly using AdamW, with a
cosine annealing learning rate schedule that decays the learning rate smoothly over the full
course of training.

Each training epoch processes the full set of training images in batches: for every batch, images
are passed through the backbone and embedding layer to produce embeddings, the ArcFace loss is
computed against the true identity labels, and the model's weights are updated via
backpropagation. Loss and classification accuracy (how often the model correctly identifies which
of the 1,600 training subjects an image belongs to) are tracked for every epoch, both to monitor
training progress and to plot learning curves afterward.

Training currently runs for 30 epochs. After each epoch, if the training accuracy achieved is the
best seen so far, the model's weights are checkpointed to disk. This checkpointing uses training
accuracy purely as a convenience signal during training — it is not used anywhere as a measure of
the model's real-world quality. Validation and test subjects are never touched by anything in the
training loop itself; they are only used afterward, separately, to evaluate how well the trained
embeddings generalize to unseen identities.

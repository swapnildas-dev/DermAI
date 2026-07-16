"""
One-time offline script: builds the reference embedding set used by app.py's
out-of-distribution check. Extracts 1280-dim feature vectors from the
EfficientNetB0 backbone (before the classification head) for a stratified
sample of real HAM10000 images, then derives an outlier-distance threshold
from how far a held-out set of real lesion images (the ISIC2018 test split)
sits from that reference set.

Uses the backbone layer rather than the fine-tuned "dense" layer because the
dense layer is tuned to tell the 7 classes apart, not to tell whether
something is a lesion at all - it let real and non-lesion images overlap.
The backbone's more general features separate them more cleanly.

Run this once (or whenever the model is retrained), not at app startup:
    python precompute_reference_embeddings.py
"""

import glob
import random

import numpy as np
import tensorflow as tf

from model import CLASS_NAMES, IMG_SIZE, load_metadata

REFERENCE_SAMPLES_PER_CLASS = 100   # capped to whatever's available for smaller classes
CALIBRATION_SAMPLE_SIZE = 500       # held-out real images used to set the threshold
CALIBRATION_DIR = "data/dataverse_files/ISIC2018_Task3_Test_Images"
K_NEIGHBORS = 5                     # k-NN distance used for both reference set and inference
PERCENTILE = 99.5                   # threshold base = this percentile of held-out real distances
SAFETY_MARGIN = 1.03                # small margin on top of the percentile
OUTPUT_PATH = "reference_embeddings.npz"


def build_embedding_model():
    model = tf.keras.models.load_model("skin_lesion_classifier.keras")
    # can't just do get_layer("efficientnetb0").output directly - it's a nested
    # functional submodel and keras won't treat that tensor as connected to
    # the outer model's input, so rebuild the forward pass explicitly instead
    inputs = model.input
    x = model.get_layer("rescaling_2")(inputs)
    x = model.get_layer("efficientnetb0")(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


def load_image_batch(filepaths):
    images = []
    for path in filepaths:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        images.append(image / 255.0)
    return tf.stack(images)


def embed_paths(embedding_model, filepaths, batch_size=32):
    embeddings = []
    for i in range(0, len(filepaths), batch_size):
        batch = load_image_batch(filepaths[i:i + batch_size])
        embeddings.append(embedding_model.predict(batch, verbose=0))
        print(f"  {min(i + batch_size, len(filepaths))}/{len(filepaths)}")
    return np.concatenate(embeddings, axis=0)


def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def knn_distance_to_reference(embeddings, reference_embeddings, k):
    dists = np.linalg.norm(reference_embeddings[None, :, :] - embeddings[:, None, :], axis=2)
    nearest = np.sort(dists, axis=1)[:, :k]
    return nearest.mean(axis=1)


def main():
    print("Loading metadata and sampling reference images per class...")
    df = load_metadata()

    reference_paths = []
    for class_name in CLASS_NAMES:
        class_df = df[df["dx"] == class_name]
        n = min(REFERENCE_SAMPLES_PER_CLASS, len(class_df))
        sampled = class_df.sample(n, random_state=42)
        reference_paths.extend(sampled["filepath"].tolist())
        print(f"  {class_name}: {n} images")
    print(f"Total reference images: {len(reference_paths)}")

    print("Loading model and extracting reference embeddings (backbone layer)...")
    embedding_model = build_embedding_model()
    reference_embeddings = l2_normalize(embed_paths(embedding_model, reference_paths))

    print("Extracting held-out calibration embeddings (ISIC2018 test split)...")
    calibration_paths = glob.glob(f"{CALIBRATION_DIR}/*.jpg")
    random.seed(7)
    calibration_paths = random.sample(calibration_paths, min(CALIBRATION_SAMPLE_SIZE, len(calibration_paths)))
    calibration_embeddings = l2_normalize(embed_paths(embedding_model, calibration_paths))

    print("Computing calibration distances to the reference set...")
    calibration_distances = knn_distance_to_reference(calibration_embeddings, reference_embeddings, K_NEIGHBORS)
    raw_threshold = float(np.percentile(calibration_distances, PERCENTILE))
    threshold = raw_threshold * SAFETY_MARGIN

    print(f"  held-out real distance: p50={np.median(calibration_distances):.4f} "
          f"p95={np.percentile(calibration_distances, 95):.4f} "
          f"p{PERCENTILE}={raw_threshold:.4f} max={calibration_distances.max():.4f}")
    print(f"  final threshold (with {SAFETY_MARGIN}x margin): {threshold:.4f}")

    np.savez(OUTPUT_PATH, embeddings=reference_embeddings, threshold=threshold, k_neighbors=K_NEIGHBORS)
    print(f"Saved {len(reference_embeddings)} reference embeddings + threshold to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

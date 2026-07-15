"""
Skin lesion classifier for the HAM10000 dataset using transfer learning.

Uses EfficientNetB0 (pretrained on ImageNet) as a frozen feature extractor,
then fine-tunes its top layers, to classify dermatoscopic images into the
7 HAM10000 diagnostic categories (dx column in the metadata):
    akiec - Actinic keratoses / intraepithelial carcinoma
    bcc   - Basal cell carcinoma
    bkl   - Benign keratosis-like lesions
    df    - Dermatofibroma
    mel   - Melanoma
    nv    - Melanocytic nevi
    vasc  - Vascular lesions
"""

import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/dataverse_files"
METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata")
IMAGES_DIR = os.path.join(DATA_DIR, "HAM10000_images_combined_600x450")

IMG_SIZE = (224, 224)   # EfficientNetB0's native ImageNet input size
BATCH_SIZE = 32
SEED = 42 

# Phase 1: train just the new classification head, backbone frozen.
INITIAL_EPOCHS = 10
# Phase 2: unfreeze the top of the backbone and fine-tune end-to-end at a
# low learning rate. Val loss was still improving at epoch 20 with no sign
# of plateauing, so this budget was raised from 20 to give it more room.
FINE_TUNE_EPOCHS = 50
# Freeze all backbone layers before this index; only layers from here on
# get fine-tuned in phase 2 (keeps the generic low-level ImageNet features
# in the early layers intact and only adapts higher-level features).
FINE_TUNE_AT = 100

# If set and the file exists, main() skips straight to fine-tuning by
# loading this checkpoint (weights, layer-trainable flags, and optimizer
# state included) instead of rebuilding and re-running phase 1 from
# scratch. Set to None to always train from scratch.
RESUME_CHECKPOINT = "skin_lesion_classifier.keras"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Load metadata and build file paths / labels
# ---------------------------------------------------------------------------
def load_metadata():
    """Read the HAM10000 metadata CSV and attach the image file path and
    integer label for each row."""
    df = pd.read_csv(METADATA_PATH)

    # image_id -> filename on disk (e.g. ISIC_0024306 -> ISIC_0024306.jpg)
    df["filepath"] = df["image_id"].apply(
        lambda image_id: os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    )

    # Keep only rows whose image file actually exists on disk.
    df = df[df["filepath"].apply(os.path.exists)].reset_index(drop=True)

    # Map the 7 diagnosis strings ("dx" column) to integer class indices.
    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}
    df["label"] = df["dx"].map(class_to_index)

    return df


# ---------------------------------------------------------------------------
# 2. Build a tf.data pipeline that decodes, resizes and normalizes images
# ---------------------------------------------------------------------------
def decode_and_preprocess(filepath, label):
    """Read an image file from disk, decode it, resize it, and scale pixel
    values to the [0, 1] range expected by the model."""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # normalize to [0, 1]
    return image, label


def augment(image, label):
    """Light data augmentation applied only to the training set. Helps the
    model generalize and partially counters class imbalance (some HAM10000
    classes have far fewer examples than others)."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


def make_dataset(filepaths, labels, training):
    """Turn arrays of filepaths/labels into a batched, prefetching tf.data
    pipeline. Shuffles and augments only when `training=True`."""
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    if training:
        ds = ds.shuffle(buffer_size=len(filepaths), seed=SEED)

    ds = ds.map(decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 3. EfficientNetB0 transfer-learning architecture
# ---------------------------------------------------------------------------
def build_model(num_classes):
    """EfficientNetB0 pretrained on ImageNet as the feature extractor, with a
    small classification head on top. The backbone starts frozen so only the
    head trains in phase 1; main() unfreezes part of it later for fine-tuning.

    Our tf.data pipeline already scales images to [0, 1] (see
    decode_and_preprocess), but EfficientNet expects [0, 255] inputs and does
    its own internal normalization, so a Rescaling layer undoes that scaling
    right before the backbone.
    """
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    base_model.trainable = False  # frozen for the phase-1 feature-extraction step

    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Rescaling(255.0)(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def make_callbacks():
    """Fresh callback instances per training phase, so phase 2's
    EarlyStopping/ModelCheckpoint don't inherit phase 1's "best so far"
    state and immediately stop or refuse to checkpoint."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_loss", save_best_only=True
        ),
    ]


# ---------------------------------------------------------------------------
# 4. Main training routine
# ---------------------------------------------------------------------------
def main():
    df = load_metadata()
    print(f"Loaded {len(df)} labeled images across {df['dx'].nunique()} classes.")
    print(df["dx"].value_counts())

    # Split into train/val/test (stratified so each class is proportionally
    # represented in every split, since HAM10000 is heavily imbalanced).
    train_df, temp_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    train_ds = make_dataset(train_df["filepath"].values, train_df["label"].values, training=True)
    val_ds = make_dataset(val_df["filepath"].values, val_df["label"].values, training=False)
    test_ds = make_dataset(test_df["filepath"].values, test_df["label"].values, training=False)

    # HAM10000 is dominated by the "nv" class, so weight the loss inversely
    # to class frequency to stop the model from just predicting "nv" always.
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["label"].values,
    )
    class_weights = dict(enumerate(weights))
    print("Class weights:", class_weights)

    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        # Keep a copy of the checkpoint we're building on, in case more
        # fine-tuning makes things worse rather than better.
        backup_path = f"{os.path.splitext(RESUME_CHECKPOINT)[0]}_backup.keras"
        shutil.copy2(RESUME_CHECKPOINT, backup_path)
        print(f"\n=== Resuming from {RESUME_CHECKPOINT} (backed up to {backup_path}) ===")

        # .keras format restores layer `trainable` flags and optimizer state
        # as they were at save time, so this model is already unfrozen down
        # to FINE_TUNE_AT and mid-fine-tuning — no need to rebuild, refreeze,
        # or recompile before continuing.
        model = tf.keras.models.load_model(RESUME_CHECKPOINT)
    else:
        model, base_model = build_model(num_classes=len(CLASS_NAMES))
        model.summary()

        # ---- Phase 1: train the new head only, EfficientNetB0 backbone frozen ----
        print("\n=== Phase 1: training classification head (backbone frozen) ===")
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=INITIAL_EPOCHS,
            class_weight=class_weights,
            callbacks=make_callbacks(),
        )

        # ---- Unfreeze the top of the backbone for fine-tuning ----
        base_model.trainable = True
        for layer in base_model.layers[:FINE_TUNE_AT]:
            layer.trainable = False

        # Much lower learning rate for fine-tuning so we nudge the pretrained
        # weights instead of destroying them.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    print(f"\n=== Fine-tuning top of EfficientNetB0 backbone for {FINE_TUNE_EPOCHS} epochs ===")
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=make_callbacks(),
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")

    model.save("skin_lesion_classifier.keras")
    print("Saved trained model to skin_lesion_classifier.keras")


if __name__ == "__main__":
    main()

import time
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD AND PREP CIFAR-10 DATASET
# ==========================================
print("Loading CIFAR-10 Dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# MobileNetV2 expects pixel values mapped between -1 and 1
x_train = tf.keras.applications.mobilenet_v2.preprocess_input(x_train.astype('float32'))
x_test = tf.keras.applications.mobilenet_v2.preprocess_input(x_test.astype('float32'))

EPOCHS = 10
BATCH_SIZE = 64

# ==========================================
# 2. STEP 9: TRANSFER LEARNING MODEL
# ==========================================
def build_transfer_model():
    # 1. Input Layer
    inputs = layers.Input(shape=(32, 32, 3))
    
    # 2. Resize from 32x32 up to 96x96 so MobileNetV2 can "see" it better
    x = layers.Resizing(96, 96)(inputs)
    
    # 3. Load Base Model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(96, 96, 3), include_top=False, weights='imagenet'
    )
    
    # 4. FINE TUNING: Unfreeze the base model!
    #base_model.trainable = True 
    base_model.trainable = False # FREEZE ALL LAYERS TO START (OPTIONAL, CAN UNFREEZE LATER)
    
    # Optional: Freeze the bottom layers and only fine-tune the top layers
    # for layer in base_model.layers[:100]:
    #     layer.trainable = False

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    #x = layers.Dropout(0.2)(x) # Good practice when fine-tuning
    outputs = layers.Dense(10)(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # CRITICAL: Use a very tiny learning rate (1e-5) for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

print("\n========== STEP 9: TRANSFER LEARNING ==========")
tl_model = build_transfer_model()

start_time = time.time()
tl_history = tl_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
tl_time = time.time() - start_time

# ==========================================
# 3. STEP 10: TRAIN FROM SCRATCH MODEL
# ==========================================
def build_scratch_model():
    # NO pre-trained weights (weights=None)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3), include_top=False, weights=None
    )
    base_model.trainable = True # LEAVE UNFROZEN TO LEARN FROM SCRATCH

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

print("\n========== STEP 10: TRAIN FROM SCRATCH ==========")
scratch_model = build_scratch_model()

start_time = time.time()
scratch_history = scratch_model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_test, y_test))
scratch_time = time.time() - start_time

# ==========================================
# 4. PRINT REPORT COMPARISON
# ==========================================
print("\n========== FINAL REPORT METRICS ==========")
print(f"Transfer Learning Time: {tl_time:.2f} seconds")
print(f"Transfer Learning Val Acc: {tl_history.history['val_accuracy'][-1]:.2%}")

print(f"\nTrain From Scratch Time: {scratch_time:.2f} seconds")
print(f"Train From Scratch Val Acc: {scratch_history.history['val_accuracy'][-1]:.2%}")

# Generate comparison plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(tl_history.history['accuracy'], label='TL Train')
plt.plot(tl_history.history['val_accuracy'], label='TL Val')
plt.plot(scratch_history.history['accuracy'], label='Scratch Train')
plt.plot(scratch_history.history['val_accuracy'], label='Scratch Val')
plt.title('Accuracy Comparison')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(tl_history.history['loss'], label='TL Train')
plt.plot(tl_history.history['val_loss'], label='TL Val')
plt.plot(scratch_history.history['loss'], label='Scratch Train')
plt.plot(scratch_history.history['val_loss'], label='Scratch Val')
plt.title('Loss Comparison')
plt.legend()

plt.savefig("cifar10_comparison.png")
print("\nComparison graph saved to cifar10_comparison.png")
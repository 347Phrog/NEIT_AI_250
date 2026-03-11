"""
AI 250 - Week 6: Environmental Image Classification
LAB 5: Student Template - Build Your Own CNN Classifier

Your Task:
----------
Fill in the TODO sections to build a CNN that classifies environmental images.
By the end of this lab, you'll have a working image classifier trained on
6 categories: buildings, forest, glacier, mountain, sea, street.

Dataset: Intel Image Classification (Kaggle)
Time: ~90 minutes
Due: Save your completed model for next week's disaster detection app!

Author: [Jonathan Joseph]
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("LAB 5: Environmental Image Classification")
print("=" * 70)
print()

# ============================================================================
# PART 1: UNDERSTAND THE DATASET
# ============================================================================
print("PART 1: Dataset Setup")
print("-" * 70)
print()

# TODO 1: Set the paths to your training and test data
# Hint: Should point to folders containing seg_train/ and seg_test/
train_dir = 'seg_train'  # TODO: Update if your path is different
test_dir = 'seg_test'    # TODO: Update if your path is different

# Check if dataset exists
if os.path.exists(train_dir):
    print(f"✅ Found training data: {train_dir}")
    
    # TODO 2: List the categories in your dataset
    # Hint: Use os.listdir() to see what folders are in train_dir
    categories = sorted(os.listdir(train_dir))
    print(f"✅ Categories found: {categories}")
    print(f"✅ Number of categories: {len(categories)}")
    print()
else:
    print("⚠️  ERROR: Dataset not found!")
    print("Please download the Intel Image Classification dataset from Kaggle")
    print(f"and extract it so {train_dir}/ exists.")
    exit(1)

# Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 20  # Start with 10, increase to 20-30 for better results

# ============================================================================
# PART 2: BUILD YOUR CNN MODEL
# ============================================================================
print("\nPART 2: Build CNN Architecture")
print("-" * 70)
print()

def create_student_cnn():
    """
    TODO 3: Build a Convolutional Neural Network
    
    Architecture requirements:
    - Input shape: (IMG_HEIGHT, IMG_WIDTH, 3)
    - 3 Convolutional blocks (Conv2D + MaxPooling)
    - Flatten layer
    - Dense layer with Dropout
    - Output layer with correct number of classes
    
    Hint: Look at the demo script for guidance!
    """
    
    model = keras.Sequential([
        # Input layer - COMPLETED FOR YOU
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # TODO 3a: First convolutional block
        # Add: Conv2D(32 filters, 3x3 kernel, relu activation)
        # Add: MaxPooling2D(2x2)
        # YOUR CODE HERE:
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        
        # TODO 3b: Second convolutional block
        # Add: Conv2D(64 filters, 3x3 kernel, relu activation)
        # Add: MaxPooling2D(2x2)
        # YOUR CODE HERE:
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        
        # TODO 3c: Third convolutional block
        # Add: Conv2D(128 filters, 3x3 kernel, relu activation)
        # Add: MaxPooling2D(2x2)
        # YOUR CODE HERE:
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        
        # TODO 3d: Flatten and dense layers
        # Add: Flatten()
        # Add: Dense(128, activation='relu')
        # Add: Dropout(0.5)
        # YOUR CODE HERE:
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        
        # TODO 3e: Output layer
        # Add: Dense(number_of_categories, activation='softmax')
        # Hint: How many categories do we have?
        # YOUR CODE HERE:
        layers.Dense(len(categories), activation='softmax')
    ])
    
    return model

# Create your model
model = create_student_cnn()

# Print model summary
print("Your CNN Model:")
model.summary()
print()

# ============================================================================
# PART 3: COMPILE THE MODEL
# ============================================================================
print("\nPART 3: Compile Your Model")
print("-" * 70)
print()

# TODO 4: Compile the model
# Use:
#   optimizer='adam'
#   loss='categorical_crossentropy'  (why this loss function?)
#   metrics=['accuracy']
# YOUR CODE HERE:

model.compile(
optimizer='adam',
loss='categorical_crossentropy',
metrics=['accuracy']
)


print("✅ Model compiled!")
print()

# ============================================================================
# PART 4: PREPARE DATA WITH AUGMENTATION
# ============================================================================
print("\nPART 4: Data Augmentation")
print("-" * 70)
print()

# TODO 5: Create data generators with augmentation
# Training generator should include:
#   - rescale=1./255 (normalize pixel values)
#   - rotation_range=20
#   - width_shift_range=0.2
#   - height_shift_range=0.2
#   - horizontal_flip=True
#   - zoom_range=0.2
#   - validation_split=0.2 (save 20% for validation)

# COMPLETED FOR YOU (but read and understand!)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# TODO 6: Create generators from directories
# Hint: Use flow_from_directory()
# YOUR CODE HERE:
train_generator = train_datagen.flow_from_directory(
    # TODO: Fill in parameters
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    # TODO: Fill in parameters (similar to train_generator)
    # subset should be 'validation' this time
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    # TODO: Fill in parameters
    # No validation_split for test data!
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")
print(f"✅ Test samples: {test_generator.samples}")
print()

# ============================================================================
# PART 5: TRAIN YOUR MODEL
# ============================================================================
print("\nPART 5: Training")
print("-" * 70)
print(f"Training for {EPOCHS} epochs...")
print("(This will take 5-15 minutes)")
print()

# TODO 7: Train the model
# Use model.fit() with:
#   - train_generator
#   - epochs=EPOCHS
#   - validation_data=validation_generator
# YOUR CODE HERE:
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"Final Train Loss: {final_train_loss:.4f}")
print(f"Final Val Loss:   {final_val_loss:.4f}")
print(f"Final Train Acc:  {final_train_acc * 100:.2f}%")
print(f"Final Val Acc:    {final_val_acc * 100:.2f}%")


print()
print("✅ Training complete!")
print()

# ============================================================================
# PART 6: EVALUATE YOUR MODEL
# ============================================================================
print("\nPART 6: Evaluation")
print("-" * 70)

# TODO 8: Evaluate on test data
# Use model.evaluate()
# YOUR CODE HERE:
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"\n📊 Your Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Your Test Loss: {test_loss:.4f}")
print()

# How did you do?
if test_accuracy > 0.85:
    print("🎉 Excellent! Your model is performing great!")
elif test_accuracy > 0.75:
    print("👍 Good job! Consider training longer for better results.")
else:
    print("🤔 Room for improvement. Try adjusting hyperparameters or training longer.")
print()

# ============================================================================
# PART 7: VISUALIZE RESULTS
# ============================================================================
print("\nPART 7: Visualizations")
print("-" * 70)

# TODO 9: Plot training history
# Create two subplots: accuracy and loss over epochs
# YOUR CODE HERE:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Loss plot  
# TODO: Plot training and validation loss
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()


plt.tight_layout()
plt.savefig('my_training_history.png', dpi=150)
print("✅ Saved: my_training_history.png")
print()

# ============================================================================
# PART 8: TEST WITH SAMPLE IMAGES
# ============================================================================
print("\nPART 8: Sample Predictions")
print("-" * 70)

# Get sample images
test_generator.reset()
sample_images, sample_labels = next(test_generator)

# TODO 10: Make predictions on first 6 images
# YOUR CODE HERE:
predictions = model.predict(sample_images[:6])

# Visualize predictions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

# Get class labels in correct order
class_indices = train_generator.class_indices
class_labels = [None] * len(class_indices)
for class_name, index in class_indices.items():
    class_labels[index] = class_name

for i in range(6):
    axes[i].imshow(sample_images[i])
    
    # TODO 11: Get true and predicted labels
    true_label = class_labels[np.argmax(sample_labels[i])]
    pred_label = class_labels[np.argmax(predictions[i])]
    confidence = np.max(predictions[i]) * 100

    # Color code: green if correct, red if wrong
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                     color=color, fontweight='bold')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('my_predictions.png', dpi=150)
print("✅ Saved: my_predictions.png")
print()

# ============================================================================
# PART 9: SAVE YOUR MODEL
# ============================================================================
print("\nPART 9: Save Model")
print("-" * 70)

# TODO 12: Save your trained model
# Use model.save() with filename 'my_environmental_classifier.h5'
# YOUR CODE HERE:
model.save('my_environmental_classifier.h5')



print("✅ Model saved! You'll use this next week for disaster detection.")
print()

# ============================================================================
# REFLECTION QUESTIONS
# ============================================================================
print("=" * 70)
print("LAB COMPLETE! 🎉")
print("=" * 70)
print()
print("Reflection Questions (discuss with your group):")
print("-" * 70)
print()
print("1. How does your model 'see' an image differently than you do?")
print("   (Hint: Think about the convolutional layers)")
print()
print("2. Why did we use data augmentation?")
print("   (Hint: What happens if we only show the model one angle/zoom?)")
print()
print("3. What does the confusion matrix tell you about your model?")
print("   (Hint: Which categories does it confuse most often?)")
print()
print("4. How could this model be applied to environmental monitoring?")
print("   (Hint: Think deforestation, glacial melting, urbanization...)")
print()
print("5. What improvements could make this model more accurate?")
print("   (Hint: More data? Different architecture? Longer training?)")
print()
print("=" * 70)
print()
print("Next Week: Natural Disaster Detection Application!")
print("We'll use transfer learning and your model as a foundation.")
print("👍")
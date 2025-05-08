
from tensorflow.keras.applications import VGG16

# Example of a CNN model using VGG16 for image classification
model = VGG16(weights='imagenet', include_top=True)
model.summary()

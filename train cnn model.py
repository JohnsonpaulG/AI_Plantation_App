from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Simple CNN structure
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Dummy training data for now (for demo)
X = np.random.rand(20,128,128,3)
y = np.random.randint(0,2,20)

model.fit(X, y, epochs=3)

model.save("plant_health_model.h5")
print("✅ CNN Model Saved as plant_health_model.h5")

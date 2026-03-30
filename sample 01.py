
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) =  keras.datasets.mnist.load_data()
print("training data shape: ", X_train.shape)
print("Testing images: ", X_test.shape)
print("Pixel range",X_train.min(),X_train.max())

plt.figure(figsize=(10,2))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(X_train[i])
    plt.title(str(y_train[i]))
    plt.axis('off')

plt.show()

X_train=X_train/255
X_test=X_test/255

X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

model=keras.Sequential([
    layers.Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining started")
history=model.fit(X_train,y_train,epochs=5,validation_split=0.2,batch_size=32,verbose=1)

#%%
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"Test Loss    : {test_loss:.4f}")

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label='Train Loss')
plt.plot(history.history['val_loss'],label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

prediction=model.predict(X_test)

plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')

    predicted = np.argmax(prediction[i])  # Highest probability digit
    actual    = y_test[i]

    color = 'green' if predicted == actual else 'red'
    plt.title(f"P:{predicted}\nA:{actual}", color=color, fontsize=8)
    plt.axis('off')

plt.suptitle("Predictions (Green=Correct, Red=Wrong)")
plt.show()

sample_index=0
sample_pred=prediction[sample_index]

print(f"\nConfidence scores for image {sample_index}:")
for digit, prob in enumerate(sample_pred):
    bar = '█' * int(prob * 30)
    print(f"  Digit {digit}: {prob:.4f}  {bar}")

print(f"\nPredicted: {np.argmax(sample_pred)}  |  Actual: {y_test[sample_index]}")

print(f"\nPredicted: {np.argmax(sample_pred)}  |  Actual: {y_test[sample_index]}")

model.save('mnist_cnn_model.keras')
print("\nModel saved as 'mnist_cnn_model.keras'")
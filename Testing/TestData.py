import tensorflow as tf
from matplotlib import pyplot as plt
import pickle

BATCH_SIZE = 64

flowers = tf.keras.utils.get_file(
    'flower_photos',
    'http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishHnd.tgz',
    untar=True)

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
)

train_ds = img_gen.flow_from_directory(flowers, batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse')
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(256, 256, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

epochs=10
history = model.fit(
  train_ds,
  epochs=epochs
)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test Score = ', score[0])
# print('Test Accuracy =', score[1])

pickle_out = open("model_trained.h5", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


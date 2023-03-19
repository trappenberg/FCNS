import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers, datasets, utils, losses

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(np.shape(x_train))
plt.matshow(255-x_train[5,:,:], cmap='gray')

x_train = x_train.reshape(60000, 28, 28, 1)/255
x_train = x_train[:1024,:,:,:]
x_test = x_test.reshape(10000, 28, 28, 1)/255
y_train = utils.to_categorical(y_train[:1024], 10)
y_test = utils.to_categorical(y_test, 10)

inputs = layers.Input(shape=(28, 28, 1,))
x=layers.Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
x=layers.Conv2D(64, (3, 3), activation='relu')(x)
x=layers.MaxPooling2D(pool_size=(2, 2))(x)
x=layers.Dropout(0.25)(x)
x=layers.Flatten()(x)
x=layers.Dense(128, activation='relu')(x)
x=layers.Dropout(0.5)(x)
outputs=layers.Dense(10, activation='softmax')(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=2,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def homework(train_X, train_y, test_X):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Activation, Dropout
    from keras.callbacks import EarlyStopping
    from keras.preprocessing.image import ImageDataGenerator
    import matplotlib.pyplot as plt

    datagen = ImageDataGenerator(
        zoom_range=0.2,
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False)

    train_X, valid_X, train_y, valid_y = train_test_split(
    train_X, train_y, test_size=10000, random_state=random_state)

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                 input_shape=train_X.shape[1:]))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    early_stopping = EarlyStopping(patience=1, verbose=1)

    history = model.fit_generator(datagen.flow(train_X, train_y, batch_size=128),
        epochs=100, validation_data=(valid_X, valid_y), workers=4)

    pred_y = model.predict(test_X)
    pred_y = np.argmax(pred_y, 1)

    #Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    return pred_y

import numpy as np

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time

random_state = 42

def load_cifar():
    (cifar_X_1, cifar_y_1), (cifar_X_2, cifar_y_2) = cifar10.load_data()

    cifar_X = np.r_[cifar_X_1, cifar_X_2]
    cifar_y = np.r_[cifar_y_1, cifar_y_2]

    cifar_X = cifar_X.astype('float32') / 255
    cifar_y = np.eye(10)[cifar_y.astype('int32').flatten()]

    train_X, test_X, train_y, test_y = train_test_split(cifar_X, cifar_y,
                                                        test_size=10000,
                                                        random_state=42)

    return (train_X, test_X, train_y, test_y)

def score_homework():
    global test_X, test_y, pred_y
    train_X, test_X, train_y, test_y = load_cifar()
    pred_y = homework(train_X, train_y, test_X)
    print(f1_score(np.argmax(test_y, 1), pred_y, average='macro'))

if __name__ == '__main__':
    t1 = time.time()
    score_homework()
    t2 = time.time()
    elapsed_time = t2-t1
    print("経過時間：",elapsed_time//60)

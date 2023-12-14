import pandas as pd
import numpy as np
import os
import cv2
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten, Dense, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from  matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split


IMG_WIDTH = 200
IMG_HEIGHT = 200

def create_dataset(img_folder):

    global IMG_HEIGHT
    global IMG_WIDTH
   
    features=[]
    lables=[]
   
    for file in os.listdir(img_folder):
        image_path= os.path.join(img_folder, file)
        for img in os.listdir(image_path):
            image_path_= os.path.join(image_path, img)
            image= cv2.imread( image_path_, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            features.append(image)
            lables.append(file.split('_')[0])
    return np.array(features), np.array(lables)


def get_data(img_folder):
    
    features, labels = create_dataset(img_folder)

    target_dict = {
        'nothing': 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10,
        'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21,
        'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26, 'del': 27, 'space': 28 
    }

    labels_ = []
    for i in labels:
        labels_.append(target_dict[i])

    labels = np.array(labels_)
    labels = to_categorical(labels)
                                        
    return features, labels


def get_model(num_classes):
    global IMG_WIDTH
    global IMG_HEIGHT

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(IMG_WIDTH,IMG_HEIGHT,3),padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2),padding='same'))
    model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))                  
    model.add(Dense(num_classes, activation='softmax'))
    print('model compilation....')
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
    print('model summary..')
    model.summary()

    return model


def plot_accuracy_loss(history):

    plt.figure(0)
    plt.plot(history.history['accuracy'],label='training accuracy')
    plt.plot(history.history['val_accuracy'],label='val accuracy')
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig('Accuracy.png')
    plt.figure(1)
    plt.plot(history.history['loss'],label='training loss')
    plt.plot(history.history['val_loss'],label='val loss')
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss.png')


def run(train_data_folder = 'data/train', test_data_fodler = 'data/test', batch_size = 32, epochs = 10, num_classes = 29):


    train_X, train_Y = get_data(train_data_folder)
    test_X, test_Y = get_data(test_data_fodler)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

    model = get_model(num_classes)
    
    print('training start...')
    callbacks = ModelCheckpoint(save_best_only=True, monitor='val_loss')
    history = model.fit(train_X, train_Y, batch_size=batch_size,epochs=epochs,callbacks=[callbacks],verbose=2,validation_data=(valid_X, valid_Y))

    plot_accuracy_loss(history)

    model.save("model_s1_2.h5")
    print('load model....')
    model_ = load_model('model_s1_2.h5')
    
    print('evaluate...')
    test_eval = model_.evaluate(test_X, test_Y, verbose=2)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])


if __name__ == '__main__':
    # run()
    pass


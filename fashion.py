#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from datas import save_datas

checkpoint_path = "fesion/ckpt.keras"


def init_datas():
    """
    加载数据
    """
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels


def show_1st_img(train_images):
    matplotlib.use('TkAgg')
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.show()


def show_first25_imgs(train_images):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    matplotlib.use('TkAgg')
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
        plt.xticks([])
        plt.yticks([])
    plt.show()


def init_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    return model


def train_model(model, train_images, train_labels, epochs):
    # 训练模型
    model.fit(train_images, train_labels, epochs=epochs)


def evaluat_model(model, test_images, test_labels):
    # 评估模型
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    return test_loss, test_acc


def find_epochs(from_index, to_index, steps=1):
    print(f"find_epochs({from_index},{to_index},{steps})")
    train_images, train_labels, test_images, test_labels = init_datas()
    indexes = []
    test_losses = []
    test_accs = []
    for i in range(from_index, to_index, steps):
        model = init_model()
        train_model(model,
                    train_images,
                    train_labels,
                    epochs=i)
        test_loss, test_acc = evaluat_model(
            model,
            test_images,
            test_labels)

        test_losses.append(test_loss)
        test_accs.append(test_acc)
        indexes.append(i)

    save_datas(indexes, test_accs, test_losses, "./counter.csv")

    for i in range(len(indexes)):
        print(f"[{indexes[i]}] acc={test_accs[i]} loss={test_losses[i]}")


def train_and_save_model(filename):
    train_images, train_labels, test_images, test_labels = init_datas()

    # 训练模型
    model = init_model()
    model.fit(train_images, train_labels, epochs=10)
    model.save(filename)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"epochs=10, acc={test_acc} loss={test_loss}")


def load_and_evaluate(filename):
    _, _, test_images, test_labels = init_datas()
    model = keras.models.load_model(filename)
    model.summary()

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
    print(f"loaded, acc={test_acc} loss={test_loss}")


if __name__ == "__main__":
    print("fesion_model_main")
    # find_epochs(1,51,5) better is 6 and 11
    # find_epochs(1,16,1) best is 10
    # train_and_save_model(checkpoint_path)
    # load_and_evaluate(checkpoint_path)

    train_images, train_labels, test_images, test_labels = init_datas()
    show_first25_imgs(train_images)

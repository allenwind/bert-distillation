import glob
import os
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.applications import xception
import numpy as np
from distiller import Distiller

# 这里提供Distiller的使用例子
# 数据集下载：https://www.floydhub.com/fastai/datasets/cats-vs-dogs

# 下载数据后修改一下路径即可
_CD = "/home/zhiwen/workspace/dataset/fastai-datasets-cats-vs-dogs/"
def load_cats_vs_dogs(file=_CD, batch_size=32, image_shape=[256, 256]):
    train_dogs_pattern = file + "train/dogs/*.jpg"
    train_cats_pattern = file + "train/cats/*.jpg"
    valid_dogs_pattern = file + "valid/dogs/*.jpg"
    valid_cats_pattern = file + "valid/cats/*.jpg"

    def load_files(dogs_dir, cats_dir):
        dog_files = glob.glob(dogs_dir) # or use tf.data.Dataset.list_files
        cat_files = glob.glob(cats_dir)
        files = np.array(dog_files + cat_files)
        labels = np.array([0] * len(dog_files) + [1] * len(cat_files))
        np.random.seed(8899)
        np.random.shuffle(files)
        np.random.seed(8899) # 保证标签对齐
        np.random.shuffle(labels)
        labels = tf.keras.utils.to_categorical(labels)
        return files, labels

    def fn(filename, label):
        image_string = tf.io.read_file(filename)            # 读取原始文件
        image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
        image_resized = tf.image.resize(image_decoded, image_shape) / 255.0
        return image_resized, label

    files, labels = load_files(train_dogs_pattern, train_cats_pattern)
    train_dataset = tf.data.Dataset.from_tensor_slices((files, labels)) \
                             .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                             .shuffle(buffer_size=256) \
                             .batch(batch_size) \
                             .prefetch(tf.data.experimental.AUTOTUNE)

    # 验证集
    files, labels = load_files(valid_dogs_pattern, valid_cats_pattern)
    valid_dataset = tf.data.Dataset.from_tensor_slices((files, labels)) \
                                   .map(fn, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                   .batch(batch_size)
    return train_dataset, valid_dataset


train_dataset, valid_dataset = load_cats_vs_dogs(
    batch_size=32,
    image_shape=(299, 299)
)

# 教师网络
xception = xception.Xception(weights="imagenet")
xception.trainable = False
inputs = xception.input
pool = xception.layers[-2].output
outputs = Dense(2)(pool)
teacher = Model(inputs=inputs, outputs=outputs)

teacher.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)
# teacher.summary()
if not os.path.exists("cat_vs_dog_weights.index"):
    teacher.fit(train_dataset, epochs=1, validation_data=valid_dataset)
    teacher.evaluate(valid_dataset)
    teacher.save_weights("cat_vs_dog_weights")
else:
    teacher.load_weights("cat_vs_dog_weights")
# 评估教师网络 acc:0.9950
teacher.evaluate(valid_dataset)

# 学生网络
student = Sequential()
student.add(Conv2D(32, (3, 3), input_shape=(299, 299, 3)))
student.add(LeakyReLU(alpha=0.2))
student.add(BatchNormalization())
student.add(MaxPooling2D(pool_size=(2, 2)))
student.add(Dropout(0.25))

student.add(Conv2D(64, (3, 3)))
student.add(LeakyReLU(alpha=0.2))
student.add(BatchNormalization())
student.add(MaxPooling2D(pool_size=(2, 2)))
student.add(Dropout(0.25))

student.add(Conv2D(128, (3, 3)))
student.add(LeakyReLU(alpha=0.2))
student.add(BatchNormalization())
student.add(MaxPooling2D(pool_size=(2, 2)))
student.add(Dropout(0.25))

student.add(Flatten())
student.add(Dense(512, activation="relu"))
student.add(BatchNormalization())
student.add(Dropout(0.25))
student.add(Dense(2))

# 蒸馏
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
    student_loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss=tf.keras.losses.KLDivergence(),
    alpha=0.5,
    temperature=10
)
distiller.fit(train_dataset, epochs=3)
# 评估学生网络性能
distiller.evaluate(valid_dataset)
# student.compile
# student.evaluate(valid_dataset)

# 普通训练，用于对比
student2 = tf.keras.models.clone_model(student)
student2.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)
# student2.summary()
student2.fit(train_dataset, epochs=3)
# 0.7415
student2.evaluate(valid_dataset)

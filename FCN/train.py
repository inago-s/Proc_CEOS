from keras import metrics
from keras.metrics import categorical_accuracy
from util.FCN_model import FCN
from util.generator import Generator
from keras import optimizers
import numpy as np


FCN = FCN(256, 256, 10)
model = FCN.create_model()
sgd = optimizers.SGD(learning_rate=1e-6, momentum=0.9)
model.compile(optimizer=sgd, loss='categorical_crossentropy',
              metrics=['accuracy'])

with open('./train.txt') as f:
    image_path_list = f.read().split('\n')
    image_path_list.remove('')

with open('./class.txt') as f:
    class_path_list = f.read().split('\n')
    class_path_list.remove('')

val_count = int(len(image_path_list)*0.1)

train_gen = Generator(
    image_path_list[val_count:], class_path_list[val_count:], 256, 256, 2)

val_gen = Generator(
    image_path_list[:val_count], class_path_list[:val_count], 256, 256, 2)

model.fit_generator(train_gen, steps_per_epoch=train_gen.batches_per_epoch, validation_data=val_gen,
                    validation_steps=val_gen.batches_per_epoch, epochs=100, shuffle=True)

model.save('fcn_model.h5')

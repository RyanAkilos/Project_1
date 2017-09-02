import os, cv2
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder

train_data_dir = 'F://dpL//Defect_Least(200X160)//Training'
validation_data_dir = 'F://dpL//Defect_Least(200X160)//Validation'

img_rows = 160
img_cols = 200
num_channel = 3
epochs = 30
batch_size = 64
nb_train_samples = 33032
nb_validation_samples = 1997

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_rows, img_cols),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle='True')


validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_rows, img_cols),
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        shuffle='True')

# Model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(5, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=5,
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps= nb_validation_samples // batch_size)

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# Compile
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

# Train top 2 blocks
csv_logger = CSVLogger('F://dpL//long//InceptionV3_6.csv', separator=',', append=False)
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
TF_Board = TensorBoard(log_dir='F://dpL//long//InceptionV3_6', histogram_freq=1, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=epochs,
                    callbacks=[csv_logger, TF_Board],
                    verbose=1,
                    validation_data=validation_generator,
                    validation_steps= nb_validation_samples // batch_size)

#Confusion matrix
Y_pred = model.predict_generator(validation_generator, 63)
y_pred = np.argmax(Y_pred, axis=1)
target_names = ['Breaking', 'Crack', 'Deposition', 'Fracture', 'Hole']
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
print(confusion_matrix(validation_generator.classes, y_pred))



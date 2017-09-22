import os, cv2
import math
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.utils import np_utils
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.optimizers import SGD, RMSprop, adam, nadam
from keras.callbacks import EarlyStopping, CSVLogger, TensorBoard, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils import shuffle
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import interp

top_weights_path = 'F://dpL//long//top_model_weights_ResNet50_224X224.h5'
train_data_path = 'F://dpL//small-defect-dataset(224X224)//Training'
test_data_path = 'F://dpL//small-defect-dataset(224X224)//Validation'
data_dir_list = os.listdir(train_data_path)

print(data_dir_list)

img_rows = 224
img_cols = 224
num_channel = 5
top_epochs = 35
epochs = 35
batch_size = 32

# Define the number of classes
num_classes = 5

train_img_data_list = []
test_img_data_list = []

for dataset in data_dir_list:
    train_img_list = os.listdir(train_data_path + '/' + dataset)
    print('Loaded the images of train dataset-' + '{}\n'.format(dataset))
    test_img_list = os.listdir(test_data_path + '/' + dataset)
    print('Loaded the images of test dataset-' + '{}\n'.format(dataset))
    for train_img in train_img_list:
        train_input_img = cv2.imread(train_data_path + '/' + dataset + '/' + train_img)
        train_input_img_resize = cv2.resize(train_input_img, (img_rows, img_cols))
        train_img_data_list.append(train_input_img_resize)
    for test_img in test_img_list:
        test_input_img = cv2.imread(test_data_path + '/' + dataset + '/' + test_img)
        test_input_img_resize = cv2.resize(test_input_img, (img_rows, img_cols))
        test_img_data_list.append(test_input_img_resize)


train_img_data = np.array(train_img_data_list)
train_img_data = train_img_data.astype('float32')
train_img_data /= 255
print(train_img_data.shape)
test_img_data = np.array(test_img_data_list)
test_img_data = test_img_data.astype('float32')
test_img_data /= 255
print(test_img_data.shape)

# Assigning Labels
num_of_train_samples = train_img_data.shape[0]
train_labels = np.ones((num_of_train_samples,), dtype='int64')
train_labels[0:1000] = 0
train_labels[1000:2000] = 1
train_labels[2000:3000] = 2
train_labels[3000:4000] = 3
train_labels[4000:] = 4
print(num_of_train_samples)
num_of_test_samples = test_img_data.shape[0]
test_labels = np.ones((num_of_test_samples,), dtype='int64')
test_labels[0:400] = 0
test_labels[400:800] = 1
test_labels[800:1200] = 2
test_labels[1200:1600] = 3
test_labels[1600:] = 4
print(num_of_test_samples)

Y_train = np_utils.to_categorical(train_labels, num_classes)
x_train, y_train = shuffle(train_img_data, Y_train, random_state=2)
Y_test = np_utils.to_categorical(test_labels, num_classes)
x_test, y_test = shuffle(test_img_data, Y_test, random_state=2)

# Model
base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(5, activation='softmax')(x)

# Train top model
model = Model(inputs=base_model.input, outputs=predictions)
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

check_point = ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True)
top_early_stopping = EarlyStopping(monitor='val_acc', patience=4, verbose=0)
chart = TensorBoard(log_dir='F://dpL//long//ResNet50_top_model', batch_size=batch_size, write_graph=True)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=top_epochs,
          callbacks=[check_point, top_early_stopping, chart],
          verbose=1,
          validation_data=(x_test, y_test))

# Freeze layer
model.load_weights(top_weights_path)
print('Model weights loaded')
for layer in model.layers[:173]:
    layer.trainable = False
for layer in model.layers[173:]:
    layer.trainable = True

# Compile
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fine-tuning
csv_logger = CSVLogger('F://dpL//long//ResNet50_with_top_model', separator=',', append=False)
early_stopping = EarlyStopping(monitor='val_acc', patience=4)
Final_Check_point = ModelCheckpoint('F://dpL//tensorflow//ResNet50_224X224.h5', monitor='val_acc', verbose=1, save_best_only=True)
T_Board = TensorBoard(log_dir='F://dpL//long//ResNet50_with_top_model', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

result = model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   callbacks=[csv_logger, Final_Check_point],
                   verbose=1,
                   validation_data=(x_test, y_test))


# Confusion matrix
Best_model = load_model('F://dpL//tensorflow//ResNet50_224X224.h5')
Y_pred = Best_model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report:')
class_report = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=data_dir_list)
print(class_report)
print('Confusion Matrix:')
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
print(conf_matrix)

# Micro-average ROC curve and ROC area
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], Y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), Y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(num_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= num_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure()
lw = 2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['red', 'darkorange', 'gold', 'lime', 'cornflowerblue'])
for i, color, classes in zip(range(num_classes), colors, data_dir_list):
    plt.plot(fpr[i], tpr[i], color=color,
             lw=lw, label=classes + ' (area = %0.2f)' % roc_auc[i])

plt.plot([0, 1], [0, 1], 'k--', color='navy', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True positive rate')
plt.title('Multi-class ROC curve')
plt.legend(loc="lower right")
plt.savefig('F://dpL//long//ResNet50_with_top_model_dropout.png')

# Training history
plt.plot(result.history['acc'], color='orange', lw=2)
plt.plot(result.history['val_acc'], color='red', lw=2)
plt.ylim([0.00, 1.00])
plt.title('Model accurcy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('F://dpL//long//ResNet50_with_top_modelAccuracy.png')

plt.plot(result.history['loss'], color='azure', lw=2)
plt.plot(result.history['val_loss'], color='aqua', lw=2)
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('F://dpL//long//ResNet50_with_top_modelLoss.png')

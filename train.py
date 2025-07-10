import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array

# 训练集目录路径
train_data_path = "E:/dataset_0709/building/train"

# 验证集目录路径
validate_data_path = "E:/dataset_0709/building/val"


def mobilenet_model():
    mobile = MobileNet(include_top=False, weights='imagenet', input_shape=[224, 224, 3])
    for layer in mobile.layers:
        layer.trainable = False
    last = mobile.output
    x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(last)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(26, activation='softmax')(x)
    model = Model(inputs=mobile.input, outputs=x)
    return model


def pred_data(test_image_folder_path):
    model = load_model('my_model.h5')
    # with open('./building.yaml') as yamlfile:
    #     loaded_model_yaml = yamlfile.read()
    # model = model_from_yaml(loaded_model_yaml)
    # model.load_weights('./building.h5')

    sgd = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    total_image_count = float(0)
    predicted_count = float(0)
    for f in os.listdir(test_image_folder_path):
        img = load_img(test_image_folder_path + f, target_size=(224, 224))
        img_array = img_to_array(img)
        x = np.expand_dims(img_array, axis=0)
        x = preprocess_input(x)
        result = model.predict(x, verbose=0)
        result = np.argmax(result)
        total_image_count += 1

        if result >= 0 and result < 78:
            predicted_count += 1

        print(f, result)

    accurate_rate = predicted_count / total_image_count
    print(f"识别准确率: {accurate_rate:.2f}")


font1 = {'family': 'Times New Roman', 'color': 'black', 'weight': 'normal', 'size': 40}


def training_vis(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']

    # make a figure
    fig = plt.figure(figsize=(20, 10))
    # subplot loss
    ax1 = fig.add_subplot(121)
    ax1.plot(loss, label='train_loss', linewidth=5.0)
    ax1.plot(val_loss, label='val_loss', linewidth=5.0)
    ax1.set_xlabel('Epochs', fontdict=font1)
    ax1.set_ylabel('Loss', fontdict=font1)
    ax1.set_title('Loss on Training and Validation Data', fontsize=40)
    ax1.legend()
    # subplot acc
    ax2 = fig.add_subplot(122)
    ax2.plot(acc, label='train_acc', linewidth=5.0)
    ax2.plot(val_acc, label='val_acc', linewidth=5.0)
    ax2.set_xlabel('Epochs', fontdict=font1)
    ax2.set_ylabel('Accuracy', fontdict=font1)
    ax2.set_title('Accuracy  on Training and Validation Data', fontsize=40)
    ax2.legend()
    plt.tight_layout()
    plt.show()


train_datagen = ImageDataGenerator(
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.3,
    rescale=1. / 255,
    fill_mode='nearest',
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validate_data_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')


# 初始基础训练（冻结所有预训练层）
def trainingV1():
    # model=load_model('my_model.h5')
    # 加载原模型（保留预训练权重）
    base_model = load_model('my_model.h5')

    # 移除原输出层（保留除最后一层外的所有层）
    x = base_model.layers[-2].output  # 获取倒数第二层输出

    # 添加新的78类输出层
    new_output = Dense(78, activation='softmax', name='new_output')(x)

    # 构建新模型
    model = Model(inputs=base_model.input, outputs=new_output)

    print(model.input_shape)
    # model = mobilenet_model()
    sgd = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 训练前再次验证维度
    print(f"模型输出维度: {model.output_shape}")
    print(f"数据标签维度: {train_generator.num_classes}")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=132,
        epochs=50,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=32
    )

    json_string = model.to_json()
    with open('./building.json', 'w') as outfile:
        outfile.write(json_string)

    model.save_weights('./building.h5')
    model.save('my_model.h5')

    training_vis(history)


# 解冻顶层预训练层,适用于初步微调，提升模型对建筑物全局结构的识别能力。
def trainingV2():
    model = load_model('my_model.h5')
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # plot_model(model, to_file='model.png')
    for layer in model.layers[:67]:
        layer.trainable = False
    for layer in model.layers[68:]:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    # 训练前再次验证维度
    print(f"模型输出维度: {model.output_shape}")
    print(f"数据标签维度: {train_generator.num_classes}")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=132,
        epochs=100,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=32
    )

    json_string = model.to_json()
    with open('./building.json', 'w') as outfile:
        outfile.write(json_string)

    model.save_weights('./building.h5')
    model.save('my_model.h5')

    training_vis(history)


# 解冻中层预训练层,适用于细化局部特征，区分外观相似的建筑物。
def trainingV3():
    model = load_model('my_model.h5')
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # plot_model(model, to_file='model.png')
    for layer in model.layers[:49]:
        layer.trainable = False
    for layer in model.layers[50:67]:
        layer.trainable = True

    for layer in model.layers[68:]:
        layer.trainable = False

    # for layer in model.layers[68:86]:
    #     layer.trainable = False
    # for layer in model.layers[87:]:
    #     layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    # 训练前再次验证维度
    print(f"模型输出维度: {model.output_shape}")
    print(f"数据标签维度: {train_generator.num_classes}")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=132,
        epochs=50,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=32
    )

    json_string = model.to_json()
    with open('./building.json', 'w') as outfile:
        outfile.write(json_string)

    model.save_weights('./building.h5')
    model.save('my_model.h5')

    training_vis(history)


# 解冻中高层预训练层,适用于平衡全局与局部特征，适合复杂场景。
def trainingV4():
    model = load_model('my_model.h5')
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # plot_model(model, to_file='model.png')
    for layer in model.layers[:61]:
        layer.trainable = False
    for layer in model.layers[62:67]:
        layer.trainable = True

    for layer in model.layers[68:]:
        layer.trainable = False

    # for layer in model.layers[68:86]:
    #     layer.trainable = False
    # for layer in model.layers[87:]:
    #     layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
    # 训练前再次验证维度
    print(f"模型输出维度: {model.output_shape}")
    print(f"数据标签维度: {train_generator.num_classes}")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=132,
        epochs=50,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=32
    )

    json_string = model.to_json()
    with open('./building.json', 'w') as outfile:
        outfile.write(json_string)

    model.save_weights('./building.h5')
    model.save('my_model.h5')

    training_vis(history)


def show():
    model = load_model('my_model.h5')
    # model = mobilenet_model()
    # model = MobileNet(include_top=True, weights ='imagenet', input_shape=(224,224,3))
    model.summary()
    for i, layer in enumerate(model.layers):
        print(i, layer.name)


if __name__ == '__main__':
    test_image_folder_path = "E:/dataset_0709/test_images/"

    trainingV1()
    pred_data(test_image_folder_path)

    trainingV2()
    pred_data(test_image_folder_path)

    trainingV3()
    pred_data(test_image_folder_path)

    trainingV4()
    pred_data(test_image_folder_path)

    show()

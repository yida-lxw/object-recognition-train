import os

from generate_sub_folder_list import generate_sub_folder_map
from utils.file_utils import FileUtils
from utils.string_utils import StringUtils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.applications.mobilenet import preprocess_input
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.utils import load_img, img_to_array

project_basepath = StringUtils.get_project_basepath()
project_basepath = StringUtils.replaceBackSlash(project_basepath)
project_basepath = StringUtils.to_ends_with_back_slash(project_basepath)


def pred_data(test_image_folder_path, sub_folder_map: dict, min_confidence: float = 0.70):
    test_image_folder_path = StringUtils.replaceBackSlash(test_image_folder_path)
    test_image_folder_path = StringUtils.to_ends_with_back_slash(test_image_folder_path)

    model_path = project_basepath + 'my_model.h5'
    model = load_model(model_path)
    sgd = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    for file_name in os.listdir(test_image_folder_path):
        image_file_name = FileUtils.get_file_name_without_suffix(file_name)
        img = load_img(test_image_folder_path + file_name, target_size=(224, 224))
        img_array = img_to_array(img)
        x = np.expand_dims(img_array, axis=0)
        x = preprocess_input(x)
        predictions = model.predict(x, verbose=0)
        confidence = np.max(predictions)
        if confidence < min_confidence:
            print(f"image_file_name:{file_name}, confidence:{confidence}")
            class_id = -1
            class_name = "unknown"
        else:
            class_id = np.argmax(predictions)
            class_name = sub_folder_map[class_id]
        print(
            f"image_file_name:{image_file_name}, confidence:{confidence}, class_id:{class_id}, class_name:{class_name}")


if __name__ == '__main__':
    target_sub_folder_file_path = f"{project_basepath}docs/sub_folder_list.txt"
    test_image_folder_path = "E:/dataset_0709/test_images/"
    sub_folder_map = generate_sub_folder_map(target_sub_folder_file_path)
    pred_data(test_image_folder_path, sub_folder_map)

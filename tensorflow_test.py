import tensorflow

if __name__ == '__main__':
    tensorflow_version = tensorflow.__version__
    print(f"Tensorflow Version:{tensorflow_version}")

    gpu_avaliable = tensorflow.test.is_gpu_available()
    print(f"GPU avaliable:{gpu_avaliable}")

    if tensorflow.test.gpu_device_name():
        print("GPU:", tensorflow.test.gpu_device_name())
    else:
        print("No GPU")

    config_list = tensorflow.config.list_physical_devices("GPU")
    if config_list is None or len(config_list) <= 0:
        print("No GPU")
    else:
        for config in config_list:
            device_name = config["name"]
            print(f"GPU:{device_name}")

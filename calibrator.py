import tensorrt as trt
import os

import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import numpy as np


def data_Normalize(batch_data, mean_list=[0.485, 0.456, 0.406], std_list=[0.229, 0.224, 0.225]):
    batch_data[0, :, :] = (batch_data[0, :, :] - mean_list[0]) / std_list[0]
    batch_data[1, :, :] = (batch_data[1, :, :] - mean_list[1]) / std_list[1]
    batch_data[2, :, :] = (batch_data[2, :, :] - mean_list[2]) / std_list[2]
    return batch_data


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_folder_path, cache_file, total_images, batch_size=1):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8EntropyCalibrator2.__init__(self)
        
        # Prepare data for the the following
        self.calib_folder_path = calib_folder_path
        file_list = os.listdir(self.calib_folder_path)
        self.image_path_generator = iter(file_list)
        self.cache_file = cache_file
        self.batch_size = batch_size
        # Allocate enough memory for a whole batch.
        # self.device_input = cuda.mem_alloc(self.data.nbytes * self.batch_size)
        self.device_input = cuda.mem_alloc(800*800*8*3 * self.batch_size)

    
    def get_batch_size(self):
        return self.batch_size


    def get_batch(self, names):
        try:
            image_path = next(self.image_path_generator)
            filename = self.calib_folder_path + image_path
            print(filename)
            image_array = np.asarray(Image.open(filename))
            #print(image_array.shape)
            #print(len(image_array.shape))
            if len(image_array.shape) != 3:
                print(filename, 'is not a RGB image. Copy one channel to three channels')
                print('orignal shape: ', image_array.shape)
                rgb_image = np.zeros((image_array.shape[0], image_array.shape[1], 3))
                rgb_image[:,:,0] = image_array
                rgb_image[:,:,1] = image_array
                rgb_image[:,:,2] = image_array
                image_array = rgb_image
                print('converted shape: ', image_array.shape)
            image_array = image_array.transpose(2, 0, 1)/ 255.0
            #image_array = image_array.astype(np.float32)
            image_array = data_Normalize(image_array)
            image_array = np.ascontiguousarray(image_array.astype(np.float32))
            #data = image_array.ravel()
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, image_array)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None
        

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

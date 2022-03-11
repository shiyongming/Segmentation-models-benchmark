import logging
import os
import sys
import time
import ctypes
import argparse
import numpy as np
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit

from PIL import Image

import torch
import torchvision
import torchvision.transforms.functional as F
from torchvision.transforms.functional import convert_image_dtype
#from image_batcher import ImageBatcher
#from visualize import visualize_detections


logging.basicConfig(level=logging.ERROR)  # INFO, WARNING, ERROR
logging.getLogger("EngineInference").setLevel(logging.ERROR)
log = logging.getLogger("EngineInference")



class TensorRTInfer:
    def __init__(self, engine_path, shape):
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0
        self.context.set_binding_shape(0, (1, shape[0], shape[1], shape[2]))
        assert self.engine
        assert self.context
            
        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            #print("The binding name is:", name)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.context.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)
                
    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        print("The input name is:", self.inputs[0]['name'])
        print("The input shape is:", self.inputs[0]['shape'])
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs
        
        
    def infer(self, batch, shape):
        
        """
        Execute inference on a batch of images. The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch. Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros((shape), dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        log.info("Start TRT engine")
        self.context.execute_v2(self.allocations)
        log.info("End TRT engine")
        for o in range(len(outputs)):
            log.info("Number {:} output's name is:{:}".format(o, self.outputs[o]['name']))
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])  
        return outputs


def inference(args_engine, args_input):
    log.info("Start inferencing")
    """
    PreProcessing
    """
    #batch = Image.open(args.input)
    #batch = batch.convert(mode='RGB')
    #batch = np.transpose(batch, (2, 0, 1))
    batch_int = torchvision.io.read_image(args_input)
    batch = convert_image_dtype(batch_int, dtype=torch.float)
    #print(batch.shape, batch.min().item(), batch.max().item())
    normalized_batch = F.normalize(batch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    normalized_batch_numpy = batch.unsqueeze(0).numpy()
    #print(normalized_batch_numpy.shape, normalized_batch_numpy.min().item(), normalized_batch_numpy.max().item())
    
    """
    TensorRT inference
    """
    shape = (normalized_batch.shape)
    trt_infer = TensorRTInfer(args_engine, shape)
    infer_results = trt_infer.infer(normalized_batch, shape)
    
    """
    PostPprocessing
    """
    infer_resulrs_tensor = torch.tensor(np.array(infer_results))
    log.info("Inference result shape:{:}, min:{:}, max:{:}".format(infer_resulrs_tensor.shape, infer_resulrs_tensor.min().item(), infer_resulrs_tensor.max().item()))
    
    dense_out = infer_resulrs_tensor[0]
    log.info("The dense_out shape:{:}, min:{:}, max:{:}".format(dense_out.shape, dense_out.min().item(), dense_out.max().item()))
    
    normalized_masks = torch.nn.functional.softmax(dense_out, dim=1)
    log.info("Normalized_masks shape:{:}, min:{:}, max:{:}".format(normalized_masks.shape, normalized_masks.min().item(), normalized_masks.max().item()))
    log.info("Finish inferencing")
    return normalized_masks

def eval_inference(engine, img):
    log.info("Start inferencing")
    """
    PreProcessing
    """
    # PreProcessing was finished in the dataset loader.
    
    """
    TensorRT inference
    """
    shape = (img.shape)
    trt_infer = TensorRTInfer(engine, shape)
    infer_results = trt_infer.infer(img, shape)
    
    """
    PostPprocessing
    """
    infer_resulrs_tensor = torch.tensor(np.array(infer_results))
    dense_out = infer_resulrs_tensor[0]
    normalized_masks = torch.nn.functional.softmax(dense_out, dim=1)

    return normalized_masks


def show_masks(imgs, normalized_masks, args, alpha=0.7):    
    sem_classes = [
        '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
#    selected_masks = [
#        normalized_masks[img_idx, sem_class_to_idx[cls]]
#        for img_idx in range(imgs.shape[0])
#        for cls in ('bottle', 'chair', 'person', 'pottedplant', 'tvmonitor')
#    ]
    selected_masks = [
         normalized_masks[img_idx, sem_class_to_idx[cls]]
         for img_idx in range(imgs.shape[0])
         for cls in sem_classes
     ]
    #class_dim = 1
    if args.selected_class is not None:
    	class_dim = 1
    	boolean_masks = (normalized_masks.argmax(class_dim) == sem_class_to_idx[args.selected_class])
    else:
    	class_dim = 0
    	boolean_masks = [normalized_masks[img_idx].argmax(class_dim) == torch.arange(21)[:,None,None] for img_idx in range(imgs.shape[0])]
    	#color_masks = torch.zeros(normalized_masks.shape[0], normalized_masks.shape[2], normalized_masks.shape[3])
    	#for img_idx in range(imgs.shape[0]):
    	#    for cls_idx in range(boolean_masks[img_idx].shape[0]):
    	#    	for i in range(color_masks[img_idx].shape[0]):
    	#    	    for j in range(color_masks[img_idx].shape[1]):
    	#    	    	if boolean_masks[img_idx][cls_idx, i, j]:
    	#    	    	    color_masks[img_idx, i, j] = boolean_masks[img_idx][cls_idx, i, j].int() * cls_idx
    	#
    	#torchvision.utils.save_image(color_masks[0]/255.0, args.output+'boolean_masks.png')
    	#log.info("Saved masks into {:}".format(args.output+'boolean_masks.png'))
    
    	
    from torchvision.utils import draw_segmentation_masks
    imgs_with_masks = [
        draw_segmentation_masks(img, masks=mask, alpha=0.6)
        for img, mask in zip(imgs, boolean_masks)
    ]
    #imgs_with_masks = draw_segmentation_masks(imgs[0], masks=boolean_masks, alpha=0.9)
    torchvision.utils.save_image(imgs_with_masks[0]/255.0, args.output+'results_with_masks.png')
    
    log.info("Saving results")
    log.info("Saved img_with_masks into {:}".format(args.output+'results_with_masks.png'))


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = torch.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true, ignore_backgournd=True):
        if ignore_backgournd:
            mask = (label_true > 0) & (label_true < self.num_classes)
        else: 
            mask = (label_true >= 0) & (label_true < self.num_classes)
            #mask = label_true != 0
        hist = torch.bincount(
            self.num_classes * label_true[mask] + label_pred[mask], 
            minlength=self.num_classes ** 2
            ).view(self.num_classes, self.num_classes)

        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc 

eval_mode = True
timing_only = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", default=None, help="The serialized TensorRT engine")
    parser.add_argument("-i", "--input", default='', help="Path to the image or directory to process")
    parser.add_argument("-c", "--selected_class", type=str, default=None, help="Class to be selected for segmentation")
    parser.add_argument("-o", "--output", default='', help="Directory where to save the visualization results")
#    parser.add_argument("-l", "--labels", default="./labels_coco.txt", help="File to use for reading the class labels "
#                                                                            "from, default: ./labels_coco.txt")
#    parser.add_argument("-t", "--nms_threshold", type=float, help="Override the score threshold for the NMS operation, "
#                                                                  "if higher than the threshold in the engine.")
    args = parser.parse_args()
#    if not all([args.engine, args.input, args.output]):
#        parser.print_help()
#        print("\nThese arguments are required: --engine  --input and --output")
#        sys.exit(1)
    
    
    if eval_mode:
    	import time
    	from load_gt import voc_reader
    	
    	voc_data = voc_reader(1, 1)
    	total_samples = voc_data.n_val_steps_per_epoch
    	metrics = IOUMetric(21)
    	T1 = time.time()
    	for i in range(voc_data.n_val_steps_per_epoch):
    	#for i in range(100):
    	    #if i%10 == 0:
    	    #	print('Finished {:} steps'.format(i))
    	    	
    	    val_img, val_label, val_label_onehot = voc_data.next_val_batch()
    	    val_label_tensor = torch.tensor(val_label).int()
    	    #val_label_onrhot = torch.tensor(val_label_onrhot)
    	     
    	    if val_img.shape[1] > 256 and val_img.shape[1] < 512 and val_img.shape[2] > 256 and val_img.shape[2] < 512:
    	    	infer_prob = eval_inference(args.engine, val_img)
    	    	preds = torch.argmax(infer_prob, dim=1)
    	    	if not timing_only:
    	    	    metrics.add_batch(preds[0], val_label_tensor[0])
    	    
    	    else:
    	    	#log.warning("Found one img with height ot width <256 or >512, total_numbers-1")
    	    	total_samples = total_samples - 1
    	    	continue   	

    	T2 = time.time()
    	time = T2 - T1

    	print('Time used:', time, 's')
    	print('Sample precessed:', (total_samples))
    	print('FPS:', (total_samples)/time)
    	print('')
    	
    	if not timing_only:
    	    acc, acc_cls, iu, mean_iu, fwavacc = metrics.evaluate()
    	    print('MIoU is:', mean_iu)
    	    print('acc is:', acc.item())
    	    print('acc_cls is:', acc_cls)
    
    
    else:
    	normalized_masks = inference(args)
    	imgs = torchvision.io.read_image(args.input)
    	imgs = imgs.unsqueeze(0)
    	show_masks(imgs, normalized_masks, args)




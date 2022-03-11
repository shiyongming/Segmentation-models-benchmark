import logging
import tensorrt as trt
from calibrator import EntropyCalibrator
import common


def build_int8_engine(onnx_file_path, calibrator, batch_size, calibration_cache):
    # with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network,TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size

        config.max_workspace_size = common.GiB(1)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        config.int8_calibrator = calibrator

        # Parse Onnx model
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # For the fixed batch, please use the following code
        #network.get_input(0).shape = [batch_size, 3, W, H]
        
        # For dynamic batch, please use the following code
        profile = builder.create_optimization_profile();
        profile.set_shape("input0", (1, 3, 256, 256), (1, 3, 512, 512), (1, 3, 640, 640))
        config.add_optimization_profile(profile)
        config.set_calibration_profile(profile)


        # Start to build engine and do int8 calibration.
        print('--- Starting to build engine! ---')
        engine = builder.build_engine(network, config)
        print('--- Building engine is finished! ---')

        return engine


def build_engine(onnx_file_path, precision, batch_size):
    # with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, builder.create_builder_config() as config, trt.CaffeParser() as parser:
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, trt.OnnxParser(network,TRT_LOGGER) as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = batch_size

        config.max_workspace_size = common.GiB(1)
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        # Parse Onnx model
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # For the fixed batch, please use the following code
        #network.get_input(0).shape = [batch_size, 3, W, H]
        
        # For dynamic batch, please use the following code
        profile = builder.create_optimization_profile();
        profile.set_shape("input0", (1, 3, 256, 256), (batch_size, 3, 512, 512), (batch_size, 3, 640, 640))
        config.add_optimization_profile(profile)


        print('--- Starting to build engine! ---')
        engine = builder.build_engine(network, config)
        print('--- Building engine is finished! ---')

        return engine


#model = "fcn_resnet50"
#model = "fcn_resnet101"
#model = "deeplabv3_resnet50"
#model = "deeplabv3_resnet101"
#model = 'deeplabv3_mobilenet_v3_large'
model = 'lraspp_mobilenet_v3_large'

ONNX_PATH = 'onnx_models/' + model + ".onnx"
calib_data_path = 'calibration_data/'

#TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_LOGGER = trt.Logger()      

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")

#precision = 'fp32'
precision = 'fp16'
#precision = 'int8'

batch_size = 1 # This is inference batch size that can be different from calibration batch size.
engine_path = 'engines/' + model + '_' + precision + ".trt"

if precision == 'int8':
    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    calibration_cache = 'calibcache/' + model + '_' + "calibration.cache"
    calibrator = EntropyCalibrator(calib_data_path, cache_file=calibration_cache, total_images=300, batch_size=1)
    with build_int8_engine(ONNX_PATH, calibrator, batch_size, calibration_cache) as engine, open(engine_path, "wb") as f:
        log.info("Serializing engine to file: {:}".format(engine_path))
        f.write(engine.serialize())
else:
    with build_engine(ONNX_PATH, precision, batch_size) as engine, open(engine_path, "wb") as f:
        log.info("Serializing engine to file: {:}".format(engine_path))
        f.write(engine.serialize())
        print('finished!!!')



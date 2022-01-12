
    
from tensorflow import (where, cond, concat,
                    int32, float32, int64, cast, gather_nd, multiply, tile, constant, reshape)
from tensorflow.data import TFRecordDataset
from tensorflow.math import reduce_all
import config
import feature_description as fd
import feature_extraction_test as fd_test

def get_dataset_segment(filename):
    dataset = TFRecordDataset(filename)
    dataset = dataset.map(fd.parse)
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset
def get_test_dataset_segment(filename):
    dataset = TFRecordDataset(filename)
    dataset = dataset.map(fd_test.parse)
    dataset = dataset.batch(config.BATCH_SIZE)
    return dataset
def get_shifted_data(origin, data, shape):
    # origin [agents,2]
    # data [agents, 11, 2]
    tiles = tile(origin[:], constant([1,shape[1]], int32))
    tiles = reshape(tiles,[shape[0], shape[1], 2])
    return data[:, :, :2] - tiles

def get_original_from_shift(origin, data, shape):
    tiles = tile(origin[:], constant([1,shape[1]], int32))
    tiles = reshape(tiles,[shape[0], shape[1], 2])
    return data[:, :, :2] + tiles

def get_lights_shifted_data(origin, lights_data):
    data = []
    for _ in range(origin.shape[0]):
        data.append(reshape(lights_data,[1,11,2]))
    data_tensor = concat(data,0)
    return get_shifted_data(origin, data_tensor, data_tensor.shape)

def insert_lights_state(input, light_state):
    data = []
    for i in range(input.shape[0]):
        data.append(cast(reshape(light_state,[1,11,1]),float32))
    data_tensor = concat(data,0)
    return concat([input, data_tensor],2)

def get_batch_inputs(raw_input, raw_target, valid_inputs):

    idx = where(reduce_all(valid_inputs, axis=1))
    valid_input = gather_nd(raw_input[:], idx)
    valid_target = gather_nd(raw_target[:], idx)
    valid_origin = gather_nd(raw_input[:, 0, :2], idx)
    '''
    Move to Origin
    '''
    #shifted_input = get_shifted_data(valid_origin,valid_input, valid_input.shape )
    shifted_input = concat([get_shifted_data(valid_origin,valid_input, valid_input.shape ),valid_input[:, :, 2:] ],2)
    shifted_target = get_shifted_data(valid_origin,valid_target, valid_target.shape )
    return shifted_input, shifted_target, valid_origin

def get_all_validate_inputs(raw_input):
    valid_input = raw_input
    valid_origin = raw_input[:, 0, :2]
    '''
    Move to Origin
    '''
    #shifted_input = get_shifted_data(valid_origin,valid_input, valid_input.shape )
    shifted_input = concat([get_shifted_data(valid_origin,valid_input, valid_input.shape ),valid_input[:, :, 2:] ],2)
    return shifted_input, valid_origin

def get_valid_inputs(data, valid_inputs):
    idx = where(reduce_all(valid_inputs, axis=1))
    valid_input = gather_nd(data[:], idx)
    return valid_input

def get_batch_inputs_with_lights(input, origin, lights_coord, lights_states, valid_lights):
    # have a [1, 16, 11, 2]
    valid_lights = reshape(valid_lights, [1, 16])
    input_shape = input.shape
    valid_lights_coord = get_valid_inputs(lights_coord, valid_lights)
    if valid_lights_coord.shape[0] == 0:
        ret = concat([input, constant(0, dtype=float32, shape=[input_shape[0], input_shape[1], 16*3])],2 )
        return ret

    valid_lights_coord = reshape(valid_lights_coord, [11, 2 * valid_lights_coord.shape[0]])
    shifted_coord = get_lights_shifted_data(origin,valid_lights_coord )
    valid_lights_states = get_valid_inputs(lights_states, valid_lights)
    # it may reduce to zero if non valid
    input_with_lights = insert_lights_state(shifted_coord,valid_lights_states[:,:,:1])
    
    ret = concat([input, input_with_lights],2 )
    ret = concat([ret, constant(0, dtype=float32, shape=[input_shape[0], input_shape[1], (16*3 - ret.shape[2] + config.num_of_features +1)])],2 )
    return ret



def get_actual_predictions(prediction, targets, origin):
    # for this problem prediction.shape = targets.shape = [80,2]
    # shift back
    return get_original_from_shift(origin, prediction, prediction.shape), get_original_from_shift(origin, targets, prediction.shape)
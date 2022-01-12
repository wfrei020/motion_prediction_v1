
from tensorflow.io import parse_single_example, FixedLenFeature
from tensorflow import stack, concat, reduce_any, float32, int64, string, transpose, tile, int32, constant, reshape
import sys
import config


def parse(value):
    decoded_example = parse_single_example(value, features_description)
    past_states = stack([
        decoded_example['state/past/x'], decoded_example['state/past/y'],
        decoded_example['state/past/speed'],
        decoded_example['state/past/vel_yaw'],
        decoded_example['state/past/velocity_x'],
        decoded_example['state/past/velocity_y'],
        decoded_example['state/past/bbox_yaw'],
    ], -1)

    cur_states = stack([
        decoded_example['state/current/x'], decoded_example['state/current/y'],
        decoded_example['state/current/speed'],
        decoded_example['state/current/vel_yaw'],
        decoded_example['state/current/velocity_x'],
        decoded_example['state/current/velocity_y'],
        decoded_example['state/current/bbox_yaw'],
    ], -1)
    input_states = concat([past_states, cur_states], 1)[..., :config.num_of_features]
    #print(input_states.shape)
    s_type = decoded_example['state/type'][:]
    tiles = tile(reshape(s_type,[s_type.shape[0],1]), constant([1, input_states.shape[1]], int32))
    tiles = reshape(tiles, [tiles.shape[0], tiles.shape[1], 1])
    input_states = concat([input_states, tiles], 2)
    # print(f'{input_states}')
    # quit()
    future_states = stack([
        decoded_example['state/future/x'], decoded_example['state/future/y'],
        decoded_example['state/future/speed'],
        decoded_example['state/future/vel_yaw'],
        decoded_example['state/future/velocity_x'],
        decoded_example['state/future/velocity_y'],
        decoded_example['state/future/bbox_yaw']
    ], -1)


    gt_future_states = concat([past_states, cur_states, future_states], 1)

    past_is_valid = decoded_example['state/past/valid'] > 0
    current_is_valid = decoded_example['state/current/valid'] > 0
    future_is_valid = decoded_example['state/future/valid'] > 0
    gt_future_is_valid = concat(
        [past_is_valid, current_is_valid, future_is_valid], 1)
    # If a sample was not seen at all in the past, we declare the sample as
    # invalid.
    sample_is_valid = reduce_any(
        concat([past_is_valid, current_is_valid], 1), 1)
    # TRAFFIC LIGHTS FEATURES
    lights_past_coord = stack([
        transpose(decoded_example['traffic_light_state/past/x']), transpose(decoded_example['traffic_light_state/past/y']),
    ], -1)

    lights_cur_coord = stack([
        transpose(decoded_example['traffic_light_state/current/x']), transpose(decoded_example['traffic_light_state/current/y']),
    ], -1)
    lights_past_states = stack([
        transpose(decoded_example['traffic_light_state/past/state']),
        transpose(decoded_example['traffic_light_state/past/id']),
    ], -1)

    lights_cur_states = stack([
        transpose(decoded_example['traffic_light_state/current/state']),
        transpose(decoded_example['traffic_light_state/current/id']),
    ], -1)

    lights_input_coord = concat([lights_past_coord, lights_cur_coord], 1)[..., :2]
    lights_input_states = concat([lights_past_states, lights_cur_states], 1)[..., :2]

    lights_past_is_valid = transpose(decoded_example['traffic_light_state/past/valid']) > 0
    lights_current_is_valid = transpose(decoded_example['traffic_light_state/current/valid']) > 0

    # If a sample was not seen at all in the past, we declare the sample as
    # invalid.
    lights_sample_is_valid = reduce_any(
        concat([lights_past_is_valid, lights_current_is_valid], 1), 1)


    #test = decoded_example['state/tracks_to_predict']
    #print(test > 0)
    inputs = {
        'input_states': input_states,
        'gt_future_states': gt_future_states,
        'gt_future_is_valid': gt_future_is_valid,
        'object_type': decoded_example['state/type'],
        'object_id': decoded_example['state/id'],
        'object_is_sdc': decoded_example['state/is_sdc'],
        'tracks_to_predict': decoded_example['state/tracks_to_predict'] > 0,
        'sample_is_valid': sample_is_valid,
        'lights_input_coord': lights_input_coord,
        'lights_input_states': lights_input_states,
        'lights_is_valid': lights_sample_is_valid,
        'road_xyz':decoded_example['roadgraph_samples/xyz'],
        'road_type':decoded_example['roadgraph_samples/type'],
    }
    return inputs


#
scenario = {
    'scenario/id':
    FixedLenFeature([1, 1], string, default_value=None),
}
# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        FixedLenFeature([20000, 3], float32, default_value=None),
    'roadgraph_samples/id':
        FixedLenFeature([20000, 1], int64, default_value=None),
    'roadgraph_samples/type':
        FixedLenFeature([20000, 1], int64, default_value=None),
    'roadgraph_samples/valid':
        FixedLenFeature([20000, 1], int64, default_value=None),
    'roadgraph_samples/xyz':
        FixedLenFeature([20000, 3], float32, default_value=None),
}

# Features of other agents.
state_features = {
    'state/id':
        FixedLenFeature([128], float32, default_value=None),
    'state/type':
        FixedLenFeature([128], float32, default_value=None),
    'state/is_sdc':
        FixedLenFeature([128], int64, default_value=None),
    'state/tracks_to_predict':
        FixedLenFeature([128], int64, default_value=None),
    'state/current/bbox_yaw':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/height':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/length':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/timestamp_micros':
        FixedLenFeature([128, 1], int64, default_value=None),
    'state/current/valid':
        FixedLenFeature([128, 1], int64, default_value=None),
    'state/current/vel_yaw':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/velocity_x':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/velocity_y':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/speed':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/width':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/x':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/y':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/current/z':
        FixedLenFeature([128, 1], float32, default_value=None),
    'state/future/bbox_yaw':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/height':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/length':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/timestamp_micros':
        FixedLenFeature([128, 80], int64, default_value=None),
    'state/future/valid':
        FixedLenFeature([128, 80], int64, default_value=None),
    'state/future/vel_yaw':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/velocity_x':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/velocity_y':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/speed':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/width':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/x':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/y':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/future/z':
        FixedLenFeature([128, 80], float32, default_value=None),
    'state/past/bbox_yaw':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/height':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/length':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/timestamp_micros':
        FixedLenFeature([128, 10], int64, default_value=None),
    'state/past/valid':
        FixedLenFeature([128, 10], int64, default_value=None),
    'state/past/vel_yaw':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/velocity_x':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/velocity_y':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/speed':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/width':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/x':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/y':
        FixedLenFeature([128, 10], float32, default_value=None),
    'state/past/z':
        FixedLenFeature([128, 10], float32, default_value=None),
}

traffic_light_features = {
    'traffic_light_state/current/id':
        FixedLenFeature([1, 16], int64, default_value=None),
    'traffic_light_state/current/state':
        FixedLenFeature([1, 16], int64, default_value=None),
    'traffic_light_state/current/valid':
        FixedLenFeature([1, 16], int64, default_value=None),
    'traffic_light_state/current/x':
        FixedLenFeature([1, 16], float32, default_value=None),
    'traffic_light_state/current/y':
        FixedLenFeature([1, 16], float32, default_value=None),
    'traffic_light_state/current/z':
        FixedLenFeature([1, 16], float32, default_value=None),
    'traffic_light_state/past/id':
        FixedLenFeature([10, 16], int64, default_value=None),
    'traffic_light_state/past/state':
        FixedLenFeature([10, 16], int64, default_value=None),
    'traffic_light_state/past/valid':
        FixedLenFeature([10, 16], int64, default_value=None),
    'traffic_light_state/past/x':
        FixedLenFeature([10, 16], float32, default_value=None),
    'traffic_light_state/past/y':
        FixedLenFeature([10, 16], float32, default_value=None),
    'traffic_light_state/past/z':
        FixedLenFeature([10, 16], float32, default_value=None),
}

features_description = {}
features_description.update(scenario)
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)

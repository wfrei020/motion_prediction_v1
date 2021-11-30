import motionUtils as Util
import config
from tensorflow.keras.losses import MeanSquaredError
from tensorflow import reshape, newaxis, ones, shape, tile, int64, concat
from tensorflow import range as tfrange
from model import LSTMModel
from tqdm import tqdm
import numpy as np
import motion_metric as mm
from waymo_open_dataset.metrics.python import config_util_py as config_util


class RunModel():
    def __init__(self):
        self.model = LSTMModel(config.num_agents_per_scenario, config.num_states_steps,
                               config.num_future_steps)
        self.model.load_weights(config.MODEL_PATH)
        self.mse = MeanSquaredError()

        self.motion_metric = None
        self.metric_names = None
        metrics_config = mm.default_metrics_config()
        self.motion_metric = mm.MotionMetrics(metrics_config)
        self.metric_names = config_util.get_breakdown_names_from_motion_config(metrics_config)

    def validate(self, type="metric"):
        if type == 'metric':
            self.validate_with_metrics()
        else:
            if config.NUM_OF_VAL_DATASET > 5:
                print(f'ERROR: Dataset to large to save values.')
            self.validate_save_samples()

    def validate_save_samples(self):
        dataset_loss = 0
        tqdm_datasests = tqdm(range(config.NUM_OF_VAL_DATASET))
        for set in tqdm_datasests:
            dataset = Util.get_dataset_segment(config.BASE_VALIDATION_PATH+ str(set).zfill(5) + "-of-00150")
            tqdm_dataset = tqdm(dataset, ncols=80, leave=False)

            for step, batch in enumerate(tqdm_dataset):
                # there should be apre processing step here to deal with data then just train normally
                batch_loss = 0
                lights_coord = batch['lights_input_coord']
                lights_states = batch['lights_input_states']
                lights_valid = batch['lights_is_valid']
                targets = batch['gt_future_states']
                valid_states = batch['gt_future_is_valid']
                gt_targets = targets[..., config.num_states_steps:, :2]
                batch_prediction = None
                batch_target_out = None
                for mini_step, mini_batch in enumerate(batch['input_states']):
                    batch_input, batch_target, batch_origin = Util.get_batch_inputs(mini_batch, gt_targets[mini_step], valid_states[mini_step])
                    input_with_lights = Util.get_batch_inputs_with_lights(batch_input,
                                                                          batch_origin,
                                                                          lights_coord[mini_step],
                                                                          lights_states[mini_step],
                                                                          lights_valid[mini_step])
                    pred_trajectory = self.model(input_with_lights)
                    pred_trajectory = reshape(pred_trajectory, [batch_input.shape[0], config.num_future_steps, 2])
                    pred_trajectory, target = Util.get_actual_predictions(pred_trajectory, batch_target, batch_origin)
                    if batch_prediction == None:
                        batch_prediction = reshape(pred_trajectory,[batch_input.shape[0],80,2])
                        batch_target_out = reshape(target,[target.shape[0],80,2])
                    else:
                        batch_prediction = concat([batch_prediction, reshape(pred_trajectory,[batch_input.shape[0],80,2])],0)
                        batch_target_out = concat([batch_target_out, reshape(target,[target.shape[0],80,2])],0)
                    loss_value = self.mse(target, pred_trajectory)
                batch_loss += float(loss_value)
            dataset_loss += batch_loss / step
            tqdm_datasests.set_postfix({'Dataset_Loss': (batch_loss / (step+1))})
        epoch_loss = dataset_loss / (set+1)
        print(f'Total Loss: {epoch_loss}')
        # do not save alot...
        np.save('results_total.npy',[batch_prediction.numpy(), batch_target_out.numpy()], allow_pickle=True )


    def validate_with_metrics(self):
        tqdm_datasests = tqdm(range(config.NUM_OF_VAL_DATASET))
        for set in tqdm_datasests:
            dataset = Util.get_dataset_segment(config.BASE_VALIDATION_PATH+ str(set).zfill(5) + "-of-00150")
            tqdm_dataset = tqdm(dataset, ncols=80, leave=False)

            for batch in tqdm_dataset:
                batch_prediction = None
                lights_coord = batch['lights_input_coord']
                lights_states = batch['lights_input_states']
                lights_valid = batch['lights_is_valid']
                # TODO NEED TO ADD LIGHTS
                for mini_step, mini_batch in enumerate(batch['input_states']):
                    input, origin = Util.get_all_validate_inputs(mini_batch)
                    input_with_lights = Util.get_batch_inputs_with_lights(input,
                                                                          origin,
                                                                          lights_coord[mini_step],
                                                                          lights_states[mini_step],
                                                                          lights_valid[mini_step])
                    pred_trajectory = self.model(input_with_lights)
                    pred_trajectory = reshape(pred_trajectory, [mini_batch.shape[0], config.num_future_steps, 2])
                    pred_trajectory = Util.get_original_from_shift(origin, pred_trajectory, pred_trajectory.shape)
                    if batch_prediction == None:
                        batch_prediction = reshape(pred_trajectory,[1, mini_batch.shape[0],80,2])
                    else:
                        batch_prediction = concat([batch_prediction, reshape(pred_trajectory,[1, mini_batch.shape[0],80,2])],0)
                self.update_metrics(batch, batch_prediction)
            train_metric_values = self.motion_metric.result()
            for i, m in enumerate(
                    ['min_ade', 'min_fde', 'miss_rate', 'overlap_rate', 'map']):
                for j, n in enumerate(self.metric_names):
                    print('{}/{}: {}'.format(m, n, train_metric_values[i, j]))

    def update_metrics(self, inputs, pred_trajectory):
        pred_trajectory = pred_trajectory[:, :, newaxis, newaxis]
        gt_is_valid = inputs['gt_future_is_valid']
        target = inputs['gt_future_states']
        pred_score = ones(shape=shape(pred_trajectory)[:3])
        # [batch_size, num_agents].
        object_type = inputs['object_type']
        # [batch_size, num_agents].
        batch_size = shape(inputs['tracks_to_predict'])[0]
        num_samples = shape(inputs['tracks_to_predict'])[1]
        pred_gt_indices = tfrange(num_samples, dtype=int64)
        # [batch_size, num_agents, 1].
        pred_gt_indices = tile(pred_gt_indices[newaxis, :, newaxis],
                                (batch_size, 1, 1))
        # [batch_size, num_agents, 1].
        pred_gt_indices_mask = inputs['tracks_to_predict'][..., newaxis]
        self.motion_metric.update_state(pred_trajectory, pred_score, target,
                                    gt_is_valid, pred_gt_indices,
                                    pred_gt_indices_mask, object_type, inputs['object_id'])

if __name__ == "__main__":
    model_instance = RunModel()
    model_instance.validate()

import numpy as np
from tensorflow import (GradientTape, where, not_equal,
                        int32, cast, gather_nd, logical_and, tile, constant, reshape)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model
from tensorflow.data import TFRecordDataset
from model import LSTMModel
import config
import feature_description as fd
from tqdm import tqdm
import motionUtils as Util


class Train():
    def __init__(self, load_saved_model = False):
        if load_saved_model:
            self.model = load_model(config.MODEL_PATH, compile=False)
        else:
            self.model = LSTMModel(config.num_agents_per_scenario, config.num_states_steps,
                               config.num_future_steps)
        self.optimizer = Adam(config.LR)
        self.mse = MeanSquaredError()

    def train_step(self, data):
        ret = []
        with GradientTape() as tape:
            # data should be in the shape of [BATCH_SIZE, num_states_steps, num_of_features]
            pred_trajectory = self.model(data['inputs'], training=True)
            gt_targets = data['targets']
            gt_targets = np.reshape(gt_targets,(gt_targets.shape[0],160))
            loss = self.mse(gt_targets, pred_trajectory)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss


    def train(self):
        # lets train
        tqdm_epoch = tqdm(range(config.EPOCHS))
        for epoch in tqdm_epoch:
            print(f'start of epoch {epoch}')
            dataset_loss = 0
            tqdm_datasests = tqdm(range(config.NUM_OF_DATASET))
            for set in tqdm_datasests:
                batch_loss = 0
                dataset = Util.get_dataset_segment(config.BASE_TRAINING_PATH+ str(set).zfill(5) + "-of-01000")
                tqdm_dataset = tqdm(dataset, ncols=80, leave=False)
                for step, batch in enumerate(tqdm_dataset):
                    # there should be apre processing step here to deal with data then just train normally
                    lights_coord = batch['lights_input_coord']
                    lights_states = batch['lights_input_states']
                    lights_valid = batch['lights_is_valid']
                    object_type = batch['object_type'] # leave this for later
                    targets = batch['gt_future_states']
                    gt_targets = targets[..., config.num_states_steps:, :2]
                    valid_sampled = batch['gt_future_is_valid']
                    for mini_step, mini_batch in enumerate(batch['input_states']):
                        input, target, origin = Util.get_batch_inputs(mini_batch, gt_targets[mini_step], valid_sampled[mini_step])
                        input_with_lights = Util.get_batch_inputs_with_lights(input,
                                                                              origin,
                                                                              lights_coord[mini_step],
                                                                              lights_states[mini_step],
                                                                              lights_valid[mini_step])
                        batch = {'inputs': input_with_lights, 'targets': target, 'origin': origin}
                        loss_value = self.train_step(batch)
                        
                    batch_loss += float(loss_value)
                    tqdm_dataset.set_postfix({'loss': batch_loss / (step+1)})
                tqdm_datasests.set_postfix({'Dataset_Loss': (batch_loss / (step+1))})
                dataset_loss += batch_loss / step

            epoch_loss = dataset_loss / (set+1)
            tqdm_epoch.set_postfix({'Epoch_loss': epoch_loss}) 
            # for now save after every epoch
            self.save_model()

    def save_model(self):
        self.model.save_weights(config.MODEL_PATH)
        self.model.save(config.MODEL_PATH)

if __name__ == "__main__":
    train = Train(True)
    train.train()
    #train.save_model()
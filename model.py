import tensorflow as tf
import numpy as np

class LSTMModel(tf.keras.Model):
  """A simple one-layer regressor."""

  def __init__(self, num_agents_per_scenario, num_states_steps,
               num_future_steps):
    super(LSTMModel, self).__init__()
    self._num_agents_per_scenario = num_agents_per_scenario
    self._num_states_steps = num_states_steps
    self._num_future_steps = num_future_steps
    self.model = tf.keras.Sequential(
    [
        tf.keras.layers.LSTM(
            160,
            activation='tanh',
            return_sequences=False,
            unroll=False
        ),
    ])
    self.model.add(tf.keras.layers.Dense(num_future_steps * 2,activation="linear", name="linear_layer"))


  def call(self, states):
    pred = self.model(states)
    return pred

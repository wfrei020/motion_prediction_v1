import os
START_DOWNLOAD = 30
NUMBER_OF_TRAINING_DOWNLOAD = 201
NUMBER_OF_VALIDATION_DOWNLOAD = 10

BASE_COMMAND = "gsutil -m cp "
BASE_TRAINING_URL = "gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/training/training_tfexample.tfrecord-"
BASE_VALIDATION_URL = "gs://waymo_open_dataset_motion_v_1_0_0/uncompressed/tf_example/validation/validation_tfexample.tfrecord-"
cmd = ""
for i in range(START_DOWNLOAD,NUMBER_OF_TRAINING_DOWNLOAD):
    cmd = BASE_COMMAND + BASE_TRAINING_URL + str(i).zfill(5) +"-of-01000 ~/datasets/motion_prediction/training/"
    os.system(cmd)
quit()
for i in range(NUMBER_OF_VALIDATION_DOWNLOAD):
    cmd = BASE_COMMAND + BASE_VALIDATION_URL + str(i).zfill(5) +"-of-00150 ~/datasets/motion_prediction/validation/"
    os.system(cmd)


import time
import pickle
import json
import os
import math
import unidecode
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime

from collections import Counter
from math import ceil
from sklearn.model_selection import train_test_split

rutaDatos = "../Datos"
iteration_save_path = "./institutional_affiliation_classification"
ruta = "./"
train_data_path = "./training_data"
num_samples_to_get =  50

# HuggingFace library to train a tokenizer
from tokenizers import Tokenizer, normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer

### Combining the training data from 001 notebook and artificial data

# All training samples that have less than 50 different version of the affiliation text
# ---- Created in previous notebook
lower_than = pd.read_parquet(f"{iteration_save_path}/lower_than_{num_samples_to_get}.parquet")

# All training samples that have more than 50 different version of the affiliation text
# ---- Created in previous notebook
more_than = pd.read_parquet(f"{iteration_save_path}/more_than_{num_samples_to_get}.parquet")

print('lower_than.shape: ------------------------------')
print(lower_than.shape)
print('more_than.shape: ----------------------------------')
print(more_than.shape)

full_affs_data = pd.concat([more_than, lower_than],axis=0).reset_index(drop=True)

print('full_affs_data.shape: ----------------------------')
print(full_affs_data.shape)

full_affs_data.to_parquet(f"{ruta}/full_affs_data.parquet")
print('Se crea el archivo full_affs_data.parquet ---------------------------')

full_affs_data['text_len'] = full_affs_data['original_affiliation'].apply(len)
full_affs_data = full_affs_data[full_affs_data['text_len'] < 500][['original_affiliation','affiliation_id']].copy()

full_affs_data['affiliation_id'] = full_affs_data['affiliation_id'].astype('str')

### Processing and splitting the data
full_affs_data['processed_text'] = full_affs_data['original_affiliation'].apply(unidecode.unidecode)

train_data, val_data = train_test_split(full_affs_data, train_size=0.7, random_state=1)  #0.985
train_data = train_data.reset_index(drop=True).copy()
val_data = val_data.reset_index(drop=True).copy()

print('train_data.shape: ----------------------------')
print(train_data.shape)

print('val_data.shape: ------------------------------------')
print(val_data.shape)

affs_list_train = train_data['processed_text'].tolist()
affs_list_val = val_data['processed_text'].tolist()

try:
    os.system(f"rm {ruta}/aff_text.txt")
    print("Done")
except:
    pass

print('affs_list_train[:5]:     ---------------------------------------')
print(affs_list_train[:5])

# save the affiliation text that will be used to train a tokenizer
with open(f"{ruta}/aff_text.txt", "w") as f:
    for aff in affs_list_train:
        f.write(f"{aff}\n")


try:
    os.system(f"rm {ruta}/basic_model_tokenizer")
    print("Done")
except:
    pass

full_affs_data[['processed_text','affiliation_id']].to_parquet(f"{ruta}/full_affs_data_processed.parquet")
print('Se crea el archivo full_affs_data_processed.parquet ---------------------------')

### Creating the tokenizer for the basic model

wordpiece_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

# NFD Unicode, lowercase, and getting rid of accents (to make sure text is as readable as possible)
wordpiece_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

# Splitting on whitespace
wordpiece_tokenizer.pre_tokenizer = Whitespace()

# Training a tokenizer on the training dataset
trainer = WordPieceTrainer(vocab_size=3816, special_tokens=["[UNK]"])
files = [f"{ruta}/aff_text.txt"]
wordpiece_tokenizer.train(files, trainer)

wordpiece_tokenizer.save(f"{ruta}/basic_model_tokenizer")

### Further processing of data with tokenizer

def max_len_and_pad(tok_sent):
    """
    Truncates sequences with length higher than max_len and also pads the sequence
    with zeroes up to the max_len.
    """
    max_len = 128
    tok_sent = tok_sent[:max_len]
    tok_sent = tok_sent + [0]*(max_len - len(tok_sent))
    return tok_sent

def create_affiliation_vocab(x):
    """
    Checks if affiliation is in vocab and if not, adds to the vocab.
    """
    if x not in affiliation_vocab.keys():
        affiliation_vocab[x]=len(affiliation_vocab)
    return [affiliation_vocab[x]]

# initializing an empty affiliation vocab
affiliation_vocab = {}

# tokenizing the training dataset
tokenized_output = []
for i in affs_list_train:
    tokenized_output.append(wordpiece_tokenizer.encode(i).ids)

train_data['original_affiliation_tok'] = tokenized_output

# tokenizing the validation dataset
tokenized_output = []
for i in affs_list_val:
    tokenized_output.append(wordpiece_tokenizer.encode(i).ids)

val_data['original_affiliation_tok'] = tokenized_output

# applying max length cutoff and padding
train_data['original_affiliation_model_input'] = train_data['original_affiliation_tok'].apply(max_len_and_pad)
val_data['original_affiliation_model_input'] = val_data['original_affiliation_tok'].apply(max_len_and_pad)

# creating the label affiliation vocab
train_data['label'] = train_data['affiliation_id'].apply(lambda x: create_affiliation_vocab(x))

print('len(affiliation_vocab): --------------------------------')
print(len(affiliation_vocab))

val_data['label'] = val_data['affiliation_id'].apply(lambda x: [affiliation_vocab.get(x)])

print('train_data: -------------------------------')
print(train_data.shape)
print('val_data: ------------------------------------')
print(val_data.shape)

train_data.to_parquet(f"{ruta}/train_data.parquet")
print('Se crea el archivo train_data: --------------------------')
val_data.to_parquet(f"{ruta}/val_data.parquet")
print('Se crea el archivo val_data: ---------------------------------')

# saving the affiliation vocab
with open(f"{ruta}affiliation_vocab.pkl","wb") as f:
    pickle.dump(affiliation_vocab, f)

### Creating TFRecords from the training and validation datasets   #####################
train_data = pd.read_parquet(f"{ruta}/train_data.parquet")
val_data = pd.read_parquet(f"{ruta}/val_data.parquet")

print('train_data.shape: ---------------------------------')
print(train_data.shape)
print('val_data.shape: ----------------------------------')
print(val_data.shape)

# saving the affiliation vocab
with open(f"{ruta}affiliation_vocab.pkl","rb") as f:
    affiliation_vocab = pickle.load(f)

def create_tfrecords_dataset(data, iter_num, dataset_type='train'):
    """
    Creates a TF Dataset that can then be saved to a file to make it faster to read
    data during training and allow for transferring of data between compute instances.
    """
    ds = tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(data['original_affiliation_model_input'].to_list()),
                              tf.data.Dataset.from_tensor_slices(data['label'].to_list())))
    
    serialized_features_dataset = ds.map(tf_serialize_example)
    
    filename = f"{train_data_path}/{dataset_type}/{str(iter_num).zfill(4)}.tfrecord"
    writer = tf.data.experimental.TFRecordWriter(filename)
    writer.write(serialized_features_dataset)

def tf_serialize_example(f0, f1):
    """
    Serialization function.
    """
    tf_string = tf.py_function(serialize_example, (f0, f1), tf.string)
    return tf.reshape(tf_string, ())

def serialize_example(features, label):
    """
    Takes in features and outputs them to a serialized string that can be written to
    a file using the TFRecord Writer.
    """
    features_list = tf.train.Int64List(value=features.numpy().tolist())
    label_list = tf.train.Int64List(value=label.numpy().tolist())
    
    features_feature = tf.train.Feature(int64_list = features_list)
    label_feature = tf.train.Feature(int64_list = label_list)
    
    features_for_example = {
        'features': features_feature,
        'label': label_feature
    }
    
    example_proto = tf.train.Example(features=tf.train.Features(feature=features_for_example))
    
    return example_proto.SerializeToString()

# Making sure data is in the correct format before going into TFRecord
train_data['original_affiliation_model_input'] = train_data['original_affiliation_model_input'].apply(lambda x: np.asarray(x, dtype=np.int64))

val_data['original_affiliation_model_input'] = val_data['original_affiliation_model_input'].apply(lambda x: np.asarray(x, dtype=np.int64))

os.system(f"mkdir -p {train_data_path}/train/")
os.system(f"mkdir -p {train_data_path}/val/")
print("Done")

##Creating the Train Dataset   ####################

##%%time --- es un comando de jupyter
start_cpu_time = time.process_time()
start_wall_time = time.time()
for i in range(ceil(train_data.shape[0]/1000)): #500.000
    print(i)
    low = i*1000
    high = (i+1)*1000
    create_tfrecords_dataset(train_data.iloc[low:high,:], i, 'train')
end_cpu_time = time.process_time()
end_wall_time = time.time()
print(f"CPU time used: {end_cpu_time - start_cpu_time} seconds")
print(f"Elapsed time: {(end_wall_time - start_wall_time)*1000}  μs")

#### Creating the Validation Dataset ######################33

##%%time
start_cpu_time = time.process_time()
start_wall_time = time.time()
for i in range(ceil(val_data.shape[0]/3000)):
    print(i)
    low = i*3000
    high = (i+1)*3000
    create_tfrecords_dataset(val_data.iloc[low:high,:], i, 'val')
end_cpu_time = time.process_time()
end_wall_time = time.time()
print(f"CPU time used: {end_cpu_time - start_cpu_time} seconds")
print(f"Elapsed time: {(end_wall_time - start_wall_time)*1000}  μs")

### Loading the Data

def _parse_function(example_proto):
    """
    Parses the TFRecord file.
    """
    feature_description = {
        'features': tf.io.FixedLenFeature((128,), tf.int64),
        'label': tf.io.FixedLenFeature((1,), tf.int64)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    features = example['features']
    label = example['label'][0]

    return features, label

AUTO = tf.data.experimental.AUTOTUNE

def get_dataset(path, data_type='train'):
    """
    Takes in a path to the TFRecords and returns a TF Dataset to be used for training.
    """
    tfrecords = [f"{path}/{data_type}/{x}" for x in os.listdir(f"{path}/{data_type}/") if x.endswith('tfrecord')]
    tfrecords.sort()


    raw_dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO)
    parsed_dataset = raw_dataset.map(_parse_function, num_parallel_calls=AUTO)

    parsed_dataset = parsed_dataset.apply(tf.data.experimental.dense_to_ragged_batch(512,drop_remainder=True))
    return parsed_dataset


training_data = get_dataset(train_data_path, data_type='train')
validation_data = get_dataset(train_data_path, data_type='val')

### Load Vocab ####

# Loading the affiliation (target) vocab
with open(f"{ruta}/affiliation_vocab.pkl","rb") as f:
    affiliation_vocab = pickle.load(f)

inverse_affiliation_vocab = {i:j for j,i in affiliation_vocab.items()}

print('len(affiliation_vocab ------------------------------')
print(len(affiliation_vocab))
print('inverse_affiliation_vocab _--------------------------------')
print(inverse_affiliation_vocab)

### Creating Model #####################33

# Hyperparameters to tune
emb_size = 256
max_len = 128
num_layers = 6
num_heads = 8
dense_1 = 2048
dense_2 = 1024
learn_rate = 0.00004

def scheduler(epoch, curr_lr):
    """
    Setting up a exponentially decaying learning rate.
    """
    rampup_epochs = 2
    exp_decay = 0.17
    def lr(epoch, beg_lr, rampup_epochs, exp_decay):
        if epoch < rampup_epochs:
            return beg_lr
        else:
            return beg_lr * math.exp(-exp_decay * epoch)
    return lr(epoch, start_lr, rampup_epochs, exp_decay)

# Allow for use of multiple GPUs
# mirrored_strategy = tf.distribute.MirroredStrategy()
mirrored_strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1", "CPU:2", "CPU:3", "CPU:4", "CPU:5"])
#mirrored_strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])

with mirrored_strategy.scope():
    # Model Inputs
    tokenized_aff_string_ids = tf.keras.layers.Input((128,), dtype=tf.int64, name='tokenized_aff_string_input')

    # Embedding Layers  #3816
    tokenized_aff_string_emb_layer = tf.keras.layers.Embedding(input_dim=330,
                                                               output_dim=int(emb_size),
                                                               mask_zero=True,
                                                               trainable=True,
                                                               name="tokenized_aff_string_embedding")

    tokenized_aff_string_embs = tokenized_aff_string_emb_layer(tokenized_aff_string_ids)

    # First dense layer
    dense_output = tf.keras.layers.Dense(int(dense_1), activation='relu',
                                             kernel_regularizer='L2', name="dense_1")(tokenized_aff_string_embs)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_1")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_1")(dense_output)
    pooled_output = tf.keras.layers.GlobalAveragePooling1D()(dense_output)

    # Second dense layer
    dense_output = tf.keras.layers.Dense(int(dense_2), activation='relu',
                                             kernel_regularizer='L2', name="dense_2")(pooled_output)
    dense_output = tf.keras.layers.Dropout(0.20, name="dropout_2")(dense_output)
    dense_output = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="layer_norm_2")(dense_output)

    # Last dense layer
    final_output = tf.keras.layers.Dense(len(affiliation_vocab), activation='softmax', name='cls')(dense_output)

    model = tf.keras.Model(inputs=tokenized_aff_string_ids, outputs=final_output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate, beta_1=0.9,
                                                     beta_2=0.99),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    curr_date = datetime.now().strftime("%Y%m%d")

    filepath_1 = f"{ruta}/models/{curr_date}_{dense_1}d1_{dense_2}d2/" \


    filepath = filepath_1 + "model_epoch{epoch:02d}ckpt.keras"

    # Adding in checkpointing
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss',
                                                          verbose=0, save_best_only=False,
                                                          save_weights_only=False, mode='auto',
                                                          save_freq='epoch')

    # Adding in early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=4)

    start_lr = float(learn_rate)

    # Adding in a learning rate schedule to decrease learning rate in later epochs
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)

    callbacks = [model_checkpoint, early_stopping, lr_schedule]

print('model.summary() --------------------------------------')
print(model.summary())

### Training the Model
print('Training the Model ------------------------------------')

history = model.fit(training_data, epochs=20, validation_data=validation_data, verbose=1, callbacks=callbacks)

json.dump(str(history.history), open(f"{filepath_1}_25EPOCHS_HISTORY.json", 'w+'))

model.save(f"{filepath_1}basic_model.keras")
print('Archivo del modelo guardado ---------')

print('FINALIZADO')
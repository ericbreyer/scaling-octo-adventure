#%% Defines
import glob
import sys
import time
import joblib
import mne
import matplotlib.pyplot as plt
import numpy as np

from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)

    for i, item in enumerate(it):
        yield item
        show(i+1)
    
    print("", flush=True, file=out)

def wrangle_data(person, SUB_EPOCHS=1):
    channels = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'Resp oro-nasal', 'EMG submental', 'Temp rectal']

    # Load the data
    # load the data
    X_raw= np.load(f"{person}_X.npy")
    y_raw = np.array([0,0])
    try:
        y_raw = np.load(f"{person}_y.npy")
    except:
        print(f"y file for {person} not found")
    # load the data into an mne object
    ch_types = ['eeg', 'eeg', 'eog', 'ecg', 'misc', 'misc']
    ch_names = channels
    sfreq = 100
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types, verbose=False)
    raw = mne.io.RawArray(np.hstack(X_raw), info, verbose=False)

    # rename the channels
    rename_dict = {
        'EEG Fpz-Cz': 'Fz',
        'EEG Pz-Oz': 'Pz',
        'EOG horizontal': 'eog',
        'Resp oro-nasal': 'ecg',
        'EMG submental': 'misc2',
        'Temp rectal': 'misc3'
    }
    raw.rename_channels(rename_dict, verbose=False)

    # set the montage
    montage = mne.channels.make_standard_montage('standard_1020')

    raw.set_montage(montage, verbose=False)

    num_eeg_channels = len(mne.pick_types(raw.info, meg=False, eeg=True))

    # DEBUG
    # print(raw.info)
    # raw.plot()

    # filter the data
    raw.filter(1., 40., verbose=False)

    # artifact removal
    ica = mne.preprocessing.ICA(n_components=num_eeg_channels, random_state=97, max_iter=800, verbose=False)
    ica.fit(raw, verbose=False)

    # epoch the data
    # epoch into 30 second epochs
    events = mne.make_fixed_length_events(raw, duration= (30 / SUB_EPOCHS))
    epochs = mne.Epochs(raw, events=events, tmin=0., tmax=30., verbose=False, baseline=(0, 0))
    epochs.apply_baseline((0, 0), verbose=False)

    # comiple the features into a numpy array of (epochs, features)

    # features per epoch:
    # 1. mean of both channels
    # 2. std of both channels
    # 3. power of delta band of both channels (0.5-4 Hz)
    # 4. power of theta band of both channels (4-8 Hz)
    # 5. power of alpha band of both channels (8-12 Hz)
    # 6. power of beta band of both channels (12-30 Hz)
    # 7. power of gamma band of both channels (30-40 Hz)
    # 8. average coherence between Fz and Pz
    
    X = np.zeros((epochs.get_data(verbose = False).shape[0], 14))

    # compute the features for each epoch 
    # epochs.get_data() has shape (epochs, channels, samples)
    for i, epoch in enumerate(epochs.get_data(verbose = False)):
        # mean of both channels
        X[i][0] = np.mean(epoch[0, :])
        X[i][1] = np.mean(epoch[1, :])

        # std of both channels
        X[i][2] = np.std(epoch[0, :])
        X[i][3] = np.std(epoch[1, :])

        # sfft of both channels
        f, t, Zxx = signal.stft(epoch, fs=raw.info['sfreq'])


        # power of delta band of both channels (0.5-4 Hz)
        delta_indices = np.where(np.logical_and(f >= 0.5, f <= 4))
        X[i][4] = np.sum(np.abs(Zxx[0, delta_indices, :]))
        X[i][5] = np.sum(np.abs(Zxx[1, delta_indices, :]))

        # power of theta band of both channels (4-8 Hz)
        theta_indices = np.where(np.logical_and(f >= 4, f <= 8))
        X[i][6] = np.sum(np.abs(Zxx[0, theta_indices, :]))
        X[i][7] = np.sum(np.abs(Zxx[1, theta_indices, :]))

        # power of alpha band of both channels (8-12 Hz)
        alpha_indices = np.where(np.logical_and(f >= 8, f <= 12))
        X[i][8] = np.sum(np.abs(Zxx[0, alpha_indices, :]))
        X[i][9] = np.sum(np.abs(Zxx[1, alpha_indices, :]))

        # power of beta band of both channels (12-30 Hz)
        beta_indices = np.where(np.logical_and(f >= 12, f <= 30))
        X[i][10] = np.sum(np.abs(Zxx[0, beta_indices, :]))
        X[i][11] = np.sum(np.abs(Zxx[1, beta_indices, :]))

        # power of gamma band of both channels (30-40 Hz)
        gamma_indices = np.where(np.logical_and(f >= 30, f <= 40))
        X[i][12] = np.sum(np.abs(Zxx[0, gamma_indices, :]))
        X[i][13] = np.sum(np.abs(Zxx[1, gamma_indices, :]))
        

    labels = ["mean Fz", "mean Pz", "std Fz", "std Pz", "delta Fz", "delta Pz", "theta Fz", "theta Pz", "alpha Fz", "alpha Pz", "beta Fz", "beta Pz", "gamma Fz", "gamma Pz"]

    # standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if SUB_EPOCHS > 1:
        #regroup sub epochs into epochs
        X = np.reshape(X, (X.shape[0] // SUB_EPOCHS, SUB_EPOCHS, X.shape[1]))

    return X, y_raw[:-1], labels

# %% Process the data as epoch blocks

# get all files in the Training directory
people = glob.glob("Training/*_NEW_X.npy")
people = [p[:-6] for p in people]

X, y, labels = None, None, None
for person in progressbar(people, prefix="Processing people: "):
    X_temp, y_temp, labels_temp = wrangle_data(person)
    if X is None:
        X = X_temp
        y = y_temp
        labels = labels_temp
    else:
        X = np.concatenate((X, X_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)


# fig, axs = plt.subplots(2, 1)
# for ch in range(2):
#     for f in range(7):
#         axs[ch].plot(np.arange(0, X.shape[0]), X[:, ch*7 + f][:100], label=labels[ch*7 + f])
    
#     yxs = axs[ch].twinx()
#     yxs.plot(y[:100], color='tab:green')

#     axs[ch].legend()

# plt.show()

# split the data into training and testing sets
num_categories = 7
X_prime = X
y_prime = to_categorical(y, num_categories)

X_train, X_test, y_train, y_test = train_test_split(X_prime, y_prime, test_size=0.2)

# make a keras model

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_categories, activation='softmax')
])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stop])

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# make predictions and plot them against the actual data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

y_test = np.argmax(y_test, axis=1)

fig, axs = plt.subplots(2, 2)
for ax in axs.flatten():
    rand = np.random.randint(0, len(y_test) - 100)
    ax.plot(y_pred[rand:rand+100], color='tab:red', label='Predicted')
    ax.plot(y_test[rand:rand+100], color='tab:blue', label='Actual')
    ax.legend()
plt.show()

# %% Process the data with sub epochs

# get all files in the Training directory
people = glob.glob("Training/*_NEW_X.npy")
people = [p[:-6] for p in people]
SUB_EPOCHS = 6
X, y, labels = None, None, None
for person in progressbar(people, prefix="Processing people: "):
    X_temp, y_temp, labels_temp = wrangle_data(person, SUB_EPOCHS=SUB_EPOCHS)
    if X is None:
        X = X_temp
        y = y_temp
        labels = labels_temp
    else:
        X = np.concatenate((X, X_temp), axis=0)
        y = np.concatenate((y, y_temp), axis=0)

# split the data into training and testing sets
num_categories = 7
X_prime = X
y_prime = to_categorical(y, num_categories)

X_train, X_test, y_train, y_test = train_test_split(X_prime, y_prime, test_size=0.2)

# make a keras model

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.LSTM(32, return_sequences=True),
    layers.LSTM(32),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_categories, activation='softmax')
])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stop])

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# make predictions and plot them against the actual data
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

y_test = np.argmax(y_test, axis=1)

fig, axs = plt.subplots(2, 2)
for ax in axs.flatten():
    rand = np.random.randint(0, len(y_test) - 100)
    ax.plot(y_pred[rand:rand+100], color='tab:red', label='Predicted')
    ax.plot(y_test[rand:rand+100], color='tab:blue', label='Actual')
    ax.legend()

plt.show()
# %% Predict on the evaluation data

# get all files in the Eval directory
people = glob.glob("Eval/*_NEW_X.npy")
people = [p[:-6] for p in people]
SUB_EPOCHS = 1
for person in progressbar(people, prefix="Processing people: "):
    X, _, _ = wrangle_data(person, SUB_EPOCHS=SUB_EPOCHS)

    # make predictions and plot them
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(y_pred, color='tab:red', label='Predicted')
    ax.legend()
    plt.show()

    # export the predictions to a npy file
    np.save(f"{person}_block.npy", y_pred)
# %%

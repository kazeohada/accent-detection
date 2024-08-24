## Prerequisistes

Python version 3.8.x

The follwoing Python packages are required:
music21
pandas
matplotlib
Keras 2
tensorflow 2.13
[madmom](https://github.com/CPJKU/madmom)

Other requirements
[MuseScore4](https://musescore.org/en/handbook/4/download-and-installation), make sure it is in the PATH
ffmpeg, can be installed by running:
```
$ cd onset-model/tmp && wget https://ffmpeg.org/releases/ffmpeg-4.1.tar.bz2 \
    && tar xvjf ffmpeg-4.1.tar.bz2 && cd ffmpeg-4.1 \
    && ./configure && make
$ export PATH=/tmp/ffmpeg-4.1:$PATH
```

# The Data Pipeline

Make sure you run every python script in the ./data-pipeline directory
Before running, set up Music21's MusicXML path as MuseScore4 by running config.py.
Run the data pipeline with main.py.
The data pipeline will take MusicXML files located in corpus/mxl and output a dataset in the dataset directory.
Once the transformation is complete, the script will run MuseScore4 processes for converting to audio. You can check if the processes are complete on the Task Manager (Windows) or Activity Monitor (Mac).

# The Onset Model

Original model by Bjöorn Lindqvist
(https://github.com/bjourne/onset-replication)

## Configuration

Paths to input, output and cache data has to be configured by
modifying the `CONFIGS` constant in the `config.py` file. The right
config is selected during runtime by matching on the system and
hostname. This way the same `config.py` can be used on multiple
systems without requiring any changes.

The `data-dir` field should be set to the directory containing the
dataset, `cache-dir` to a directory storing cache files in pickle
format and `model-dir` to the directory in which built models should
be stored.

The `seed` field contains the seed to the random number generators
ensuring that *exactly* the same results a produced every
time. `digest` contains the checksum of the cache file. It is
important that the cache file does not change during training or
evaluation.

## Training

Training is done using the `main.py` script:
```
$ python main.py -t 0:8 -n cnn --epochs 20
```
There are 3 options for the -n flag: multiclass, multilabel1, multilabel2.
Training can be stopped and resumed at any time.

## Evaluation

Evaluation is done using the `main.py` script:
```
$ python main.py -e 0:8 -n cnn
```

There are 3 options for the -n flag: multilabel1, multilabel2.
The evalutation function will record results of every function trained and output the records in a csv file inside ./results.
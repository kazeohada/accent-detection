import music21 as m21
import os, threading
from transform import main as transform
from musescore_conversion import main as convert_to_audio
import pandas as pd
from preprocess import instruments_df
from instruments_data import main as make_instrument_dfs
from config import set_musicxml_path

def remove_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            remove_directory(item_path)
        else:
            os.remove(item_path)
    os.rmdir(directory)

def main():

    # Create directories for data
    remove_directory('dataset')
    os.makedirs('dataset')
    for directory in ['mxl', 'bat', 'json', 'annotations', 'audio', 'labels']:
        os.makedirs(f'dataset/{directory}')

    threads = []
    mxl_files = os.listdir('corpus/mxl/')

        
    for file in mxl_files:
        name = file[:-4]
        t = threading.Thread(name=f"{name.replace('/', '_')}", target=transform, args=(f"corpus/mxl/{file}", f"{name.replace('/', '_')}"))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print('MXL Files Generated')
    make_instrument_dfs(instruments_df)

    print('Converting to Audio...')
    convert_to_audio()

if __name__ == "__main__":
    set_musicxml_path()
    main()
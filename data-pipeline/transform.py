import music21 as m21
import pandas as pd
import random, os
from preprocess import clean, make_snippets, set_tempo, count_instruments

class Pointer:
    def __init__(self, notes, offset):
        self.notes = notes
        self.index = 0
        self.offset = offset
    
    def update_index(self, index):
        self.index = index
        if len(self.notes) > index:
            self.offset = self.notes[index].offset
            return True
        else:
            return False

    def get_note(self): return self.notes[self.index]
    
    def get_index(self): return self.index
    
    def get_offset(self): return self.offset


def make_pointer_dict(piece):
    pointer_dict = {}
    for i in range(len(piece.parts)):
        if len(piece.parts[i].flatten().notes) > 0:
            pointer_dict[i] = Pointer(piece.parts[i].flatten().notes, piece.parts[i].flatten().notes[0].offset)
    return pointer_dict


def get_lowest_pointers(pointer_dict):
    # returns indexes of pointers with the lowest offset
    low = min([pointer_dict[i].get_offset() for i in pointer_dict])
    return [i for i in pointer_dict if pointer_dict[i].get_offset() == low]


def move_pointers(pointer_dict):
    to_move = get_lowest_pointers(pointer_dict)
    for i in to_move:
        if pointer_dict[i].update_index(pointer_dict[i].get_index() + 1) == False:
            del pointer_dict[i]
    return pointer_dict


def transform(snippet, name, tempo):

    df = pd.DataFrame(columns=['onset', 'accent'])

    for part in snippet.getElementsByClass(m21.stream.Part):
        part = set_tempo(part, tempo)

    snippet = snippet.makeMeasures()

    pointer_dict = make_pointer_dict(snippet)

    # inner function for accenting notes
    def accent(ptrs):
        for i in ptrs:
            note = pointer_dict[i].get_note()
            if note.tie == None or note.tie.type == 'start': # do not accent if note is tied
                note.articulations = [m21.articulations.Accent()]

    # iterating through every note in snippet
    while len(pointer_dict) > 0:
        lowest_ptrs = get_lowest_pointers(pointer_dict)

        # if any note in lowest_ptrs is already accented, accent the rest of the notes in lowest_ptrs
        if any(m21.articulations.Accent in note.articulations for note in (pointer_dict[i].get_note() for i in lowest_ptrs)):
            accent(lowest_ptrs)
        # accent notes
        elif random.random() < (0.2):
            accent(lowest_ptrs)
        
        # if all notes of lowest_ptrs are tied, do not register as onset, unless it is the very first note of the snippet
        note = next((note for note in (pointer_dict[i].get_note() for i in lowest_ptrs) if (note.tie == None or note.tie.type == 'start')), None)
        if note != None:
            onset = "{:.9f}".format(float(note.offset * 60 / tempo)) #convert to seconds
            isAccent = any(isinstance(articulation, m21.articulations.Accent) for articulation in note.articulations)
            df.loc[len(df)] = {
                'onset': onset,
                'accent': isAccent
            }
        else:
            note = pointer_dict[lowest_ptrs[0]].get_note()
            if note != None and note.offset == 0:
                onset = "{:.9f}".format(float(note.offset * 60 / tempo)) #convert to seconds
                isAccent = any(isinstance(articulation, m21.articulations.Accent) for articulation in note.articulations)
                df.loc[len(df)] = {
                    'onset': onset,
                    'accent': isAccent
                }

        pointer_dict = move_pointers(pointer_dict)

    
    count_instruments(snippet, name)
    write(snippet, df, name)


def write(snippet, df, name):
    os.makedirs(f"dataset/annotations/{name}")

    snippet.write('mxl', f'dataset/mxl/{name}.mxl')
    
    # onsets_df = df[df['accent'] == False] # for one-hot labels
    onsets_df = df
    onsets_df["onset"].to_csv(f'dataset/annotations/{name}/{name}.ONSETS', index=False, header=False, float_format='%.9f')

    accents_df = df[df['accent'] == True]
    accents_df["onset"].to_csv(f'dataset/annotations/{name}/{name}.ACCENTS', index=False, header=False, float_format='%.9f')
    
    # labels for audacity
    os.makedirs(f"dataset/labels/{name}")
    with open(f"dataset/labels/{name}/true-onsets.txt", "a") as onsets_file:
        for i, row in onsets_df.iterrows():
            onsets_file.write(f"{row['onset']}\t{row['onset']}\t\n")
    
    with open(f"dataset/labels/{name}/true-accents.txt", "a") as accents_file:
        for i, row in accents_df.iterrows():
            accents_file.write(f"{row['onset']}\t{row['onset']}\t\n")  


def main(file, name):
    piece = m21.converter.parse(file)
    print(f"Transforming {name}...")
    
    piece = clean(piece)

    # create a list of snippets of the piece
    snippets, tempos = make_snippets(piece)

    # choose up to 5 snippets for the dataset
    num_chosen = random.randint(min(3, len(snippets)), min(4, len(snippets)))
    chosen_snippets = random.sample(list(range(len(snippets))), num_chosen)

    n = 1
    # transform each chosen snippet
    for i in chosen_snippets:
        transform(snippets[i], f'{name}_{n}', tempos[i])
        n += 1

    print(f"DONE {name}")
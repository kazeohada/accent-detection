import pandas as pd
from matplotlib import pyplot as plt
import os
import music21 as m21

def count_unique(df : pd.DataFrame):
    unique_ins_df = pd.DataFrame(columns=['instrument', 'count'])
    instruments = df.columns[2:]
    for instrument in instruments:
        count = 0 
        for i, row in df.iterrows():
            if row[instrument] > 0:
                count += 1
        unique_ins_df.loc[len(unique_ins_df.index)] = [instrument, count]

    unique_ins_df = unique_ins_df.sort_values(by=['count']).reset_index(drop=True)

    plt.figure(figsize=(13, 10)) 
    plt.barh(width=unique_ins_df['count'], y=unique_ins_df['instrument'], height=0.7)
    plt.savefig('dataset/instruments_unique_count_per_snippet.png')

    return unique_ins_df

def count_mono_poly(df : pd.DataFrame):
    mono_poly_df = pd.DataFrame({'type' : ['monophonic', 'polyphonic'], 'count' : [0, 0]})
    for i, row in df.iterrows():
        a = mono_poly_df[mono_poly_df['type'] == row['type']]
        mono_poly_df.loc[a.index, 'count'] += 1
    return mono_poly_df


instruments_df = pd.DataFrame(columns=['snippet', 'type'])
def count_instruments(piece, name):
    new_row = [name, None]
    for i in range(2, len(instruments_df.columns)):
        new_row.append(0)

    n = 0

    for part in piece.parts:
        thereIsANote = False
        for measure in part.getElementsByClass(m21.stream.Measure):
            for note in measure.getElementsByClass(m21.note.Note):
                thereIsANote = True
                break
            for chord in measure.getElementsByClass(m21.chord.Chord): # piece is polyphonic
                n += 1
                thereIsANote = True
                break

        if (thereIsANote): 
            n += 1
            for instrument in part.getInstruments():
                if instrument.classes[0] not in instruments_df.columns:
                    instruments_df.insert(len(instruments_df.columns), instrument.classes[0], 0)
                    new_row.append(1)
                else:
                    new_row[instruments_df.columns.get_loc(instrument.classes[0])] += 1

    if n == 1:
        new_row[1] = "monophonic"
    else:
        new_row[1] = "polyphonic"
    instruments_df.loc[len(instruments_df.index)] = new_row

def count_parts(df : pd.DataFrame):
    df = df.set_index('snippet')
    parts_df = pd.DataFrame({'parts' : [], 'count' : []})
    parts_df.set_index('parts', inplace=True)
    for i, row in df.iterrows():
        instruments = row.values.flatten().tolist()[1:]
        n_parts = sum(instruments)
        if n_parts not in parts_df.index:
            parts_df.loc[n_parts] = [1]
        else:
            parts_df.loc[n_parts]['count'] += 1

    parts_df.sort_index(ascending=True, inplace=True)

    plt.figure(figsize=(8, 6))
    plt.bar(parts_df.index, parts_df['count'])
    plt.xticks(parts_df.index)
    plt.xlabel('Number of Parts')
    plt.ylabel('Count')
    plt.savefig('dataset/number_parts.png')

    return parts_df


def count_dynamics(df : pd.DataFrame):
    df = df.set_index('snippet')
    snippet_names = list(df.index.values)
    dynamics_df = pd.DataFrame(columns=['dynamic', 'count'])
    dynamics_df.set_index('dynamic', inplace=True)

    for name in snippet_names:
        dynamics = []
        snippet = m21.converter.parse(f"dataset/mxl/{name}.mxl")
        for part in snippet.getElementsByClass(m21.stream.Part):
            no_dynamic = True
            for measure in part.getElementsByClass(m21.stream.Measure):
                for dynamic in measure.getElementsByClass(m21.dynamics.Dynamic):
                    if dynamic.value not in dynamics:
                        dynamics.append(dynamic.value)
                    no_dynamic = False
            if no_dynamic:
                dynamics.append('normal')
        
        for dynamic in dynamics:
            if dynamic in dynamics_df.index:
                dynamics_df.loc[dynamic]['count'] += 1
            else:
                dynamics_df.loc[dynamic] = [1]
    
    dynamics_df = dynamics_df.reindex(["ppp", "pp", "p", "mp", "normal", "mf", "f", "ff", "fff", "other-dynamics"])
    dynamics_df.loc["other"] = dynamics_df.loc["other-dynamics"]
    dynamics_df = dynamics_df.drop("other-dynamics")

    plt.figure(figsize=(8, 8))
    plt.bar(dynamics_df.index, dynamics_df['count'])
    plt.xticks(dynamics_df.index)
    plt.xlabel('Dynamic')
    plt.ylabel('Count')
    plt.savefig('dataset/dynamics.png')
    return dynamics_df




def main(df : pd.DataFrame):
    df.sort_values(by=['snippet']).reset_index(drop=True)
    df.to_csv('dataset/instruments.csv', sep=",", index=False)

    count_mono_poly(df).to_csv('dataset/snippet_type_count.csv', sep=",", index=False)
    count_unique(df).to_csv('dataset/instruments_unique_count_per_snippet.csv', sep=",", index=False)
    count_parts(df).to_csv('dataset/number_parts.csv', sep=",")
    count_dynamics(df).to_csv('dataset/dyamics_in_snippets.csv', sep=",")

    return

if __name__ == "__main__":
    if os.path.exists("dataset/instruments.csv"):
        main(pd.read_csv("dataset/instruments.csv"))
    else:
        for snippet in os.listdir("dataset/mxl"):
            count_instruments(m21.converter.parse(f"dataset/mxl/{snippet}"), snippet[:-4])
        main(instruments_df)




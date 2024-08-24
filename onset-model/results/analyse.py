import music21 as m21
import os
import pandas as pd
from matplotlib import pyplot as plt
from math import floor
from madmom.evaluation.onsets import OnsetEvaluation
ONSET_EVAL_WINDOW = 0.025
# Combine onsets detected less than 30 ms apart.
ONSET_EVAL_COMBINE = 0.03


def clean(l):
    for i in range(len(l)):
        l[i] = l[i].split("\t")[0]
        if l[i] == "": 
            l.remove(l[i])
        else: 
            l[i] = float(l[i])
            l[i] = floor(l[i]  * 100) / 100.0

    l.sort()
    return l

def num_parts(files):
    global instruments_df
    per_nparts_df = pd.DataFrame(columns=['parts', 'tp_onsets', 'fp_onsets', 'fn_onsets', 'tp_accents', 'fp_accents', 'fn_accents'])
    per_nparts_df.set_index('parts', inplace=True)


    for file in files:
        row = instruments_df.loc[file[0]]
        instruments = row.values.flatten().tolist()[1:]
        n_parts = sum(instruments)
        guess_onsets = file[1]
        true_onsets = file[2]
        guess_accents = file[3]
        true_accents = file[4]

        ons_eval = OnsetEvaluation( # onset evaluation
            guess_onsets, true_onsets,
            combine = ONSET_EVAL_COMBINE,
            window = ONSET_EVAL_WINDOW)

        acc_eval = OnsetEvaluation( # accent evaluation
            guess_accents, true_accents,
            combine = ONSET_EVAL_COMBINE,
            window = ONSET_EVAL_WINDOW)
        

        if n_parts not in per_nparts_df.index:
            per_nparts_df.loc[n_parts] = [ons_eval.num_tp, ons_eval.num_fp, ons_eval.num_fn, acc_eval.num_tp, acc_eval.num_fp, acc_eval.num_fn]
        else:
            row = per_nparts_df.loc[n_parts]
            row['tp_onsets'] += ons_eval.num_tp
            row['fp_onsets'] += ons_eval.num_fp
            row['fn_onsets'] += ons_eval.num_fn
            row['tp_accents'] += acc_eval.num_tp
            row['fp_accents'] += acc_eval.num_fp
            row['fn_accents'] += acc_eval.num_fn


    per_nparts_df.sort_index(inplace=True)
    ons_metrics = {

    }
    acc_metrics = {
        
    }

    for nparts in per_nparts_df.index.values.tolist():
        row = per_nparts_df.loc[nparts]
        ons_prec = row['tp_onsets'] / (row['tp_onsets'] + row['fp_onsets'])
        ons_rec = row['tp_onsets'] / (row['tp_onsets'] + row['fn_onsets'])
        acc_prec = row['tp_accents'] / (row['tp_accents'] + row['fp_accents'])
        acc_rec = row['tp_accents'] / (row['tp_accents'] + row['fn_accents'])
        ons_metrics[nparts] = (ons_prec, ons_rec)
        acc_metrics[nparts] = (acc_prec, acc_rec)
    

    n_parts = list(ons_metrics.keys())
    precision = [val[0] for val in ons_metrics.values()]  
    recall = [val[1] for val in ons_metrics.values()] 
    plt.ylim(0, 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    x = range(len(n_parts))
    ax.bar(x, precision, width=bar_width, label='Precision')
    ax.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall')
    ax.set_xlabel('Number of Parts')
    ax.set_ylabel('Score')
    ax.set_title('Metrics for Onset Detection per Number of Parts')
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(n_parts)
    ax.legend()
    plt.savefig("onset-model/results/onset_per_parts.png")
    plt.cla()

    n_parts = list(acc_metrics.keys())
    precision = [val[0] for val in acc_metrics.values()]  
    recall = [val[1] for val in acc_metrics.values()]
    plt.ylim(0, 1)
    ax.bar(x, precision, width=bar_width, label='Precision')
    ax.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall')
    ax.set_xlabel('Number of Parts')
    ax.set_ylabel('Score')
    ax.set_title('Metrics for Accented Onset Detection per Number of Parts')
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(n_parts)
    ax.legend()
    plt.savefig("onset-model/results/accent_per_parts.png")
    plt.cla()
    
    ons_metrics = {

    }
    acc_metrics = {
        
    }

    row = per_nparts_df.loc[1]
    ons_prec = row['tp_onsets'] / (row['tp_onsets'] + row['fp_onsets'])
    ons_rec = row['tp_onsets'] / (row['tp_onsets'] + row['fn_onsets'])
    acc_prec = row['tp_accents'] / (row['tp_accents'] + row['fp_accents'])
    acc_rec = row['tp_accents'] / (row['tp_accents'] + row['fn_accents'])
    ons_metrics['Monophonic'] = (ons_prec, ons_rec)
    acc_metrics['Monophonic'] = (acc_prec, acc_rec)

    tp_onsets, fp_onsets, fn_onsets, tp_accents, fp_accents, fn_accents = 0, 0, 0, 0, 0, 0
    for nparts in per_nparts_df.index.values.tolist()[1:]:
        row = per_nparts_df.loc[nparts]
        tp_onsets += row['tp_onsets']
        fp_onsets += row['fp_onsets']
        fn_onsets += row['fn_onsets']
        tp_accents += row['tp_accents']
        fp_accents += row['fp_accents']
        fn_accents += row['fn_accents']
    
    ons_prec = tp_onsets / (tp_onsets +fp_onsets)
    ons_rec = tp_onsets / (tp_onsets +fn_onsets)
    acc_prec = tp_accents / (tp_accents + fp_accents)
    acc_rec = tp_accents / (tp_accents + fn_accents)
    ons_metrics['Polyphonic'] = (ons_prec, ons_rec)
    acc_metrics['Polyphonic'] = (acc_prec, acc_rec)
    

    x = [0, 1]
    fig, ax = plt.subplots(figsize=(6, 6))
    n_parts = list(ons_metrics.keys())
    precision = [val[0] for val in ons_metrics.values()]  
    recall = [val[1] for val in ons_metrics.values()]
    plt.ylim(0, 1)
    ax.bar(x, precision, width=bar_width, label='Precision')
    ax.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall')
    ax.set_xlabel('Number of Parts')
    ax.set_ylabel('Score')
    ax.set_title('Metrics for Onset Detection, Monophonic v Polyphonic')
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(n_parts)
    ax.legend()
    plt.savefig("onset-model/results/onset_mvp.png")
    plt.cla()

    n_parts = list(acc_metrics.keys())
    precision = [val[0] for val in acc_metrics.values()]  
    recall = [val[1] for val in acc_metrics.values()]
    plt.ylim(0, 1)
    ax.bar(x, precision, width=bar_width, label='Precision')
    ax.bar([i + bar_width for i in x], recall, width=bar_width, label='Recall')
    ax.set_xlabel('Number of Parts')
    ax.set_ylabel('Score')
    ax.set_title('Metrics for Accented Onset Detection, Monophonic v Polyphonic')
    ax.set_xticks([i + bar_width / 2 for i in x])
    ax.set_xticklabels(n_parts)
    ax.legend()
    plt.savefig("onset-model/results/accent_mvp.png")
    plt.cla()

def dynamic(files):
    global instruments_df
    dynms = { 
        'ppp': [0, 0], 
        'pp': [0, 0], 
        'p': [0, 0], 
        'mp': [0, 0], 
        'normal': [0, 0], 
        'mf': [0, 0], 
        'f': [0, 0], 
        'ff': [0, 0], 
        'fff': [0, 0]
    }
    for file in files:
        name = file[0]
        if instruments_df.loc[name]['type'] == 'monophonic':
            true_onsets = file[2]
            guess_accents = file[3]
            true_accents = file[4]
            onset_list = [None for v in true_onsets]
            tp_accents = []
            for t in true_accents:
                for g in guess_accents:
                    if abs(g - t) <= 0.03:
                        tp_accents.append(t)
                        break
            
            fn_accents = [v for v in true_accents if v not in tp_accents]



            snippet = m21.converter.parse(os.path.join(os.getcwd(), f"data-pipeline/dataset/mxl/{name}.mxl"))
            count = 0
            for part in snippet.getElementsByClass(m21.stream.Part):
                currentDynamic = None
                for measure in part.getElementsByClass(m21.stream.Measure):
                    dynamics = []
                    for d in measure.getElementsByClass(m21.dynamics.Dynamic):
                        dynamics.append(d)

                    for note in measure.getElementsByClass(m21.note.Note):
                        if note.tie == None or note.tie.type == 'start':
                            if len(dynamics) > 0:
                                for d in dynamics:
                                    if d.offset <= note.offset:
                                        currentDynamic = d
                                        dynamic = currentDynamic
                                        break
                            articulations = note.articulations
                            if any(isinstance(articulation, m21.articulations.Accent) for articulation in note.articulations):
                                dynamic = currentDynamic
                                onset = true_onsets[count]
                                dynamic_name = 'normal'
                                if dynamic == None: dynamic_name = 'normal'
                                else: dynamic_name = dynamic.value
                                if onset in tp_accents:
                                    dynms[dynamic_name][0] += 1
                                else:
                                    dynms[dynamic_name][1] += 1
                                dynamic
                            count += 1


    to_delete = []
    for k in dynms.keys():
        if dynms[k][0] == 0 and dynms[k][1] == 0:
            to_delete.append(k)
    for k in to_delete: del dynms[k]

    precisions = []
    for k in dynms.keys():
        precisions.append(
            dynms[k][0] / (dynms[k][0] + dynms[k][1]) if dynms[k][1] != 0 else 0
        )

    print(dynms)
    print(precisions)

    plt.ylim(0, 1)
    plt.bar(list(dynms.keys()), precisions)
    plt.xlabel('Dynamic')
    plt.ylabel('Precision')
    plt.title('Precision of Accented Onset Detection Based on Note\'s Dynamic')
    plt.savefig("onset-model/results/prec_acc_dyn.png")
    plt.cla()


def pitch(files):
    global instruments_df
    octaves = { 
        '1': [0, 0], 
        '2': [0, 0], 
        '3': [0, 0], 
        '4': [0, 0], 
        '5': [0, 0], 
        '6': [0, 0]
    }
    
    for file in files:
        name = file[0]
        if instruments_df.loc[name]['type'] == 'monophonic':
            true_onsets = file[2]
            guess_accents = file[3]
            true_accents = file[4]
            onset_list = [None for v in true_onsets]
            tp_accents = []
            for t in true_accents:
                for g in guess_accents:
                    if abs(g - t) <= 0.03:
                        tp_accents.append(t)
                        break
            
            fn_accents = [v for v in true_accents if v not in tp_accents]



            snippet = m21.converter.parse(os.path.join(os.getcwd(), f"data-pipeline/dataset/mxl/{name}.mxl"))
            count = 0
            for part in snippet.getElementsByClass(m21.stream.Part):
                currentDynamic = None
                for measure in part.getElementsByClass(m21.stream.Measure):
                    for note in measure.getElementsByClass(m21.note.Note):
                        if note.tie == None or note.tie.type == 'start':
                            if any(isinstance(articulation, m21.articulations.Accent) for articulation in note.articulations):
                                onset = true_onsets[count]
                                octave = str(note.octave)
                                if octave == '1':
                                    octave
                                if onset in tp_accents:
                                    octaves[octave][0] += 1
                                else:
                                    octaves[octave][1] += 1
                                dynamic
                            count += 1
    

    precisions = []
    for k in octaves.keys():
        precisions.append(
            octaves[k][0] / (octaves[k][0] + octaves[k][1]) if octaves[k][1] != 0 else 0
        )

    plt.ylim(0, 1)
    plt.bar(list(octaves.keys()), precisions)
    plt.xlabel('Octave')
    plt.ylabel('Precision')
    plt.title('Precision of Accented Onset Detection Based on Note\'s Pitch')
    plt.savefig("onset-model/results/prec_acc_pitch.png")
    plt.cla()




os.chdir("../..")
files = os.listdir("onset-model/results/multilabel1")
for i in range(len(files)):
    file = files[i]
    guess_onsets = open(f"onset-model/results/multilabel1/{file}/guess_onset.txt", "r").read().split("\n")
    true_onsets =  open(f"data-pipeline/dataset/annotations/{file}/{file}.ONSETS", "r").read().split("\n")
    guess_accents = open(f"onset-model/results/multilabel1/{file}/guess_accent.txt", "r").read().split("\n")
    true_accents = open(f"data-pipeline/dataset/annotations/{file}/{file}.ACCENTS", "r").read().split("\n")

    guess_onsets = clean(guess_onsets)
    true_onsets = clean(true_onsets)
    guess_accents = clean(guess_accents)
    true_accents = clean(true_accents)
    files[i] = (file,guess_onsets, true_onsets,guess_accents,true_accents)

global instruments_df
instruments_df = pd.read_csv("data-pipeline/dataset/instruments.csv")
instruments_df.set_index('snippet', inplace=True)

num_parts(files)
dynamic(files)
pitch(files)

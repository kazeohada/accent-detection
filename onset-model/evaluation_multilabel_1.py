##############################################
# Original code by Bjöorn Lindqvist
# https://github.com/bjourne/onset-replication
##############################################

########################################################################
# Evaluation
# ==========
# This file contains functions for evaluating the models built.
########################################################################
from keras.models import load_model
from tensorflow import argmax
from sklearn.metrics import confusion_matrix
from madmom.evaluation.onsets import OnsetEvaluation
from os.path import join
import numpy as np
import pandas as pd
import os

########################################################################
# Evaluation settings
########################################################################
ONSET_EVAL_WINDOW = 0.025
# Combine onsets detected less than 30 ms apart.
ONSET_EVAL_COMBINE = 0.03


def sum_evaluation(evals):
    tp = sum(e.num_tp for e in evals)
    fp = sum(e.num_fp for e in evals)
    tn = sum(e.num_tn for e in evals)
    fn = sum(e.num_fn for e in evals)

    # Do some calucations
    if (tp == 0): # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
        if (fp > 0 or fn > 0):
            prec, rec, f_measure = 0, 0, 0
        else:
            prec, rec, f_measure = 1, 1, 1
    else:
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f_measure = 2 * prec * rec / (prec + rec)

    ret = '  #: %6d TP: %6d FP: %5d FN: %5d\n' \
        % (tp + fn, tp, fp, fn)
    ret += '  Prec: %.3f Rec: %.3f F-score: %.3f' \
        % (prec, rec, f_measure)
    return ret, f"{prec:.3f}", f"{rec:.3f}", f"{f_measure:.3f}"

def mean_evaluation(evals):
    n_anns = np.mean([e.num_annotations for e in evals])
    tp = np.mean([e.num_tp for e in evals])
    fn = np.mean([e.num_fn for e in evals])
    fp = np.mean([e.num_fp for e in evals])
    prec = np.mean([e.precision for e in evals])
    recall = np.mean([e.recall for e in evals])
    f_measure = np.mean([e.fmeasure for e in evals])
    ret = 'sum for %d files\n' % len(evals)
    ret += '  #: %5.2f TP: %6.2f FP: %5.2f FN: %5.2f\n' \
        % (n_anns, tp, fp, fn)
    ret += '  Prec: %.3f Rec: %.3f F-score: %.3f' \
        % (prec, recall, f_measure)
    return ret

def write_audacity_labels(name, onsets, accents):
    if not os.path.exists(f"results/multilabel1/{name}"):
        os.makedirs(f"results/multilabel1/{name}")

    with open(f"results/multilabel1/{name}/guess_onset.txt", "w") as file:
        for o in onsets:
            file.write(f"{o}\t{o}\n")
    
    with open(f"results/multilabel1/{name}/guess_accent.txt", "w") as file:
        for a in accents:
            file.write(f"{a}\t{a}\n")


def evaluate_audio_sample(nn, model, d, epoch):
    a = d.a
    a_ons = d.a[0] / 100 # convert to seconds
    a_acc = d.a[1] / 100 # convert to seconds
    x = nn.samples_in_audio_sample(d)
    y_guess = model.predict(x)
    y_ons_guess = y_guess[:, 0].squeeze()
    y_acc_guess = y_guess[:, 1].squeeze()

    a_ons_guess, a_acc_guess = nn.postprocess_y(y_ons_guess, y_acc_guess, 0.5, 0.15)

    # onset evaluation
    ons_eval = OnsetEvaluation( 
        a_ons_guess, a_ons,
        combine = ONSET_EVAL_COMBINE,
        window = ONSET_EVAL_WINDOW)
    # accent evaluation
    acc_eval = OnsetEvaluation( 
        a_acc_guess, a_acc,
        combine = ONSET_EVAL_COMBINE,
        window = ONSET_EVAL_WINDOW)
    
    if epoch == 'best_val':
        write_audacity_labels(
            d.name, 
            np.concatenate((ons_eval.tp, ons_eval.fp)),
            np.concatenate((acc_eval.tp, acc_eval.fp))
        )

    return ons_eval, acc_eval

def evaluate_fold(nn, model, fold, epoch):
    for audio_sample in fold:
        print(audio_sample.name, end = ' ', flush = True)
        ons_eval, acc_eval = evaluate_audio_sample(nn, model, audio_sample, epoch)
        yield (ons_eval, acc_eval)
         

def evaluate_folds(nn, folds, int_range, output_dir, epoch):
    onset_evals = []
    accent_evals = []        
    all_evals = []
    fmt = '[%d/%d] Evaluating fold, '
    n_folds = len(folds)
    model_file = f'model_{epoch}.h5'
    for i in int_range:
        print(fmt % (i + 1, n_folds), end = '', flush = True)
        model_path = join(output_dir, '%02d' % i, model_file)
        print('loading %s, ' % model_file, end = '', flush = True)
        model = load_model(model_path)
        fold = folds[i]
        evals = []
        evals.extend(evaluate_fold(nn, model, fold, epoch))
        for eval in evals:
            onset_evals.append(eval[0])
            accent_evals.append(eval[1])

    print('DONE')
    all_evals += onset_evals
    all_evals += accent_evals
    return all_evals, onset_evals, accent_evals

def evaluate(nn, folds, int_range, output_dir):
    eval_df = pd.DataFrame(columns=['epoch', 
                                'onset precision', 'onset recall', 'onset f-score',
                                'accent precision', 'accent recall', 'accent f-score',
                                'total precision', 'total recall', 'total f-score'])


    for i in range(1, 21):
        all_evals, onset_evals, accent_evals = evaluate_folds(nn, folds, int_range, output_dir, str(i).zfill(3))
        all_print, all_prec, all_rec, all_f = sum_evaluation(all_evals)
        ons_print, ons_prec, ons_rec, ons_f = sum_evaluation(onset_evals)
        acc_print, acc_prec, acc_rec, acc_f = sum_evaluation(accent_evals)
        new_row = [i, ons_prec, ons_rec, ons_f, acc_prec, acc_rec, acc_f, all_prec, all_rec, all_f]
        eval_df.loc[len(eval_df)] = new_row

    
    all_evals, onset_evals, accent_evals = evaluate_folds(nn, folds, int_range, output_dir, 'best_val')
    all_print, all_prec, all_rec, all_f = sum_evaluation(all_evals)
    ons_print, ons_prec, ons_rec, ons_f = sum_evaluation(onset_evals)
    acc_print, acc_prec, acc_rec, acc_f = sum_evaluation(accent_evals)
    new_row = ['best', ons_prec, ons_rec, ons_f, acc_prec, acc_rec, acc_f, all_prec, all_rec, all_f]
    eval_df.loc[len(eval_df)] = new_row

    print("Evaluation of all labels")
    print(all_print)
    print("\nEvaluation of detection of  all onsets")
    print(ons_print)
    print("\nEvaluation of detection of accents")
    print(acc_print)

    eval_df.to_csv("results/eval_multilabel1.csv", sep=",")
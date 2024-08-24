import music21 as m21
import random, threading
from operator import xor
import pandas as pd
from math import floor



def clean(piece : m21.stream.Score):

    for part in piece.getElementsByClass(m21.stream.Part):

        # correct time signatures
        actualTimeSignature = None
        for measure in part.getElementsByClass(m21.stream.Measure):
            barDuration = measure.highestTime
            if measure.timeSignature != None:
                actualTimeSignature = measure.timeSignature
            if barDuration != actualTimeSignature.barDuration.quarterLength:
                n = 2
                t = barDuration
                while floor(t) != t:
                    t = t * 2
                    n += 1
                measure.timeSignature = m21.meter.TimeSignature(f"{int(t)}/{2**n}")
            else: 
                measure.timeSignature = m21.meter.TimeSignature(f"{actualTimeSignature.numerator}/{actualTimeSignature.denominator}")
    
    tempo_expressions = {
        "Grave": 35,
        "Largo": 50,
        "Lento": 53, 
        "Larghetto": 63,
        "Adagio": 71, 
        "Andante": 92,
        "Andantino": 94,
        "Moderato": 114,
        "Allegretto": 116,
        "Allegro": 144,
        "Vivace": 172,
        "Presto": 187,
        "Prestissimo": 200
    }


    for part in piece.getElementsByClass(m21.stream.Part):
        
        firstMetronomeVisited = False

        # for every measure in part
        for measure in part.getElementsByClass(m21.stream.Measure):
            # remove chord symbols
            for chord in measure.getElementsByClass(m21.harmony.ChordSymbol):
                measure.remove(chord)


            for rest in measure.getElementsByClass(m21.note.Rest):

                # remove all expressions and articulations, includes Fermata
                rest.expressions = []
                rest.articulations = []

           
            for note in measure.getElementsByClass(m21.note.Note):
                
                # remove all expressions and articulations
                note.expressions = []
                note.articulations = []

                # remove grace notes
                if note.duration.isGrace:
                    measure.remove(note)
            
            for chord in measure.getElementsByClass(m21.chord.Chord):
                # remove all expressions and articulations
                chord.expressions = []
                chord.articulations = []

            # remove glissandos
            for gliss in measure.getElementsByClass(m21.spanner.Glissando):
                measure.remove(gliss)
            # remove tempo changes 
            for expression in measure.getElementsByClass(m21.expressions.TextExpression):
                if expression.content in tempo_expressions:
                    tempo = tempo_expressions[expression.content]
                    measure.insert(expression.offset, m21.tempo.MetronomeMark(tempo))
                measure.remove(expression)
            
            # remove repeats
            for repeat in measure.getElementsByClass(m21.bar.Repeat):
                measure.remove(repeat)
            for repeatMarker in measure.getElementsByClass(m21.repeat.RepeatExpressionMarker):
                measure.remove(repeatMarker)
            for repeatCommand in measure.getElementsByClass(m21.repeat.RepeatExpressionCommand):
                measure.remove(repeatCommand)
            
            # remove subsequent metrononmes
            for metronome in measure.getElementsByClass(m21.tempo.MetronomeMark):
                if (firstMetronomeVisited):
                    measure.remove(metronome)
                else: 
                    firstMetronomeVisited = True
        
        for repeat in part.getElementsByClass(m21.spanner.RepeatBracket):
            part.remove(repeat)
        
        for gliss in part.getElementsByClass(m21.spanner.Glissando):
            part.remove(gliss)
                
    return piece

def find_qbpm(piece):
    qbpm = 120
    for part in piece.getElementsByClass(m21.stream.Part):
        for measure in part.getElementsByClass(m21.stream.Measure):
            for metronome in measure.getElementsByClass(m21.tempo.MetronomeMark):
                qbpm = metronome.getQuarterBPM()
                return qbpm
    return qbpm

def get_measure_index_with_offset(piece : m21.stream.Score, offset):
    for i, measure in enumerate(piece.parts[0].getElementsByClass(m21.stream.Measure)[:-1]):
        if measure.offset <= offset and offset <= piece.parts[0].getElementsByClass(m21.stream.Measure)[i+1].offset:
            return i
    return None

def make_snippets(piece : m21.stream.Score):

    piecetempo = find_qbpm(piece)

    snippets = []
    tempos = []

    first_measure, first_offset, last_measure, last_offset = 0, 0, 0, 0

    while last_measure != None:

        tempo = random.randrange(max(15, round(piecetempo - 35)), min(299, round(piecetempo + 35)))

        duration = random.randrange(5, 20) # duration of snippet in seconds
        
        # how many quarter notes fit in 'duration' seconds at 'tempo' bpm in 4/4?
        num_qn = floor(( tempo / 60 ) * duration)
        last_offset = first_offset + num_qn
        last_measure = get_measure_index_with_offset(piece, last_offset)

        if (last_measure != None):
            snippet = piece.measures(first_measure, last_measure + 1, indicesNotNumbers=True)
            tempos.append(tempo)
            snippets.append(snippet)

            first_measure = last_measure + 1
            first_offset = piece.parts[0].getElementsByClass(m21.stream.Measure)[first_measure].offset



    # for each snippet, take each part's last dynamic and append to the beginning of the next snippet
    for i, snippet in enumerate(snippets[:-1]):
        for j, part in enumerate(snippet.parts):
            last_dynamic = None
            for measure in part.getElementsByClass(m21.stream.Measure):
                if len(measure.getElementsByClass(m21.dynamics.Dynamic)) > 0:
                    last_dynamic = measure.getElementsByClass(m21.dynamics.Dynamic)[-1]
            if last_dynamic != None:
                ms = snippets[i+1].parts[j].getElementsByClass(m21.stream.Measure)

                while ms.elementsLength == 0:
                    snippets.remove(snippets[i+1])
                    tempos.remove(tempos[i+1])
                    if i + 1 >= len(snippets): break
                    ms = snippets[i+1].parts[j].getElementsByClass(m21.stream.Measure)
                    
                if i + 1 < len(snippets):
                    ms[0].insert(0, m21.dynamics.Dynamic(last_dynamic.value))
                else: break

    return snippets, tempos

def set_tempo(part, tempo):
    tempoIsSet = False
    for measure in part.getElementsByClass(m21.stream.Measure):
        metronomes = measure.getElementsByClass(m21.tempo.MetronomeMark)
        if len(metronomes) > 0:
            for metronome in metronomes:
                metronome.setQuarterBPM(tempo)
                tempoIsSet = True

    if (not tempoIsSet):
        part.insert(0.0, m21.tempo.MetronomeMark(number=tempo))


instruments_df = pd.DataFrame(columns=['snippet', 'type'])
lock = threading.Lock()

def count_instruments(piece, name):
    global instruments_df
    lock.acquire(blocking=True)
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
    lock.release()
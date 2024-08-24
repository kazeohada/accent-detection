import music21 as m21

def set_musicxml_path():
    us = m21.environment.UserSettings()
    us['musicxmlPath'] = 'C:/Program Files/MuseScore 4/bin/MuseScore4.exe'
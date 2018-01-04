import os.path

transcriptions = {}

def load():
    path = 'ground-truth/transcription.txt'
    dirname = os.path.dirname(os.path.abspath(__file__))
    f = open(os.path.join(dirname, path), 'r')
    for line in f:
        id, trans = line.split()
        transcriptions[id] = trans


def transcription(id):
    """
    :param id (str): ID of word image, e.g., "270-04-01"
    :return: (str) transcription of word image, e.g., "l-a-r"
    """
    if not transcriptions:
        load()

    return transcriptions[id]
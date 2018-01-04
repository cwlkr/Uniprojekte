from os.path import splitext
from os.path import basename

def basename_without_ext(path):
    """
    :param path (str): e.g. "foo/bar.jpg" 
    :return (str): "bar"
    """
    return splitext(basename(path))[0]
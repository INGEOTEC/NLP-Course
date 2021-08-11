import re
from typing import Tuple


def start_project(txt: str) -> Tuple[int, int]:
    """Use regular expressions to find 
    the start and end point of the line *** START OF THE PROJECT ... ***.
    The ... indicates any text except the carriage return. 
    Usage: 

    >>> start_project("*** START OF THE PROJECT GUTENBERG EBOOK THE ILIAD ***")
    (0, 54)
    >>> start_project("Hi! *** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE *** Excellent book")
    (4, 77)

    This function will be tested on two of the 25 most downloaded books from https://www.gutenberg.org. 
    The format used is txt and the file is read using open(fname).read()"""
import re
from typing import Tuple
from collections import Counter


def start_project(txt: str) -> Tuple[int, int]:
    """Use regular expressions to find 
    the start and end point of the line *** START OF THE PROJECT ... ***.
    The ... indicates any text except the carriage return. 
    Usage: 

    >>> start_project("*** START OF THE PROJECT GUTENBERG EBOOK THE ILIAD ***")
    (0, 54)
    >>> start_project("Hi! *** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE *** Excellent book")
    (4, 77)

    This function will be tested on two of the 
    25 most downloaded books from https://www.gutenberg.org. 
    The format used is txt and the file is read using open(fname).read()"""


def entity_identification(txt: str) -> Counter:
    """Count the number of appearances of each entity using 
    the simple heuristic of identifying as an entity 
    each word starting with a capital letter or an abbreviation.
    Usage:
    
    >>> entity_identification("America is a continent.")
    Counter({'America': 1})
    >>> entity_identification("The United States is in America")
    Counter({'The': 1, 'United': 1, 'States': 1, 'America': 1})
    >>> entity_identification("U.S.A. is in America and Mexico is in America")
    Counter({'America': 2, 'U.S.A.': 1, 'Mexico': 1})
    >>> entity_identification("USA. U.S.A America.")
    Counter({'USA.': 1, 'U.S.A': 1, 'America.': 1})
    """
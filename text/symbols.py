""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.
"""
_pad = "_"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "x",
    "y",
    "đ",
    "̀",
    "́",
    "̂",
    "̃",
    "̆",
    "̉",
    "̛",
    "̣",
]


# Export all symbols:
symbols = [_pad] + list(_punctuation) + _letters

# Special symbol ids
SPACE_ID = symbols.index(" ")

# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 15:54:03 2018

@author: Ansam
"""

__author__ = 'KKishore'

PAD_WORD = '#=KISHORE=#'

HEADERS = ['CATEGORY', 'TITLE']
FEATURE_COL = 'TITLE'
LABEL_COL = 'CATEGORY'
WEIGHT_COLUNM_NAME = 'weight'
TARGET_LABELS = ['t', 'b','m','e']
TARGET_SIZE = len(TARGET_LABELS)
HEADER_DEFAULTS = [['NA'], ['NA']]

MAX_DOCUMENT_LENGTH = 100
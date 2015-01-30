"""
Calculates the precision of the word predictor using
the Gutenberg corpus from the nltk library.
"""
import sys
sys.path.append('../word_predictor')
import math
from collections import deque
from word_predictor.word_predictor import WordPredictor
import nltk

wp = WordPredictor()
gutenberg_files = nltk.corpus.gutenberg.fileids()
# Train on 75 % of files (not necessarily 75 % of sentences)
training_set_size = int(math.floor(len(gutenberg_files) * 0.75))
for corpus in gutenberg_files[0:training_set_size]:
    wp.learn_from_text(nltk.corpus.gutenberg.raw(corpus))
# Test
num_correct = 0
num_total = 0
print "Trained..."
for corpus in gutenberg_files[training_set_size:]:
    text = nltk.corpus.gutenberg.raw(corpus)
    terms = wp._tokenize_phrase(text)
    recent = deque([], maxlen=wp.order)
    # Estimate for each pair of words
    for term in terms:
        tokens = wp._predict_from_tokens(list(recent))
        # If one of top 3 predictions is word, count as correct prediction
        if term in map(lambda x: x[0], tokens[0:3]):
            num_correct += 1
        num_total += 1
        recent.append(term)

print "%d / %d" % (num_correct, num_total)

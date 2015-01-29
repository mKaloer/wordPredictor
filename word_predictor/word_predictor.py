from collections import deque
from scipy.sparse import dok_matrix
import numpy as np
import re

class WordPredictor(object):
    """Main entry point for word predictions."""

    def __init__(self, order=2, case_sensitive=True, vocab_size=65536):
        """Instantiate new word predictor.

        Arguments:
        order -- The order of the underlying Markov chain,
                 i.e. the number of terms to take into account.
        case_sensitive -- Whether or not the predictor is case sensitive.
        vocab_size -- The maximum number of unique words.
        """
        self.order = order
        self.case_sensitive = case_sensitive
        self.vocab_size = vocab_size
        # Mapping from term to id
        self._id_lookup = {}
        # Mapping from id to term
        self._term_lookup = {}
        self._id_ctr = 0

        # The matrix size is vocabulary size times the order of the markov chain.
        # We represent a n'th order markov chain as a 1st order chain with state
        # representation [k-1, n-2, ..., k-n].
        trans_dim = pow(self.vocab_size, self.order)
        self._transitions = dok_matrix((trans_dim, trans_dim), dtype=np.uint32)

    def _tokenize_phrase(self, phrase):
        """Tokenizes the given phrase."""
        tokens = re.findall(r"[\w']+|[.,!?;]", phrase)
        if not self.case_sensitive:
            return [x.lower() for x in tokens]
        else:
            return tokens

    def learn_from_sentence(self, sentence):
        """Learn from the given sentence"""
        # If first word of sentence, previous is the empty string
        hash_deq = deque([], maxlen=self.order)
        for word in self._tokenize_phrase(sentence):
            if not word in self._id_lookup:
                if self._id_ctr > self.vocab_size:
                    # Skip if vocabulary size is exceeded
                    continue
                self._id_lookup[word] = self._id_ctr
                self._term_lookup[self._id_ctr] = word
                self._id_ctr += 1
            str_hash = self._id_lookup[word]
            state = 0
            # Add history for n previous words, 0 < n <= order of Markov chain
            for order in reversed(range(0, len(hash_deq))):
                # Calculate state hash
                state = state * self.vocab_size + hash_deq[order]
                # Update counts for state
                self._transitions[state, str_hash] += 1
            hash_deq.append(str_hash)

    def predict(self, text):
        """Predict a number of following word candidates based on the given text.

        Arguments:
        text -- The temporary phrase

        Returns predicted words as an array of (word, probability) pairs
        """
        tokens = self._tokenize_phrase(text)
        str_hash = 0
        # Find ids for n previous words
        for n in range(0, self.order):
            if len(tokens) >= n+1 and tokens[len(tokens)-1-n] in self._id_lookup:
                str_hash = (str_hash * self.vocab_size) + \
                           self._id_lookup[tokens[len(tokens)-1-n]]
        row = self._transitions[str_hash, :]
        nonzero = row.nonzero()
        # P(k-1, k-2,...k-n)
        state_tot = 0#self._transitions[h,:].sum()
        for elmt in nonzero[1]:
            state_tot += row[0, elmt]
        # Find P(k,k-1,..k-n)
        terms = []
        for elmt in nonzero[1]:
            val = row[0, elmt]
            prob = val / float(state_tot)
            terms.append((self._term_lookup[elmt], prob))
        return sorted(terms, key=lambda x: (-x[1], x[0]))

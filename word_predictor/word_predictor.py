from collections import deque
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
from nltk.tokenize import wordpunct_tokenize
import marisa_trie

class WordPredictor(object):
    """Main entry point for word predictions."""

    def __init__(self, order=2, case_sensitive=True, vocab_size=10000):
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
        # The matrix size is vocabulary size to the power of the order of the markov chain.
        # We represent a n'th order markov chain as a 1st order chain with state
        # representation [k-1, n-2, ..., k-n].
        self._transitions = dok_matrix((pow(self.vocab_size, self.order), vocab_size), dtype=np.uint32)
        self._transitions_csr = None

    def _tokenize_phrase(self, phrase):
        """Tokenizes the given phrase."""
        tokens = wordpunct_tokenize(phrase)
        if not self.case_sensitive:
            return [x.lower() for x in tokens]
        else:
            return tokens

    def learn_from_text(self, text):
        """Learn from the given text"""
        # If first word of text, previous is the empty string
        hash_deq = deque([], maxlen=self.order)
        for word in self._tokenize_phrase(text):
            if not word in self._id_lookup:
                if self._id_ctr >= self.vocab_size:
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
        # Invalidate eventually generated csr matrix
        self._transitions_csr = None

    def predict(self, text):
        """Predict a number of following word candidates based on the given text.

        Arguments:
        text -- The temporary phrase

        Returns the predicted words as a Patricia-trie (PatriciaTrieWrapper instance).
        """
        tokens = self._tokenize_phrase(text)
        return self._predict_from_tokens(tokens)

    def _predict_from_tokens(self, tokens):
        """Predict a number of following word candidates based on the given tokens.

        Arguments:
        text -- The temporary phrase

        Returns the predicted words as a Patricia-trie (PatriciaTrieWrapper instance).
        """
        str_hash = 0
        # Find ids for n previous words
        for n in range(0, self.order):
            if len(tokens) >= n+1 and tokens[len(tokens)-1-n] in self._id_lookup:
                str_hash = (str_hash * self.vocab_size) + \
                           self._id_lookup[tokens[len(tokens)-1-n]]
        # Convert matrix to csr for faster calculations
        if self._transitions_csr == None:
            self._transitions_csr = csr_matrix(self._transitions)
        row = self._transitions_csr[str_hash, :]
        nonzero = row.nonzero()
        # P(k, k-1,...k-n+1)
        state_tot = row.sum()
        # Find P(k+1,k,k-1,..k-n+1)
        terms = {}
        for elmt in nonzero[1]:
            val = row[0, elmt]
            prob = val / float(state_tot)
            terms[self._term_lookup[elmt]] = prob

        # Convert to Patricia-trie
        return PatriciaTrieWrapper(terms)

class PatriciaTrieWrapper(object):
    """
    Patricia Trie (wrapper of the marisa_trie class)
    """

    def __init__(self, terms):
        self._terms = terms
        self._trie = marisa_trie.Trie(terms.keys())

    def terms(self, prefix=""):
        """Get the terms ordered by probability, starting by given prefix.

        Arguments:
        prefix -- The prefix of the terms.

        Returns a sorted list of predicted words starting with the specified
        prefix, as an array of (word, probability) pairs. Notice that if a
        prefix is specified, the probabilities may not be normalized for
        performance reasons.
        """
        # Array of (term, probability)
        terms = []
        for w in self._trie.iterkeys(unicode(prefix)):
            terms.append((w, self._terms[w]))
        return sorted(terms, key=lambda x: (-x[1], x[0]))

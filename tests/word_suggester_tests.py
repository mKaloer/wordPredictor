from word_suggester.word_suggester import WordSuggester
from nose.tools import *

class TestWordSuggester():

    @classmethod
    def setup_class(cls):
        cls.corpus = [
            "Love all, trust a few, do wrong to none.",
            "To be, or not to be: that is the question.",
            "There is nothing either good or bad, but thinking makes it so.",
            "To mourn a mischief that is past and gone is the next way to draw new mischief on",
             "That that is is that that is not is not is that it it is."
        ]


    @classmethod
    def assert_almost_eq_suggestions(cls, suggestions, expected):
        # Test manually due to floating point precision
        eq_(len(expected), len(suggestions), "Lists not of equal length")
        for i,s in enumerate(expected):
            eq_(s[0], suggestions[i][0])
            assert_almost_equal(s[1], suggestions[i][1])

    def test_first_order(self):
        ws = WordSuggester(order=1, case_sensitive=True)
        for s in self.corpus:
            ws.learn_from_sentence(s)
        suggestions = ws.suggest("that is")
        expected = [
            ("not", 0.2),
            ("that", 0.2),
            ("the", 0.2),
            (".", 0.1),
            ("is", 0.1),
            ("nothing", 0.1),
            ("past", 0.1),
        ]
        eq_(expected, suggestions)

    def test_second_order_regular(self):
        ws = WordSuggester(order=2, case_sensitive=True)
        for s in self.corpus:
            ws.learn_from_sentence(s)
        suggestions = ws.suggest("that is")
        expected = [
            ("is", 0.25),
            ("not", 0.25),
            ("past", 0.25),
            ("the", 0.25)
        ]
        eq_(expected, suggestions)

    def test_second_order_single_unknown_word(self):
        ws = WordSuggester(order=2, case_sensitive=True)
        for s in self.corpus:
            ws.learn_from_sentence(s)
        suggestions = ws.suggest("that")
        expected = [
            ("is", 0.66666667),
            ("it", 0.16666667),
            ("that", 0.16666667)
        ]
        self.assert_almost_eq_suggestions(expected, suggestions)

    def test_first_order_case_sensitive(self):
        ws = WordSuggester(order=1, case_sensitive=True)
        for s in self.corpus:
            ws.learn_from_sentence(s)
        suggestions = ws.suggest("That")
        expected = [
            ("that", 1.0)
        ]
        eq_(expected, suggestions)

    def test_first_order_case_insensitive(self):
        ws = WordSuggester(order=1, case_sensitive=False)
        for s in self.corpus:
            ws.learn_from_sentence(s)
        suggestions = ws.suggest("That")
        expected = [
            ("is", 0.571428571),
            ("that", 0.285714286),
            ("it", 0.142857143)
        ]
        self.assert_almost_eq_suggestions(expected, suggestions)

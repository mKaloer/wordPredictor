# -*- coding: utf-8 -*-
from word_predictor.word_predictor import WordPredictor
from nose.tools import *

class TestWordPredictor(object):

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
    def assert_almost_eq_predictions(cls, predictions, expected):
        # Test manually due to floating point precision
        eq_(len(expected), len(predictions), "Lists not of equal length")
        for i,s in enumerate(expected):
            eq_(s[0], predictions[i][0])
            assert_almost_equal(s[1], predictions[i][1])

    def test_first_order(self):
        wp = WordPredictor(order=1, case_sensitive=True)
        for s in self.corpus:
            wp.learn_from_text(s)
        predictions = wp.predict("that is").terms()
        expected = [
            ("not", 0.2),
            ("that", 0.2),
            ("the", 0.2),
            (".", 0.1),
            ("is", 0.1),
            ("nothing", 0.1),
            ("past", 0.1),
        ]
        eq_(expected, predictions)

    def test_second_order_regular(self):
        wp = WordPredictor(order=2, case_sensitive=True)
        for s in self.corpus:
            wp.learn_from_text(s)
        predictions = wp.predict("that is")
        expected = [
            ("is", 0.25),
            ("not", 0.25),
            ("past", 0.25),
            ("the", 0.25)
        ]
        eq_(expected, predictions.terms())

    def test_second_order_single_unknown_word(self):
        wp = WordPredictor(order=2, case_sensitive=True)
        for s in self.corpus:
            wp.learn_from_text(s)
        predictions = wp.predict("that")
        expected = [
            ("is", 0.66666667),
            ("it", 0.16666667),
            ("that", 0.16666667)
        ]
        self.assert_almost_eq_predictions(expected, predictions.terms())

    def test_first_order_case_sensitive(self):
        wp = WordPredictor(order=1, case_sensitive=True)
        for s in self.corpus:
            wp.learn_from_text(s)
        predictions = wp.predict("That")
        expected = [
            ("that", 1.0)
        ]
        eq_(expected, predictions.terms())

    def test_first_order_case_insensitive(self):
        wp = WordPredictor(order=1, case_sensitive=False)
        for s in self.corpus:
            wp.learn_from_text(s)
        predictions = wp.predict("That")
        expected = [
            ("is", 0.571428571),
            ("that", 0.285714286),
            ("it", 0.142857143)
        ]
        self.assert_almost_eq_predictions(expected, predictions.terms())


    def test_unicode(self):
        wp = WordPredictor(order=1, case_sensitive=False)
        wp.learn_from_text(u"This is a test 👮")
        predictions = wp.predict("This is a test")
        expected = [
            (u"👮", 1.0),
        ]
        eq_(expected, predictions.terms())

    def test_prefix(self):
        wp = WordPredictor(order=1, case_sensitive=True)
        for s in self.corpus:
            wp.learn_from_text(s)
        predictions = wp.predict("that is").terms("th")
        expected = [
            ("that", 0.2),
            ("the", 0.2)
        ]
        eq_(expected, predictions)

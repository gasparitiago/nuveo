import unittest
from spam_detector import SpamDetector, SpamDetectorMode


class TestSpamDetector(unittest.TestCase):
    spam_detector = SpamDetector('Model/sms_model_v1.pkl')
    spam_mesage = "Dorothy@kiefer.com (Bank of Granite issues Strong-Buy) EXPLOSIVE PICK FOR OUR MEMBERS *****UP OVER 300% *********** Nasdaq Symbol CDGT That is a $5.00 per.."
    ham_mesage = "If you're thinking of lifting me one then no."
        
    def test_is_spam(self):
        spam = self.spam_detector.is_spam(self.spam_mesage)
        ham = self.spam_detector.is_spam(self.ham_mesage)
        self.assertTrue(spam)
        self.assertFalse(ham)
    
    def test_prob_spam(self):
        spam = self.spam_detector.prob_spam(self.spam_mesage)
        ham = self.spam_detector.prob_spam(self.ham_mesage)
        self.assertTrue(spam > 0.5)
        self.assertTrue(ham < 0.5)

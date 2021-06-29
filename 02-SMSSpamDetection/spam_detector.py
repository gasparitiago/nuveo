import pickle
from enum import Enum


THRESHOLD_NO_FALSE_POSITIVE = 0.05
THRESHOLD_NO_FALSE_NEGATIVE = 0.7


class SpamDetectorMode(Enum):
    """
    Enumeration to determine different modes to
    determine if a message is spam or not.

    default: analyzes which probability is higher: ham or spam
    no_spam_allowed: just consider that a message is spam if
                     the probability is higher than 90%
    no_ham_as_spam: just consider that a message is spam if
                    the probability of if is lower than 90%
    """
    default = 0
    no_spam_allowed = 1
    no_ham_as_spam = 2

class SpamDetector:

    def __init__(self, model_path):
        self.sms_model = pickle.load(open(model_path, "rb"))
    
    def prob_spam(self, input_message):
        """
        Returns the probability of a message being spam.
        Keyword arguments:
        input_message -- input sms message
        """
        prob_value = self.sms_model.predict_proba([input_message])[0][1]
        return prob_value

    def is_spam(self, input_message, mode=SpamDetectorMode.default):
        """
        Returns if a message is spam (True) or if it's a ham message (False).
        Keyword arguments:
        input_message -- input sms message
        mode -- one of the SpamDetectorMode choices
        """
        if mode == SpamDetectorMode.default:
            return self.sms_model.predict([input_message])[0] == 'spam'
        elif mode == SpamDetectorMode.no_spam_allowed:
            return self.sms_model.predict_proba([input_message])[0][1] >= THRESHOLD_NO_FALSE_POSITIVE
        return self.sms_model.predict_proba([input_message])[0][1] >= THRESHOLD_NO_FALSE_NEGATIVE

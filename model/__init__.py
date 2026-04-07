"""
Auteur: de clankers
Date: 7 april 2026
"""

from .classifier import CHDClassifier

def model_factory():
    # Hier ga je normaal je al getrainde model laden
    # Voor de competitie train je lokaal en sla je bv. een pickle op
    # of je traint in __init__ van CHDClassifier.
    return CHDClassifier()

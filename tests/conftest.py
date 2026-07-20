"""
Configuration partagee par tous les tests.

pytest charge ce fichier AVANT les modules de test, ce qui en fait le bon
endroit pour deux choses :
  1. rendre le projet importable (src/, app/) sans bidouiller sys.path partout
  2. importer src.config, qui pose USE_TF=0 et compagnie, pour que transformers
     ne tente jamais de charger TensorFlow, quel que soit l'ordre des imports
"""

import os
import sys

# Racine du projet, pour que "import src..." et "import app..." fonctionnent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cet import a un effet de bord voulu : il pose les variables d'environnement
# qui desactivent TensorFlow / Flax dans transformers.
import src.config  # noqa: E402,F401

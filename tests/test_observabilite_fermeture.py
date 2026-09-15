"""Non-regression : fermer puis recreer le client ne doit jamais bloquer.

Les clients Langfuse partagent un gestionnaire de ressources par cle
publique. Un `shutdown()` arrete ses threads d'envoi ; un nouveau client avec
la meme cle le reutilise, et la fermeture suivante attend indefiniment une
file sans consommateur. C'est ce qui bloquait tests/test_api.py, ou chaque
test demarre puis arrete l'application.

Le cycle tourne dans un thread avec delai : en cas de regression, ce test
echoue en 20 secondes au lieu de figer toute la suite.
"""

import threading

from app import observabilite


def test_fermer_puis_recreer_ne_bloque_pas() -> None:
    def cycles() -> None:
        for n in range(5):
            with observabilite.observer(f"etape-{n}"):
                pass
            observabilite.fermer()

    fil = threading.Thread(target=cycles, daemon=True)
    fil.start()
    fil.join(timeout=20)

    assert not fil.is_alive(), "fermer() bloque apres plusieurs cycles"

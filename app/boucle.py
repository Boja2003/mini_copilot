"""Correctif de boucle d'evenements pour Windows.

psycopg en mode asynchrone refuse la boucle par defaut de Windows
(ProactorEventLoop) et exige SelectorEventLoop. Sur Linux — donc dans le
conteneur et sur Fly.io — il n'y a rien a faire : ce module est alors sans
effet. C'est uniquement le confort du developpement local sous Windows.
"""

import asyncio
import sys


def configurer() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

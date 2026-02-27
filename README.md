# Projet de Najib et Victor

Un super projet sur les espaces de configuration.

## Package `calculs/uconf`

Le dossier `calculs/uconf` contient des modèles combinatoires d'opérades utilisés
dans les calculs.

- `surjection.py` : opérade des surjections (`Surjection`).
- `barratt_eccles.py` : opérade de Barratt-Eccles (`BarrattEccles`).
- `lie.py` : composantes de l'opérade de Lie (`Lie`).
- `bar.py` : construction bar d'une opérade (`BarConstruction`).
- `operad.py` : protocole de typage (`OperadProtocol`) pour homogénéiser l'API.

### API commune (éléments)

Les classes d'éléments exposent notamment :

- `arity()`
- `boundary()`
- `permute(...)`

Selon le modèle, on trouve aussi :

- `planarize()`
- `complexity()`
- `diagonal()` (Barratt-Eccles)

### Applications et morphismes déjà branchés

Au chargement du package (`import uconf`), deux applications sont attachées
dynamiquement :

- `BarrattEccles.Element.table_reduction() -> Surjection.Element`
- `Surjection.Element.section() -> BarrattEccles.Element`

Ces morphismes sont construits paresseusement et mis en cache au niveau des
parents (`module_morphism`).

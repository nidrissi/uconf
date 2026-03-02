# Projet de Najib et Victor

Un super projet sur les espaces de configuration.

## Package `calculs/uconf`

Le dossier `calculs/uconf` contient des modèles combinatoires d'opérades utilisés
- `surjection.py` : opérade des surjections (`Surjection`).
- `barratt_eccles.py` : opérade de Barratt-Eccles (`BarrattEccles`).
- `lie.py` : composantes de l'opérade de Lie (`Lie`).
- `operad.py` : protocole de typage (`OperadProtocol`) pour homogénéiser l'API.
- `cooperad.py` : protocole de typage dual (`CooperadProtocol`) pour les constructions bar/cobar.
- `signs.py` : conventions de signes partagées (suspension/Koszul).
- `shifted_operad.py` : décalage opéradique (`ShiftedOperad`) avec signes de suspension.

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

### Opérade décalée (`ShiftedOperad`)

On peut construire une opérade décalée à partir d'une opérade existante
Exemple minimal :

```python
from uconf import Lie, ShiftedOperad

# Σ^d Lie
ShiftLie = ShiftedOperad(Lie, 1)

# Composantes d'arité fixée
L2 = ShiftLie(2)
x = L2((1,))
# Action symétrique tordue par sgn(σ)^d
xp = x.permute([2, 1])

# Composition partielle avec signe de décalage
z = ShiftLie.compose(x, 2, x)

Conventions implémentées (type Loday--Vallette) :

- degré : `|a|_shift = |a| + d (n - 1)` en arité `n` ;
- action de `S_n` : facteur `sgn(σ)^d` ;
- composition : facteur
	`(-1)^(d ((i-1)(n-1) + (m-1)|y|))` pour `x \circ_i y`, avec
	`m = arité(x)`, `n = arité(y)` et `|y|` degré dans l'opérade de base.

## Tests

Les tests principaux sont organisés par opérade :
- `test_common_operad.py` : tests communs (protocole opéradique).
- `test_barratt_eccles.py` : tests spécifiques à Barratt-Eccles.
- `test_surjection.py` : tests spécifiques aux surjections.
- `test_lie.py` : tests de l'opérade de Lie.
- `test_shifted_operad.py` : tests de l'opérade décalée.

Exécution recommandée depuis `calculs/` : `cd calculs && pytest`.

Pour exécuter seulement le trio commun/Barratt-Eccles/Surjection :
`cd calculs && pytest -q test_common_operad.py test_barratt_eccles.py test_surjection.py`.
# Projet de Najib et Victor

Un super projet sur les espaces de configuration.

## Package `calculs/uconf`

Le dossier `calculs/uconf` contient des modèles combinatoires d'opérades utilisés
dans les calculs.

- `surjection.py` : opérade des surjections (`Surjection`).
- `barratt_eccles.py` : opérade de Barratt-Eccles (`BarrattEccles`).
- `lie.py` : composantes de l'opérade de Lie (`Lie`).
- `operad.py` : protocole de typage (`OperadProtocol`) pour homogénéiser l'API.
- `cooperad.py` : protocole de typage dual (`CooperadProtocol`) pour les constructions bar/cobar.
- `signs.py` : conventions de signes partagées (suspension/Koszul).
- `shifted_operad.py` : décalage opéradique (`ShiftedOperad`) avec signes de suspension.

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

### Opérade décalée (`ShiftedOperad`)

On peut construire une opérade décalée à partir d'une opérade existante
(`Lie`, `Surjection`, etc.) et d'un entier `d`.

Exemple minimal :

```python
from uconf import Lie, ShiftedOperad

# Σ^d Lie
ShiftLie = ShiftedOperad(Lie, 1)

# Composantes d'arité fixée
L2 = ShiftLie(2)
x = L2((1,))

# Action symétrique tordue par sgn(σ)^d
xp = x.permute([2, 1])

# Composition partielle avec signe de décalage
z = ShiftLie.compose(x, 2, x)
```

Conventions implémentées (type Loday--Vallette) :

- degré : `|a|_shift = |a| + d (n - 1)` en arité `n` ;
- action de `S_n` : facteur `sgn(σ)^d` ;
- composition : facteur
	`(-1)^(d ((i-1)(n-1) + (m-1)|y|))` pour `x \circ_i y`, avec
	`m = arité(x)`, `n = arité(y)` et `|y|` degré dans l'opérade de base.

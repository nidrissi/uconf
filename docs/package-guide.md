# Package guide

The codebase is organised by mathematical role.

## Top-level package

- `uconf` re-exports the main public API for convenience.
- The top-level import also wires lazy conversions such as Barratt--Eccles table
  reduction and the surjection section map.

## `uconf.core`

Shared protocols and low-level utilities: operad/cooperad interfaces,
morphisms, twisting morphisms, trees, sign conventions, and display helpers.

## `uconf.models`

Concrete operad and cooperad models such as `Surjection`, `BarrattEccles`,
`Lie`, `Associative`, `Commutative`, and their simplicial companions.

## `uconf.constructions`

Bar and cobar functors, together with twisted bar/cobar algebraic
constructions built on top of the core models.

## `uconf.algebraic`

Algebra and coalgebra wrappers, free/cofree constructions, simplicial and
spherical models, and configuration-space helpers.

## `uconf.morphisms`

Classical operad morphisms, canonical twisting morphisms, and the
`E_ν`-comodule map machinery.

## `uconf.wrappers`

Shifted operads/cooperads and Hadamard-product constructions that adapt the
core models while preserving their SageMath-native behaviour.

## Standalone modules

- `uconf.homology` contains chain-complex and homology helpers.
- `uconf.sampling` contains construction-aware random sampling utilities used
  throughout the tests.

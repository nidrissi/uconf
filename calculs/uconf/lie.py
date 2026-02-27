"""Finite-arity model of the Lie operad in a Hall-type basis."""

import itertools as py_itertools
from typing import Any, ClassVar

from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]


class Lie(CombinatorialFreeModule):
    """Lie operad component in arity ``n``.

    Basis keys are permutations of ``(1, ..., n-1)`` and encode nested brackets
    ``[x_i, -]`` ending in ``x_n``.
    """

    name: ClassVar[str] = "Lie"

    def __init__(self, n, base_ring=QQ):
        """Initialize ``Lie(n)`` over ``base_ring``."""

        assert n >= 0, f"Arity must be non-negative. Got {n}."
        name = f"{self.name}{n}"
        super().__init__(
            base_ring,
            tuple,
            prefix=name,
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename(name)
        self._arity = n
        self._symmetric_group = SymmetricGroup(n)
        self.boundary = self.module_morphism(
            on_basis=lambda basis: self.zero(), codomain=self
        )

    def _basis_keys(self) -> list[tuple[int, ...]]:
        """Return and cache the list of basis keys in this arity."""

        if hasattr(self, "_basis_keys_cache"):
            return self._basis_keys_cache

        n = int(self.arity())
        if n == 0:
            self._basis_keys_cache: list[tuple[int, ...]] = []
        elif n == 1:
            self._basis_keys_cache = [()]
        else:
            self._basis_keys_cache = list(py_itertools.permutations(range(1, n), n - 1))
        return self._basis_keys_cache

    def _validate_basis_key(self, basis_key: tuple | list) -> tuple[int, ...] | None:
        """Validate and normalize a basis key.

        Returns ``None`` only in arity ``0`` where no nonzero basis elements
        exist.
        """

        if self.arity() == 0:
            return None

        if not isinstance(basis_key, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple/list, got {type(basis_key)}")

        clean = tuple(int(i) for i in basis_key)
        n = int(self.arity())

        if n == 1:
            if clean != ():
                raise ValueError(f"The only basis key in arity 1 is (). Got {clean}.")
            return clean

        if len(clean) != n - 1:
            raise ValueError(
                f"Basis key in arity {n} must have length {n - 1}. Got {len(clean)}."
            )

        if set(clean) != set(range(1, n)):
            raise ValueError(
                f"Basis key must be a permutation of 1..{n - 1}. Got {clean}."
            )

        return clean

    def _element_constructor_(self, x):
        """Build elements from basis keys or sparse dictionaries."""

        if isinstance(x, dict):
            clean_dict = {}
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if clean_key is None:
                    continue
                clean_dict[clean_key] = coeff
            return super()._element_constructor_(clean_dict)

        if isinstance(x, (tuple, list)):
            clean_key = self._validate_basis_key(x)
            if clean_key is None:
                return self.zero()
            return self.term(clean_key)

        raise TypeError(
            f"Input must be a dictionary (for linear combinations) or a tuple/list (for basis elements). Got {x} ({type(x)})."
        )

    def _bracket(
        self, left: dict[tuple[int, ...], Any], right: dict[tuple[int, ...], Any]
    ):
        """Compute commutator ``[left, right]`` on associative expansions."""

        out: dict[tuple[int, ...], Any] = {}
        for wl, cl in left.items():
            for wr, cr in right.items():
                c = cl * cr
                if c == 0:
                    continue

                wlr = wl + wr
                wrl = wr + wl
                out[wlr] = out.get(wlr, self.base_ring().zero()) + c
                out[wrl] = out.get(wrl, self.base_ring().zero()) - c

        return {w: c for w, c in out.items() if c != 0}

    def _assoc_from_basis_key(
        self, basis_key: tuple[int, ...]
    ) -> dict[tuple[int, ...], Any]:
        """Expand one Lie basis element into the associative word basis."""

        n = int(self.arity())
        if n == 0:
            return {}
        if n == 1:
            return {(1,): self.base_ring().one()}

        seq = basis_key + (n,)
        current: dict[tuple[int, ...], Any] = {(seq[-1],): self.base_ring().one()}
        for a in reversed(seq[:-1]):
            current = self._bracket({(a,): self.base_ring().one()}, current)
        return current

    def _word_basis(self) -> list[tuple[int, ...]]:
        """Return and cache the associative word basis in arity ``n``."""

        if hasattr(self, "_word_basis_cache"):
            return self._word_basis_cache

        n = int(self.arity())
        if n == 0:
            self._word_basis_cache: list[tuple[int, ...]] = []
        else:
            self._word_basis_cache = list(py_itertools.permutations(range(1, n + 1), n))
        return self._word_basis_cache

    def _pbw_matrix(self):
        """Return the PBW change-of-basis matrix from Lie to words."""

        if hasattr(self, "_pbw_matrix_cache"):
            return self._pbw_matrix_cache

        words = self._word_basis()
        keys = self._basis_keys()

        if self.arity() == 0:
            mat = matrix(self.base_ring(), 0, 0, [])
            self._pbw_matrix_cache = mat
            return mat

        if self.arity() == 1:
            mat = matrix(self.base_ring(), 1, 1, [1])
            self._pbw_matrix_cache = mat
            return mat

        data = []
        for w in words:
            for key in keys:
                data.append(
                    self._assoc_from_basis_key(key).get(w, self.base_ring().zero())
                )
        mat = matrix(self.base_ring(), len(words), len(keys), data)
        self._pbw_matrix_cache = mat
        return mat

    def _element_from_assoc(self, assoc: dict[tuple[int, ...], Any]) -> "Lie.Element":
        """Reconstruct a Lie element from associative coefficients."""

        if not assoc:
            return self.zero()

        if self.arity() == 0:
            return self.zero()

        words = self._word_basis()
        vec = vector(
            self.base_ring(), [assoc.get(w, self.base_ring().zero()) for w in words]
        )

        if self.arity() == 1:
            return vec[0] * self.term(())

        pbw = self._pbw_matrix()
        coords = pbw.solve_right(vec)
        keys = self._basis_keys()
        return self.sum_of_terms(
            (keys[i], coords[i]) for i in range(len(keys)) if coords[i] != 0
        )

    def arity(self) -> int:
        """Return the arity of this Lie operad component."""

        return self._arity

    @staticmethod
    def unit():
        """Return the operadic unit in arity ``1``."""

        return Lie(1)(())

    def degree_on_basis(self, element) -> int:
        """Return homological degree on basis elements (always ``0`` here)."""

        return 0

    def _repr_term(self, basis_element: tuple[int, ...]) -> str:
        """String representation of one basis element as nested brackets."""

        n = self.arity()
        if n == 0:
            return "0"
        if n == 1:
            return "x1"

        expr = f"x{n}"
        for i in reversed(basis_element):
            expr = f"[x{i},{expr}]"
        return expr

    def _latex_term(self, basis_element: tuple[int, ...]) -> str:
        """LaTeX representation of one basis element as nested brackets."""

        n = self.arity()
        if n == 0:
            return "0"
        if n == 1:
            return "x_{1}"

        def _var(k: int) -> str:
            return f"x_{{{k}}}"

        expr = _var(n)
        for i in reversed(basis_element):
            expr = f"\\left[{_var(i)}, {expr}\\right]"
        return expr

    @staticmethod
    def compose(x: "Lie.Element", i: int, y: "Lie.Element") -> "Lie.Element":
        """Operadic composition ``x \circ_i y`` in the Lie operad."""

        x_parent = x.parent()
        y_parent = y.parent()
        m = x_parent.arity()
        n = y_parent.arity()

        assert 1 <= i <= m, f"Index i must be between 1 and {m}. Got {i}."
        assert (
            x_parent.base_ring() == y_parent.base_ring()
        ), "Both elements must have the same base ring."

        target = Lie(m + n - 1, base_ring=x_parent.base_ring())
        result_assoc: dict[tuple[int, ...], Any] = {}

        for x_key, x_coeff in x:
            outer_assoc = x_parent._assoc_from_basis_key(x_key)
            for outer_word, outer_coeff in outer_assoc.items():
                pos = outer_word.index(i)
                left = outer_word[:pos]
                right = outer_word[pos + 1 :]

                shifted_left = tuple(j if j < i else j + n - 1 for j in left)
                shifted_right = tuple(j if j < i else j + n - 1 for j in right)

                for y_key, y_coeff in y:
                    inner_assoc = y_parent._assoc_from_basis_key(y_key)
                    for inner_word, inner_coeff in inner_assoc.items():
                        shifted_inner = tuple(j + i - 1 for j in inner_word)
                        new_word = shifted_left + shifted_inner + shifted_right
                        coeff = x_coeff * y_coeff * outer_coeff * inner_coeff
                        result_assoc[new_word] = (
                            result_assoc.get(new_word, target.base_ring().zero())
                            + coeff
                        )

        result_assoc = {w: c for w, c in result_assoc.items() if c != 0}
        return target._element_from_assoc(result_assoc)

    class Element(CombinatorialFreeModule.Element):
        """Elements of a fixed-arity Lie component."""

        def arity(self) -> int:
            """Return the element arity."""

            return self.parent().arity()

        def boundary(self) -> "Lie.Element":
            """Apply the differential (trivial in this model)."""

            return self.parent().boundary(self)

        def permute(self, sigma: Permutation | list | tuple) -> "Lie.Element":
            """Apply the right action of a permutation on generators."""

            if isinstance(sigma, (list, tuple)):
                sigma_perm: Permutation = self.parent()._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent")
                and sigma.parent() == self.parent()._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity()}. Got {sigma} ({type(sigma)})."
                )
            else:
                sigma_perm: Permutation = sigma

            assoc: dict[tuple[int, ...], Any] = {}
            for key, coeff in self:
                key_assoc = self.parent()._assoc_from_basis_key(key)
                for word, word_coeff in key_assoc.items():
                    permuted_word = tuple(sigma_perm(i) for i in word)
                    assoc[permuted_word] = (
                        assoc.get(permuted_word, self.base_ring().zero())
                        + coeff * word_coeff
                    )

            assoc = {w: c for w, c in assoc.items() if c != 0}
            return self.parent()._element_from_assoc(assoc)

r"""Rank-4 torus cochain models and explicit Surjection actions.

This module implements the reduced cochain complex of ``S^1 × S^1`` (the torus)
as a rank-4 module with basis elements [0], [α], [β], [γ] and equips it with
the explicit ``Surjection``-algebra structure.

**Basis elements:**
- [0] = [*] ⊗ [*]  (dimension 0)
- [α] = [a_1] ⊗ [*]  (dimension 1)
- [β] = [*] ⊗ [a_1]  (dimension 1)
- [γ] = [a_1] ⊗ [a_1]  (dimension 2)

**Algebra structure:**
The Surjection-algebra structure is determined by the following operations:

1. μ_σ([0]^⊗n) = [0] if k=0 (degree k-1)

2. μ_σ([α]^⊗a, [0]^⊗d) = (-1)^(a(a-1)/2) ψ_{I_A}(σ)[α] 
   with a+d=n, if k=a, where I_A is the subset of inputs equal to [α]

3. μ_σ([β]^⊗b, [0]^⊗d) = (-1)^(b(b-1)/2) ψ_{I_B}(σ)[β]
   with b+d=n, if k=b, where I_B is the subset of inputs equal to [β]

4. μ_σ([γ]^⊗c, [α]^⊗a, [β]^⊗b, [0]^⊗d) = (-1)^(...) ψ_{I_C∪A}(σ_1) ψ_{I_C∪B}(σ_2) [γ]
   with c+a+b+d=n, if k=2c+a+b-1

5. All other products are zero.

Reference: Article.tex, Section on "The E_∞-algebra structure of C*(S^1) ⊗ C*(S^1)"
"""

from __future__ import annotations

from math import prod

from sage.all import CombinatorialFreeModule, GradedModulesWithBasis, cached_method, Family

from uconf.algebraic.algebra import OperadAlgebra
from uconf.core.display import latex_linear_combination
from uconf.models.surjection import Surjection

#TODO: Rechecker si la formule ici correspond bien à celle du papier. De loin elle a l'air de correspondre, mais pas il y a plusieurs termes que je comprends pas, notamment pourquoi Claude a appelé la fonction _extract_concatenated_permutations dans son code. 

def _psi_function(u_tuple: list[tuple[int, ...]], index_subset: tuple[int, ...]) -> int:
    r"""Compute the ψ_{I_S} function for Barratt-Eccles/surjection action.

    Given an s-tuple of permutations (σ_0, ..., σ_{s-1}) and a subset I_S
    of {1, ..., n}, compute ψ_{I_S}(u) which is:
    - sign(permutation defined by first occurrences) if all distinct
    - 0 if any element appears twice
    """
    if not u_tuple or not index_subset:
        return 0

    s = len(u_tuple)
    first_occurrences = []

    for perm in u_tuple:
        found = None
        for pos, val in enumerate(perm):
            if val in index_subset:
                found = val
                break
        if found is None:
            return 0
        first_occurrences.append(found)

    if len(set(first_occurrences)) != len(first_occurrences):
        return 0

    sorted_subset = sorted(index_subset)
    mapping = {val: i + 1 for i, val in enumerate(sorted_subset)}
    permutation_image = tuple(mapping[val] for val in first_occurrences)

    return _permutation_sign(permutation_image)

def _extract_concatenated_permutations(
    u: tuple[int, ...], n: int,
) -> list[tuple[int, ...]] | None:
    r"""Return the permutation blocks if u has the sphere-admissible form.

    A surjection u of arity n and degree d*(n-1) is
    sphere-admissible if it can be written as the concatenation of
    (d+1) permutations σ_1,...,σ_{d+1} of {1,...,n} with
    the overlap condition σ_j(n) = σ_{j+1}(1) for j=1,...,d.
    """
    if n <= 0:
        return None
    if n == 1:
        if u == (1,):
            return [(1,)]
        return None

    expected_len = n + d * (n - 1)
    if len(u) != expected_len:
        return None

    perms: list[tuple[int, ...]] = []
    valid = set(range(1, n + 1))
    for i in range(0, len(u) - 1, n - 1):
        block = tuple(u[i : i + n])
        if set(block) != valid:
            return None
        perms.append(block)

    return perms

def _permutation_sign(perm: tuple[int, ...]) -> int:
    """Return the sign of a permutation."""
    inversions = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            if perm[i] > perm[j]:
                inversions += 1
    return -1 if inversions % 2 else 1


class ReducedTorusCochains(CombinatorialFreeModule):
    r"""Reduced cochains of S^1 × S^1 (torus) as a rank-4 graded module.

    Basis elements:
    - "0": dimension 0 (corresponds to [*] ⊗ [*])
    - "α": dimension 1 (corresponds to [a_1] ⊗ [*])
    - "β": dimension 1 (corresponds to [*] ⊗ [a_1])
    - "γ": dimension 2 (corresponds to [a_1] ⊗ [a_1])

    The boundary is zero (for cohomological grading).
    """

    def __init__(self, base_ring):
        generators = ["0", "α", "β", "γ"]
        super().__init__(
            base_ring,
            generators,
            prefix="N*T²",
            category=GradedModulesWithBasis(base_ring),
        )
        self.rename("N*T²")
        self._generators = {gen: self(gen) for gen in generators}
        self.connectivity = -2
        self.boundary = self.module_morphism(on_basis=lambda _: self.zero(), codomain=self)

    def degree_on_basis(self, element: str) -> int:
        """Return the homological degree of a basis element."""
        degrees = {"0": 0, "α": -1, "β": -1, "γ": -2}
        return degrees.get(element, 0)

    def _weight_on_basis(self, _) -> int:
        return 0

    def generator(self, name: str):
        """Return the generator by name."""
        return self._generators.get(name)

    def basis_iter(self, d: int):
        """Iterate over basis elements of a given degree."""
        degree_map = {"0": 0, "α": -1, "β": -1, "γ": -2}
        for gen, deg in degree_map.items():
            if deg == d:
                yield self(gen)

    @cached_method
    def graded_basis(self, d: int):
        return Family(self.basis_iter(d))

    def basis_weight_iter(self, d: int, w: int):
        if w == 0:
            yield from self.basis_iter(d)

    @cached_method
    def graded_weighted_basis(self, d: int, w: int):
        return Family(self.basis_weight_iter(d, w))

    def _repr_term(self, element: str) -> str:
        return element

    def _latex_term(self, element: str) -> str:
        latex_map = {"0": "[0]", "α": "[\\alpha]", "β": "[\\beta]", "γ": "[\\gamma]"}
        return latex_map.get(element, element)

    class Element(CombinatorialFreeModule.Element):
        def _repr_latex_(self) -> str:
            return latex_linear_combination(self, lambda basis: self.parent()._latex_term(basis))


class SurjectionTorusCochainAlgebra(OperadAlgebra):
    r"""Explicit Surjection-algebra structure on reduced cochains of S^1 × S^1.

    Implements the Barratt-Eccles/Surjection algebra structure on the torus
    cochains according to the formulas in article.tex.

    The structure map decomposes the input based on how many copies of each
    basis element ([α], [β], [γ]) appear and applies the corresponding formula.
    """

    def __init__(self, base_ring):
        module = ReducedTorusCochains(base_ring=base_ring)
        super().__init__(
            module=module,
            operad_cls=Surjection,
            structure_map=self._act_impl,
        )

    def _extract_input_composition(self, algebra_elements):
        """Extract the composition (counts and values) of input elements."""
        element_dict = {"0": 0, "α": 0, "β": 0, "γ": 0}
        total_degree = 0

        for elem in algebra_elements:
            coeff_sum = 0
            for basis, scalar in elem:
                degree = self.module.degree_on_basis(basis)
                coeff_sum += scalar
                total_degree += degree
                element_dict[basis] += scalar

        return element_dict, total_degree

    def _compute_sign_factor(self, num_a: int, num_b: int, num_c: int) -> int:
        r"""Compute the sign factor from the formula.

        For case with [γ]^⊗c, [α]^⊗a, [β]^⊗b:
        sign = (-1)^(c(c-1)/2 + a(a-1)/2 + b(b-1)/2 + ab + c + b - 1)
        """
        exp = (c := num_c) * (c - 1) // 2
        exp += (a := num_a) * (a - 1) // 2
        exp += (b := num_b) * (b - 1) // 2
        exp += a * b + c + b - 1
        return -1 if exp % 2 else 1

    def _act_impl(self, p_element: Surjection.Element, algebra_elements):
        r"""Implement the Surjection algebra action on the torus cochains."""
        if p_element.arity() == 0:
            return self.module.zero()

        element_dict, _ = self._extract_input_composition(algebra_elements)
        num_alpha = int(element_dict["α"])
        num_beta = int(element_dict["β"])
        num_gamma = int(element_dict["γ"])
        num_zero = int(element_dict["0"])
        n = p_element.arity()

        if num_alpha + num_beta + num_gamma + num_zero != n:
            return self.module.zero()

        input_scalar = 1
        for elem in algebra_elements:
            for basis, scalar in elem:
                if basis != "0":
                    input_scalar *= scalar

        if input_scalar == 0:
            return self.module.zero()

        k = p_element.degree()

        # Case 1: All inputs are [0]
        if num_alpha == 0 and num_beta == 0 and num_gamma == 0:
            if k == 0:
                return input_scalar * self.module.generator("0")
            else:
                return self.module.zero()

        # Case 2: Only [α]^⊗a and [0]^⊗d
        if num_beta == 0 and num_gamma == 0 and num_alpha > 0:
            if k != num_alpha:
                return self.module.zero()

            indices_A = tuple(i + 1 for i, elem in enumerate(algebra_elements) if any(b == "α" for b, _ in elem))

            coeff = 0
            for basis_u, p_coeff in p_element:
                perms = _extract_concatenated_permutations(basis_u, n, num_alpha - 1)
                if perms is not None:
                    psi_val = _psi_function(perms, indices_A)
                    if psi_val != 0:
                        sign_factor = (-1) ** (num_alpha * (num_alpha - 1) // 2)
                        coeff += p_coeff * psi_val * sign_factor

            if coeff == 0:
                return self.module.zero()
            return input_scalar * coeff * self.module.generator("α")

        # Case 3: Only [β]^⊗b and [0]^⊗d
        if num_alpha == 0 and num_gamma == 0 and num_beta > 0:
            if k != num_beta:
                return self.module.zero()

            indices_B = tuple(i + 1 for i, elem in enumerate(algebra_elements) if any(b == "β" for b, _ in elem))

            coeff = 0
            for basis_u, p_coeff in p_element:
                perms = _extract_concatenated_permutations(basis_u, n, num_beta - 1)
                if perms is not None:
                    psi_val = _psi_function(perms, indices_B)
                    if psi_val != 0:
                        sign_factor = (-1) ** (num_beta * (num_beta - 1) // 2)
                        coeff += p_coeff * psi_val * sign_factor

            if coeff == 0:
                return self.module.zero()
            return input_scalar * coeff * self.module.generator("β")

        # Case 4: Mixed case with [γ], [α], [β]
        if num_gamma > 0:
            expected_k = 2 * num_gamma + num_alpha + num_beta - 1
            if k != expected_k:
                return self.module.zero()

            indices_CA = tuple(
                i + 1
                for i, elem in enumerate(algebra_elements)
                if any(b in ("γ", "α") for b, _ in elem)
            )
            indices_CB = tuple(
                i + 1
                for i, elem in enumerate(algebra_elements)
                if any(b in ("γ", "β") for b, _ in elem)
            )

            coeff = 0
            for basis_u, p_coeff in p_element:
                perms_1 = _extract_concatenated_permutations(basis_u, n, num_gamma + num_alpha - 1)
                perms_2 = _extract_concatenated_permutations(basis_u, n, num_gamma + num_beta - 1)

                if perms_1 is not None and perms_2 is not None:
                    psi_1 = _psi_function(perms_1, indices_CA)
                    psi_2 = _psi_function(perms_2, indices_CB)

                    if psi_1 != 0 and psi_2 != 0:
                        sign_factor = self._compute_sign_factor(num_alpha, num_beta, num_gamma)
                        coeff += p_coeff * psi_1 * psi_2 * sign_factor

            if coeff == 0:
                return self.module.zero()
            return input_scalar * coeff * self.module.generator("γ")

        return self.module.zero()

from itertools import combinations, pairwise
from sage.all import *  # pyright: ignore[reportWildcardImportFromLibrary]


class BarrattEccles(CombinatorialFreeModule):
    def __init__(self, n, base_ring=QQ):
        super().__init__(
            base_ring,
            tuple,
            prefix=f"BE{n}",
            category=GradedModulesWithBasis(base_ring),
        )
        self.arity = n
        self._symmetric_group = SymmetricGroup(n)
        self.boundary = self.module_morphism(
            on_basis=self._boundary_on_basis, codomain=self
        )

    def _element_constructor_(self, x: "BarrattEccles.Element | dict | tuple | list"):
        """
        Intercepts element creation to enforce types.
        x can be:
          - A basis key (tuple of permutations)
          - A linear combination (dictionary)
        """
        # Case 1: x is a Dictionary (Linear Combination)
        if isinstance(x, dict):
            # Validate keys before passing to super
            clean_dict = {}
            for key, coeff in x.items():
                clean_key = self._validate_basis_key(key)
                if self._has_consecutive_identical(clean_key):
                    # Skip zero terms
                    continue
                clean_dict[clean_key] = coeff
            return super()._element_constructor_(clean_dict)

        # Case 2: x is a Basis Key (Tuple)
        # We try to treat it as a single basis element
        if isinstance(x, (tuple, list)):
            # Check if it looks like a tuple of permutations vs a list of terms
            # Simple heuristic: is the first element a Permutation or convertable to one?
            try:
                clean_key = self._validate_basis_key(x)
                # Check for consecutive identical permutations
                if self._has_consecutive_identical(clean_key):
                    return self.zero()
                else:
                    # Return the monomial 1 * basis_element
                    return self.term(clean_key)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Item is not a valid element of S_{self.arity}. Got {x} ({type(x)})"
                ) from e

    def _validate_basis_key(self, basis_tuple: "tuple | list") -> tuple:
        """
        Strictly checks that the input is a tuple of S_n elements.
        """
        if not isinstance(basis_tuple, (tuple, list)):
            raise TypeError(f"Basis key must be a tuple, got {type(basis_tuple)}")

        clean_tuple = []
        for i, p in enumerate(basis_tuple):
            # 1. Check if it is already a Sage Permutation
            if hasattr(p, "parent") and p.parent() == self._symmetric_group:
                clean_tuple.append(p)
            else:
                # 2. Try to coerce it (e.g. from list [1, 2, 3])
                try:
                    p_converted = self._symmetric_group(p)
                    clean_tuple.append(p_converted)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Item {i} in basis tuple is not a valid element of S_{self.arity}. "
                        f"Got {p} ({type(p)})."
                    ) from e

        return tuple(clean_tuple)

    def _has_consecutive_identical(self, basis_tuple: tuple) -> bool:
        """
        Checks if the basis tuple has two consecutive identical permutations.
        """
        if len(basis_tuple) <= 1:
            return False
        for i in range(len(basis_tuple) - 1):
            if basis_tuple[i] == basis_tuple[i + 1]:
                return True
        return False

    def _boundary_on_basis(self, basis_element: tuple) -> "BarrattEccles.Element":
        """Standard simplicial boundary."""
        res = self.zero()
        # Boundary logic (same as before)
        if len(basis_element) <= 1:
            return res

        for i in range(len(basis_element)):
            face = basis_element[:i] + basis_element[i + 1 :]
            res += (-1) ** i * self.term(face)
        return res

    # 2. Implement the hook the Category expects
    def degree_on_basis(self, element: tuple) -> int:
        return len(element) - 1

    @staticmethod
    def compose(x: BarrattEccles, i: int, y: BarrattEccles) -> BarrattEccles:
        n = x.parent().arity
        m = y.parent().arity
        target = BarrattEccles(n + m - 1)

        # --- Helper: Composition of single permutations ---
        def _compose_perm_tuple(sigma, idx, tau):
            """
            Composes tuple sigma with tuple tau at index idx.
            Replaces the value 'idx' in sigma with the shifted sequence tau.
            """
            shift = tau.parent().degree() - 1
            res = []
            for val in sigma.tuple():
                if val < idx:
                    res.append(val)
                elif val > idx:
                    res.append(val + shift)
                else:  # val == idx
                    # Insert tau shifted by idx - 1
                    res.extend([t + idx - 1 for t in tau.tuple()])
            return tuple(res)

        # --- Helper: Eilenberg-Zilber Logic ---
        def term_generator():
            # Iterate over the linear combinations
            for x_basis, x_coeff in x:
                for y_basis, y_coeff in y:

                    # Degrees (length of tuple - 1)
                    p = len(x_basis) - 1
                    q = len(y_basis) - 1

                    # Generate Shuffles of the steps
                    # 0 represents a step in x, 1 represents a step in y
                    multiset = [0] * p + [1] * q
                    shuffles = Permutations(multiset).list()

                    for sh in shuffles:
                        # 1. Calculate the Sign of the shuffle
                        # Formula: (-1)^(sum of 1-based indices of y-steps) - q(q+1)/2
                        # This is equivalent to the sign of the permutation un-shuffling the lists
                        y_indices_sum = sum(
                            k + 1 for k, step in enumerate(sh) if step == 1
                        )
                        sign_exponent = y_indices_sum - (q * (q + 1) // 2)
                        sign = (-1) ** sign_exponent

                        # 2. Build the new basis element (Path lifting)
                        # We start at (x0, y0) and walk the path defined by the shuffle
                        path = []

                        # Current indices in the basis tuples
                        ix, iy = 0, 0

                        # The first vertex is always (x[0] o_i y[0])
                        start_perm = _compose_perm_tuple(x_basis[0], i, y_basis[0])
                        path.append(start_perm)

                        # Walk the steps
                        for step in sh:
                            if step == 0:
                                ix += 1  # Step in X
                            else:
                                iy += 1  # Step in Y

                            # Compose the permutations at the current grid point
                            comp = _compose_perm_tuple(x_basis[ix], i, y_basis[iy])
                            path.append(comp)

                        # Yield the constructed basis tuple and the combined coefficient
                        yield (tuple(path), x_coeff * y_coeff * sign)

        return target.sum_of_terms(term_generator())

    def _complexity_on_basis(self, element: tuple) -> int:
        """
        The complexity of a finite binary sequence of elements in
        :math:`\\Sigma_2` is defined as the number of consecutive distinct
        elements in it. For example, :math:`((12),(21),(21),(12))` and
        :math:`((12),(12),(12),(21))` have complexities 2 and 1 respectively.
        For any basis Barratt-Eccles element, and any pair of positive integers
        :math:`i < j` less than its arity, we can form a sequence as above by
        precomposing each permutation by the order-preserving inclusion sending
        :math:`1` and :math:`2` respectively to :math:`i` and :math:`j`. The
        complexity of a basis Barratt-Eccles element is defined as the maximum
        over $i < j$ of the complexities of these. Notice that for arity 2,
        the complexity of an element is equal to its degree plus 1. It is proven in
        [BF] that the subcomplex generated by basis Barratt-Eccles elements of
        complexity at most :math:`n` define a suboperad of :math:`\\mathcal E`
        modeling an :math:`E_{n+1}`-operad.
        """
        result = 0
        n = self.arity

        for i, j in combinations(range(1, n + 1), 2):
            seq = list(
                filter(
                    lambda x: x == i or x == j,
                    (val for perm in element for val in perm.tuple()),
                )
            )
            complexity = len([p for p, q in pairwise(seq) if p != q])
            result = max(result, complexity)
        return result

    class Element(CombinatorialFreeModule.Element):
        def boundary(self):
            return self.parent().boundary(self)

        def arity(self):
            return self.parent().arity

        def complexity(self) -> int:
            return max(
                (self.parent()._complexity_on_basis(basis) for basis in self.support()),
                default=0,
            )

        def permute(self, sigma) -> "BarrattEccles.Element":
            """
            Permutes the basis elements of self by precomposing with sigma.
            """
            if isinstance(sigma, (list, tuple)):
                sigma = self.parent()._symmetric_group(sigma)
            elif not (
                hasattr(sigma, "parent")
                and sigma.parent() == self.parent()._symmetric_group
            ):
                raise TypeError(
                    f"Permutation must be a list, tuple, or element of S_{self.parent().arity}. Got {sigma} ({type(sigma)})."
                )

            def permuted_term_generator():
                for basis, coeff in self:
                    # Precompose each permutation in the basis tuple with sigma
                    permuted_basis = tuple(sigma * p for p in basis)
                    yield (permuted_basis, coeff)

            return self.parent().sum_of_terms(permuted_term_generator())

        def diagonal(self):
            """
            Computes the Alexander-Whitney diagonal: E -> E (x) E.
            """
            # 1. Construct the Tensor Product Parent
            # Sage handles this automatically: self.tensor(self) creates the module E (x) E

            tensor_module = self.parent().tensor(self.parent())

            result = tensor_module.zero()

            # 2. Iterate linearly
            for basis_tuple, coeff in self:
                k = len(basis_tuple)  # Degree

                # 3. Apply formula: Sum (front_face) (x) (back_face)
                for i in range(k + 1):
                    # Left side: (sigma_0 ... sigma_i)
                    left_basis = basis_tuple[: i + 1]

                    # Right side: (sigma_i ... sigma_k)
                    right_basis = basis_tuple[i:]

                    # Create the tensor element
                    # In Sage, tensor basis is ((left_key, right_key))
                    tensor_term = tensor_module.term((left_basis, right_basis))

                    result += coeff * tensor_term

            return result

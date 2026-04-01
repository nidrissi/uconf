from sage.all import tensor


def _construct_possible_tensor(module, key):
    """
        If the module has a tensor_factors method, use it to construct the possible tensor product element corresponding to the given key. Otherwise, just call
    the module on the key.

        Necessary because of the bug in SageMath's tensor product implementation, which does not allow for constructing elements by directly calling the parent on a tuple of keys. See https://github.com/sagemath/sage/issues/41882

        Args:
            module: The module in which to construct the element.
            key: The key corresponding to the element to construct. Should be a tuple of keys if the module has a tensor_factors method, and a single key otherwise.
        Returns:
            The element of the module corresponding to the given key.
    """
    if hasattr(module, "tensor_factors"):
        factors = module.tensor_factors()
        return tensor(f(x) for f, x in zip(factors, key, strict=True))
    else:
        return module(key)

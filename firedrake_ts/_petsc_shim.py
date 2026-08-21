"""ctypes access to the ``TSAdapt`` entry points petsc4py does not expose."""

import ctypes
import glob
import os

import numpy
from firedrake.petsc import PETSc

# PetscInt and PetscReal, derived from the running PETSc
_PetscInt = numpy.ctypeslib.as_ctypes_type(numpy.dtype(PETSc.IntType))
_PetscReal = numpy.ctypeslib.as_ctypes_type(numpy.dtype(PETSc.RealType))
# PetscBool is a C enum, i.e. int, independent of --with-64-bit-indices.
_PetscBool = ctypes.c_int

__all__ = [
    "ts_adapt_candidate_add",
    "ts_adapt_candidates_clear",
    "ts_adapt_choose",
    "ts_get_adapt",
]

_lib = None


def _dlopen_petsc():
    """A ctypes handle whose symbol table includes libpetsc's TS symbols.

    petsc4py loads libpetsc ``RTLD_LOCAL``, so ``CDLL(None)`` cannot see it.
    Prefer the versioned libpetsc under ``PETSC_DIR``/``PETSC_ARCH``; fall
    back to re-``dlopen``-ing petsc4py's own extension module ``RTLD_GLOBAL``,
    which promotes its already-loaded dependency into the global namespace.
    """
    import petsc4py

    cfg = petsc4py.get_config()
    pattern = os.path.join(cfg["PETSC_DIR"], cfg["PETSC_ARCH"], "lib", "libpetsc.so*")
    candidates = [*sorted(glob.glob(pattern)), PETSc.__file__]
    for path in candidates:
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        if hasattr(lib, "TSGetAdapt"):
            return lib
    raise RuntimeError(
        "could not resolve libpetsc's TS symbols; tried " + ", ".join(candidates)
    )


def _load():
    global _lib
    if _lib is not None:
        return _lib
    lib = _dlopen_petsc()
    lib.TSGetAdapt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.TSGetAdapt.restype = ctypes.c_int
    lib.TSAdaptCandidatesClear.argtypes = [ctypes.c_void_p]
    lib.TSAdaptCandidatesClear.restype = ctypes.c_int
    lib.TSAdaptCandidateAdd.argtypes = [
        ctypes.c_void_p,  # TSAdapt
        ctypes.c_char_p,  # name
        _PetscInt,  # order
        _PetscInt,  # stageorder
        _PetscReal,  # ccfl
        _PetscReal,  # cost
        _PetscBool,  # inuse
    ]
    lib.TSAdaptCandidateAdd.restype = ctypes.c_int
    lib.TSAdaptChoose.argtypes = [
        ctypes.c_void_p,  # TSAdapt
        ctypes.c_void_p,  # TS
        _PetscReal,  # h
        ctypes.POINTER(_PetscInt),  # next_sc
        ctypes.POINTER(_PetscReal),  # next_h
        ctypes.POINTER(_PetscBool),  # accept
    ]
    lib.TSAdaptChoose.restype = ctypes.c_int
    _lib = lib
    return _lib


def _check(ierr, what):
    if ierr:
        raise RuntimeError(f"{what} failed with ierr={ierr}")


def ts_get_adapt(ts):
    """The ``TSAdapt`` owned by ``ts``, as an opaque ctypes handle."""
    adapt = ctypes.c_void_p()
    _check(_load().TSGetAdapt(ts.handle, ctypes.byref(adapt)), "TSGetAdapt")
    return adapt


def ts_adapt_candidates_clear(adapt):
    _check(_load().TSAdaptCandidatesClear(adapt), "TSAdaptCandidatesClear")


def ts_adapt_candidate_add(adapt, order, stage_order, ccfl, cost, inuse):
    """Register one candidate scheme with the adapt controller."""
    _check(
        _load().TSAdaptCandidateAdd(
            adapt,
            None,
            int(order),
            int(stage_order),
            float(ccfl),
            float(cost),
            1 if inuse else 0,
        ),
        "TSAdaptCandidateAdd",
    )


def ts_adapt_choose(adapt, ts, h, last_accepted=True):
    """Returns ``(next_h, accept)``.

    The ``next_sc`` candidate index PETSc also writes is not returned: only
    one candidate is ever registered, so it is always 0, and the sole caller
    discarded it.

    ``accept`` is an IN/OUT argument, not pure output. ``TSAdaptChoose_Basic``
    reads it before writing it::

        if (!*accept) safety *= adapt->reject_safety;
            /* The last attempt also failed, shorten more aggressively */
    """
    next_sc = _PetscInt()
    next_h = _PetscReal()
    accept = _PetscBool(1 if last_accepted else 0)
    _check(
        _load().TSAdaptChoose(
            adapt,
            ts.handle,
            float(h),
            ctypes.byref(next_sc),
            ctypes.byref(next_h),
            ctypes.byref(accept),
        ),
        "TSAdaptChoose",
    )
    return next_h.value, bool(accept.value)

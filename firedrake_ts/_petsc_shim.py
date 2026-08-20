"""ctypes access to the ``TSAdapt`` entry points petsc4py does not expose.

petsc4py 3.25 binds neither ``TSAdapt`` as a type nor ``TSGetAdapt``,
``TSAdaptCandidatesClear``, ``TSAdaptCandidateAdd`` or ``TSAdaptChoose``. A
``TSPYTHON`` stepper that owns its own ``step`` must call them itself to get
``-ts_adapt_type basic`` working, so the controller and its options
(``-ts_adapt_rtol``, ``-ts_adapt_clip``, ...) stay PETSc's rather than being
reimplemented in Python.

``_dlopen_petsc`` is recovered from commit 3766c20.
"""

import ctypes
import glob
import os

import numpy
from firedrake.petsc import PETSc

# PetscInt and PetscReal, derived from the running PETSc rather than assumed.
# Hardcoding c_int/c_double is correct only for a 32-bit-index double build:
# under --with-64-bit-indices, TSAdaptChoose writes an 8-byte PetscInt through
# a 4-byte buffer (silent stack corruption), and under --with-precision=single
# every real is mis-marshalled. This module exists to get the ABI right, so it
# should not be guessing the scalar widths.
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
    # SIX arguments, not nine. Verified against the installed headers:
    #   petscts.h:1150
    #   TSAdaptChoose(TSAdapt, TS, PetscReal, PetscInt*, PetscReal*, PetscBool*)
    # An earlier draft of this shim declared trailing wlte/wltea/wlter output
    # pointers. They do not exist: the real function never writes them, so
    # they read back as ctypes' zero-initialised default and look like a
    # converged error estimate. That did not crash only because the x86-64
    # SysV ABI ignores unread trailing arguments -- an accident, not a
    # guarantee.
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
    """Register one candidate scheme with the adapt controller.

    Deliberately takes no ``name``, and passes ``NULL``.
    ``TSAdaptCandidateAdd`` stores the ``name`` POINTER without copying the
    bytes (``adapt->candidates.name[c] = name``,
    ``src/ts/interface/tsadapt.c:848``) and dereferences it later, when
    ``-ts_adapt_monitor`` prints the chosen candidate (``:1001``). Any
    Python-owned string would be freed as soon as this call returned, so a
    ``name`` parameter could only ever be a use-after-free waiting for
    someone to enable that monitor.

    PETSc's own ``TSStep_ARKIMEX`` passes ``tab->name``
    (``arkimex.c:1514``) safely only because that is a static string with
    program lifetime; ``ARKTableau.name`` is not, so mirroring that line --
    the obvious tidy-up, and what makes the monitor output prettier -- would
    break it. A caller owning a genuinely long-lived buffer can add the
    parameter back at that point; offering it now only invites the bug.
    """
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

    (``src/ts/adapt/impls/basic/adaptbasic.c:42``.) So the caller must say
    whether the PREVIOUS attempt at this step was accepted.
    ``TSStep_ARKIMEX`` declares ``accept = PETSC_TRUE`` once at step entry
    (``arkimex.c:1343``) and sets it ``PETSC_FALSE`` at its ``reject_step``
    label (``:1529``), which every rejection path reaches -- so the first
    call in a step sees true and each retry sees false.

    Passing a fresh zero-initialised ``c_int`` every call, as this did, makes
    PETSc believe the previous attempt always failed, so ``reject_safety``
    (default 0.5) is applied on the FIRST rejection as well as later ones,
    shrinking h twice as much as requested. ``last_accepted`` defaults to
    ``True`` so the common single-attempt case needs nothing from the caller.
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

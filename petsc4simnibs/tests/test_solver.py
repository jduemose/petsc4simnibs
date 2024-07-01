import numpy as np
import scipy.sparse
import pytest

from petsc4simnibs import PetscSolver

cg_ilu = " ".join(["-ksp_type cg", "-ksp_rtol 1e-10",  "-pc_type ilu"])

cg_hypre = " ".join(
    [
        "-ksp_type cg",
        "-ksp_rtol 1e-10",
        "-pc_type hypre",
        "-pc_hypre_type boomeramg ",
        "-pc_hypre_boomeramg_coarsen_type HMIS"
    ]
)

class TestSolve:
    @pytest.mark.parametrize('options', [cg_ilu, cg_hypre])
    def test_solve_petsc_diagonal(self, options):
        n = 1000
        A = scipy.sparse.diags(2 * np.ones(n)).tocsr()
        b = np.ones(n)
        solver = PetscSolver(options, A)
        x = solver.solve(b).squeeze()
        assert np.allclose(A.dot(x), b)

    @pytest.mark.parametrize('options', [cg_ilu, cg_hypre])
    def test_solve_petsc_random(self, options):
        np.random.seed(0)
        n = 1000
        A = np.random.random((5, 5))
        A += A.T
        A = scipy.sparse.csr_matrix(A)
        b = np.ones(n)
        solver = PetscSolver(options, A)
        x = solver.solve(b).squeeze()
        assert np.allclose(A.dot(x), b)

    @pytest.mark.parametrize('options', [cg_ilu, cg_hypre])
    def test_multiple_rhs(self, options):
        np.random.seed(0)
        n = 1000
        A = np.random.random((n, n))
        A += A.T
        A = scipy.sparse.csr_matrix(A)
        b = np.random.random((n, 3))
        #b = np.ones((n, 3))
        solver = PetscSolver(options, A)
        x = solver.solve(b).squeeze()
        assert np.allclose(A.dot(x), b)

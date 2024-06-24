petsc4simnibs_dir = 'petsc4simnibs'

python_sources = [
    '__init__.py',
]
python.install_sources(python_sources, subdir: petsc4simnibs_dir)

# mpi is needed as dep when build separately from petsc because petsc.pc does
# not include mpi headers in this case
mpi_dep = dependency('mpich')
petsc_dep = dependency('petsc')

python.extension_module(
    'solver',
    cython_gen_cpp.process('solver.pyx'),
    cpp_args: [cython_cpp_args],
    dependencies: [numpy_dep, mpi_dep, petsc_dep],
    install: true,
    subdir: petsc4simnibs_dir,
)
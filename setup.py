from distutils.core import setup

from Cython.Build import cythonize

# setup(ext_modules=cythonize('ilqr_server2.pyx'))

setup(
    ext_modules=cythonize(["mosaic/hyperrectangle.pyx",
                           "mosaic/interval.pyx",
                           "mosaic/point.pyx",
                            "symbolic/unroll_methods.pyx",
                           "prism/shared_rtree.pyx"], language_level="3",annotate=True)
)

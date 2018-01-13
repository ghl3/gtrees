
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

import numpy

include_dirs = [numpy.get_include()]
extra_compile_args = ["-O3"]
define_macros=[('CYTHON_TRACE', '1')]



extensions = [
    #Extension("tree._quad_tree", ["tree/_quad_tree.pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args, define_macros=define_macros),
    #Extension("tree._criterion", ["tree/_criterion.pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args, define_macros=define_macros),
    #Extension("tree._splitter", ["tree/_splitter.pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args, define_macros=define_macros),
    #Extension("tree._utils", ["tree/_utils.pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args, define_macros=define_macros),
    #Extension("tree._tree", ["tree/_tree.pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args, define_macros=define_macros),
    Extension("tree._my_tree", ["tree/_my_tree.pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args, define_macros=define_macros),
    ]


setup(
    name = "GTrees",
    ext_modules = cythonize(extensions),
)

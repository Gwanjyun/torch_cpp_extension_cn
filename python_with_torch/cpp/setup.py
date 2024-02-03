from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ext_modules=[
        CppExtension('gwanjyun_lltm_cpp', ['lltm.cpp']), # package name, package source file
    ] # from gwanjyun_lltm_cpp import xxx

cmdclass={
    'build_ext': BuildExtension
}

setup(
    name='gwanjyun_lltm_cpp',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
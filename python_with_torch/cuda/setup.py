from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules=[
    CUDAExtension(
        'gwanjyun_lltm_cuda', # package name, package source file
        [
            'lltm_cuda.cpp',
            'lltm_cuda_kernel.cu' # 名字不要与cpp相同
        ],
        extra_compile_args = {
            'cxx': ['-O2'],
            'nvcc': ['-O2']
        }
        ), 
] # from gwanjyun_lltm_cpp import xxx


cmdclass={
    'build_ext': BuildExtension
}

include_dirs = ['.']


setup(
    name='gwanjyun_lltm_cuda',
    version='1.0',
    author='Gwanjyun',
    author_email='gwanjyun@gmail.com',
    description='lltm cuda',
    long_description='lltm cuda',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    include_dirs=include_dirs,
)
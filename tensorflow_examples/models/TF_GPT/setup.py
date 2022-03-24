from setuptools import setup

setup(
    name='TF2_GPT-2',
    version='v_1.0',
    packages=['utils', 'layers'],
    url='https://github.com/Xhs753/TF2_GPT-2',
    license='MIT',
    author='watermelon',
    
    description='One NLP AI', install_requires=['click', 'tqdm', 'tensorflow', 'numpy', 'ftfy', 'sentencepiece']
)

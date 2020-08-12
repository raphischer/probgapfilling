from setuptools import setup

setup(
    name='probgf',
    version='0.0.1',
    author='Raphael Fischer',
    packages=['probgf', 'tests'],
    scripts=['scripts/example.py', 'scripts/gap_filling_viewer.py'],
    include_package_data=True,
    license='LICENSE.md',
    description='Spatio-temporal gap filling with machine learning.',
    long_description=open('README.md').read(),
    install_requires=[
		"matplotlib >= 3.2",
		"networkx >= 2.4",
        "numpy >= 1.19",
		"pillow >= 7.0",
		"pxpy == 1.0a20",
		"scikit-learn >= 0.23",
        "scikit-image >= 0.16",
        "scipy >= 1.5",
   ],
)

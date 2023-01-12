from setuptools import setup

setup(
   name='comblearn',
   version='0.1.0',
   author='Seyed Mohammad Hosseini',
   author_email='seymohhosseini@outlook.com',
   packages=['comblearn', 'comblearn.test'],
   license='LICENSE.txt',
   description='Combinatorial Auction learning',
   long_description=open('README.md').read(),
   install_requires=[
        "numpy",
        "torch>=1.12.1",
        "pytest",
   ],
)
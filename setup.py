from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will retun list of requirements from file
    '''
    requirements = []
    with open(file_path, 'r') as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='MLProject',
    version='0.0.1',
    author='Ankit',
    author_email='itforankit@gmail.com',
    packages=find_packages(),
   # install_requires=['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn']
   install_requires=get_requirements('requirements.txt')
)
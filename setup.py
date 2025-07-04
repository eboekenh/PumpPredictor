# This script configures the packaging and installation of the Pump It Up machine learning project.
# It defines metadata, dependencies, and package discovery.

from setuptools import find_packages,setup
from typing import List

# Constant for identifying editable install option in requirements
HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''
    this function will return the list of the requirements. Reads the list of dependencies from the given requirements file.
    Removes editable install flag ('-e .') if present.
    
    Args:
        file_path (str): Path to the requirements.txt file
    
    Returns:
        List[str]: Cleaned list of package names
    '''

    requirements = []
    with open(file_path) as file_obj:
        # Strip newline characters
        requirements = file_obj.readlines()
        requirements= [req.replace("\n","") for req in requirements]

        # Remove editable install option if it's in the list
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


# Setup function defines how this project should be installed as a package
setup(
    name='pumpitup', # Project/package name
    version='0.0.1',  # Initial version
    author= 'Ecem',
    author_email = 'ecemboekenheide@gmail.com', # Contact email
    packages =find_packages(), # Automatically discover all packages in the project
    install_requires = get_requirements('requirements.txt')
    
    )
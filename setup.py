from setuptools import find_packages, setup
from typing import List


HYPHEN_E_DOT= "-e ."
def get_requirements(path: str) -> List[str]:
    with open(path, "r") as file:
        requirements = file.readlines()
        requirements = [req.replace("/n", "") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="vision-transformer",
    version="0.0.1",
    author="Salah",
    author_email="salaheddineaitdaoud@gmail.com",
    package=find_packages(),
    install_requires=get_requirements("requirements.txt")

)
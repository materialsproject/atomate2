from setuptools import find_packages, setup

from atomate2 import __version__

with open("README.md", "r") as file:
    long_description = file.read()

if __name__ == "__main__":
    setup(
        name="atomate2",
        version=__version__,
        description="atomate has implementations of FireWorks workflows for "
        "Materials Science",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/hackingmaterials/amset",
        author="Alex Ganose, Anubhav Jain",
        author_email="aganose@lbl.gov, anubhavster@gmail.com",
        license="modified BSD",
        keywords="high-throughput automated workflow dft vasp",
        packages=find_packages(),
        # package_data={
        #     "amset": [
        #         "defaults.yaml",
        #         "plot/amset_base.mplstyle",
        #         "plot/revtex.mplstyle",
        #     ]
        # },
        data_files=["LICENSE"],
        zip_safe=False,
        install_requires=[
            "FireWorks>=1.4.0",
            "pymatgen>=2019.11.11",
            "custodian>=2019.8.24",
            "monty>=2.0.6",
            "numpy",
            "pydantic",
            "emmet",
            "maggma",
            "bson",
            "pydash"
        ],
        extras_require={
            "docs": [
                "mkdocs==1.1.2",
                "mkdocs-material==6.1.0",
                "mkdocs-minify-plugin==0.3.0",
                "mkdocs-macros-plugin==0.4.18",
                "markdown-include==0.6.0",
                "markdown-katex==202009.1026",
            ],
            "rtransfer": ["paramiko>=2.4.2"],
            "plotting": ["matplotlib>=1.5.2"],
            "phonons": ["phonopy>=1.10.8"],
            "tests": ["pytest==6.1.1", "pytest-cov==2.10.1"],
            "dev": [
                "coverage==5.3",
                "codacy-coverage==1.3.11",
                "pycodestyle==2.6.0",
                "mypy==0.790",
                "pydocstyle==5.1.1",
                "flake8==3.8.4",
                "pylint==2.6.0",
                "black==20.8b1",
            ],
        },
        classifiers=[
            "programming language :: python :: 3",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "Operating System :: OS Independent",
            "Topic :: Other/Nonlisted Topic",
            "Topic :: Scientific/Engineering",
        ],
        tests_require=["pytest"],
    )

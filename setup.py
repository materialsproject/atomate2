from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_description = f.read()

if __name__ == "__main__":
    setup(
        name="atomate2",
        setup_requires=["setuptools_scm"],
        use_scm_version={"version_scheme": "python-simplified-semver"},
        description="atomate2 is a library of materials science workflows",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/hackingmaterials/atomate2",
        author="Alex Ganose",
        author_email="alexganose@gmail.com",
        license="modified BSD",
        keywords="high-throughput automated workflow dft vasp",
        package_dir={"": "src"},
        package_data={"atomate2": ["py.typed"]},
        packages=find_packages("src"),
        data_files=["LICENSE"],
        zip_safe=False,
        include_package_data=True,
        install_requires=[
            "pymatgen>=2019.11.11",
            "custodian>=2019.8.24",
            "pydantic",
            "monty",
            "jobflow>=0.1.5",
            "PyYAML",
            "numpy",
            "click",
        ],
        extras_require={
            "amset": ["amset>=0.4.15", "pydash"],
            "docs": [
                "sphinx==4.4.0",
                "numpydoc==1.2",
                "m2r2==0.3.2",
                "mistune==0.8.4",
                "ipython==8.0.1",
                "FireWorks==1.9.8",
                "pydata-sphinx-theme==0.8.0",
                "autodoc_pydantic==1.6.0",
                "sphinx_panels==0.6.0",
            ],
            "tests": [
                "pytest==6.2.5",
                "pytest-cov==3.0.0",
                "FireWorks==1.9.8",
                # "amset==0.4.15",
            ],
            "dev": ["pre-commit>=2.12.1"],
            "phonons": ["phonopy>=1.10.8"],
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "Operating System :: OS Independent",
            "Topic :: Other/Nonlisted Topic",
            "Topic :: Scientific/Engineering",
        ],
        python_requires=">=3.8",
        tests_require=["pytest"],
        entry_points={
            "console_scripts": [
                "atm = atomate2.cli:cli",
            ]
        },
    )

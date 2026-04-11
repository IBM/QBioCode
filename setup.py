"""
QBioCode: Quantum Applications in Healthcare and Life Science Data

A comprehensive suite of computational resources supporting quantum machine learning
applications for healthcare and life science (HCLS) data analysis.
"""

import os
from setuptools import setup, find_packages

# Read version from version.py
def get_version():
    """Extract version from qbiocode/version.py"""
    version_file = os.path.join('qbiocode', 'version.py')
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return '0.0.1'

def read_file(fname):
    """Read file contents from the project root directory."""
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

def read_requirements():
    """
    Read and parse requirements from requirements.txt.
    Separates main dependencies from documentation dependencies.
    """
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    requirements = []
    doc_requirements = []
    in_doc_section = False
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            if '# Documentation' in line:
                in_doc_section = True
            continue
        
        # Add to appropriate list
        if in_doc_section:
            doc_requirements.append(line)
        else:
            requirements.append(line)
    
    return requirements, doc_requirements

# Get requirements
install_requires, docs_require = read_requirements()

# Read long description from README
long_description = read_file('README.md') if os.path.exists('README.md') else ''

setup(
    name="qbiocode",
    version=get_version(),
    
    # Author information
    author="Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida",
    maintainer="IBM Research",
    
    # Project description
    description=(
        "A comprehensive suite of computational resources for quantum computing "
        "applications in healthcare and life science data analysis"
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    # Project URLs
    url="https://github.com/IBM/qbiocode",
    project_urls={
        "Documentation": "https://ibm.github.io/QBioCode/",
        "Source Code": "https://github.com/IBM/qbiocode",
        "Bug Tracker": "https://github.com/IBM/qbiocode/issues",
    },
    
    # License
    license="Apache License 2.0",
    
    # Package discovery
    packages=find_packages(exclude=['tests', 'tests.*', 'docs', 'docs.*', 'archive', 'archive.*']),
    
    # Include package data
    include_package_data=True,
    package_data={
        'qbiocode': ['py.typed'],
        'apps.qprofiler': ['configs/*.yaml'],
    },
    
    # Dependencies
    install_requires=install_requires,
    extras_require={
        'apps': [
            'hydra-core',
            'joblib',
        ],
        'docs': docs_require,
        'dev': docs_require + [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
            'mypy>=1.0',
            'types-PyYAML',
        ],
        'all': docs_require + [
            'hydra-core',
            'joblib',
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=23.0',
            'flake8>=6.0',
            'mypy>=1.0',
            'types-PyYAML',
        ],
    },
    
    # Python version requirement
    python_requires='>=3.10,<3.13',
    
    # PyPI classifiers
    classifiers=[
        # Development status
        "Development Status :: 3 - Alpha",
        
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        
        # License
        "License :: OSI Approved :: Apache Software License",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3 :: Only",
        
        # Operating system
        "Operating System :: OS Independent",
        
        # Natural language
        "Natural Language :: English",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "quantum machine learning",
        "quantum computing",
        "qiskit",
        "bioinformatics",
        "healthcare",
        "life sciences",
        "omics",
        "genomics",
        "proteomics",
        "metabolomics",
        "data complexity",
        "model profiling",
        "model selection",
        "oracle",
    ],
    
    # Console scripts for command-line tools
    entry_points={
        'console_scripts': [
            'qprofiler=apps.qprofiler.cli:main',
            'qprofiler-batch=apps.qprofiler.qprofiler_batchmode:main',
            'qsage=apps.sage.sage:main',
        ],
    },
    
    # Zip safety
    zip_safe=False,
)

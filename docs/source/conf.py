# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

# Mock imports for packages that have issues
autodoc_mock_imports = ['xgboost']



project = 'qbiocode'
copyright = '2025 IBM Research' #, Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida'
author = 'Bryan Raubenolt, Aritra Bose, Kahn Rhrissorrakrai, Filippo Utro, Akhil Mohan, Daniel Blankenberg, Laxmi Parida'
release = '0.0.1'

# Documentation note
html_show_sphinx = True
html_show_copyright = True

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [ "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
     "nbsphinx",
   "myst_parser",
   "sphinx_design",
  
  # "myst_nb"
]

templates_path = ['_templates']
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ['custom.css']  # Add custom CSS
nbsphinx_execute = 'never' # O 'auto', 'always'. 'never'

# nbsphinx configuration for better notebook rendering
nbsphinx_prolog = """
.. raw:: html

    <style>
        /* Improve notebook cell spacing */
        div.nbinput.container,
        div.nboutput.container {
            margin-bottom: 1em;
        }
        
        /* Better code cell styling */
        div.highlight-ipython3 {
            margin: 0.5em 0;
        }
        
        /* Improve markdown cell rendering */
        .nbinput + .nboutput {
            margin-top: 0.5em;
        }
    </style>
"""

# Prompt for notebook execution
nbsphinx_prompt_width = "0"


# Impostazioni di MyST-Parser
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# nbsphinx automatically handles .ipynb files when included in extensions

def run_apidoc(app):
    """Generate API documentation"""
    try:
        import better_apidoc

        better_apidoc.APP = app
        better_apidoc.main(
            [
                "better-apidoc",
                "-t",
                os.path.join(".", "_templates"),
                "--force",
                "--no-toc",
                "--separate",
                "-o",
                os.path.join("source/", "api"),
                os.path.join("..", "qbiocode"),
            ]
        )
    except Exception as e:
        print(f"Warning: API documentation generation failed: {e}")
        print("Continuing with build...")


# -- Extension configuration -------------------------------------------------
add_module_names = False


napoleon_google_docstring = True
napoleon_include_init_with_doc = True

coverage_ignore_modules = []
coverage_ignore_functions = []
coverage_ignore_classes = []


myst_enable_extensions = [
    "colon_fence",  # Ensures ::: blocks work
  #  "linkify",
    "strikethrough",
    "tasklist",
    # ... any other extensions you want
]


coverage_show_missing_items = True
html_theme = 'pydata_sphinx_theme' #'sphinx_rtd_theme' # 'furo'

html_show_sourcelink = False

html_logo = "_static/QBioCode_logo.png"
if os.path.exists(os.path.join(os.path.dirname(__file__), "_static", "favicon.ico")):
    html_favicon = "_static/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IBM/QBioCode",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "show_prev_next": True,  # Enable prev/next navigation
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],  # Add navigation to center
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "header_links_before_dropdown": 8,
    "navigation_depth": 3,  # Increased depth for better navigation
    "show_toc_level": 2,  # Show 2 levels in TOC
    "collapse_navigation": False,  # Keep navigation expanded
    "navigation_with_keys": True,  # Enable keyboard navigation
}

html_context = {
    "default_mode": "light",
}

# Footer text
rst_epilog = """
.. |ai_note| replace:: *Portions of this documentation were generated with AI assistance.*
"""

# Enable sidebars for better navigation
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"],
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

def setup(app):
    app.connect("builder-inited", run_apidoc)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
from datetime import datetime
from importlib import import_module

from jinja2.filters import FILTERS

# sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "TorchDyno"
copyright = str(datetime.now().year) + ", Valerio De Caro"
author = "Valerio De Caro"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "enum_tools.autoenum",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.icon",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx_book_theme",
    "sphinx_favicon",
    "sphinx_design",
    "myst_parser",
]
autosummary_generate = True
myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

coverage_show_missing_items = True
# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "_static/images/logo_no_name.png"

html_static_path = ["_static"]
html_css_files = ["css/buttons.css", "css/custom.css"]

html_context = {
    "github_url": "https://github.com",  # or your GitHub Enterprise site
    "github_user": "vdecaro",
    "github_repo": "torchdyno",
    "github_version": "main",
    "doc_path": "docs",
}

external_links = [
    {
        "name": "Changelog",
        "url": "https://github.com/vdecaro/torchdyno/blob/main/CHANGELOG.md",
    }
]

# favicons = [
#     {
#         "rel": "icon",
#         "sizes": "16x16",
#         "href": "favicon/favicon-16x16.png",
#     },
#     {
#         "rel": "icon",
#         "sizes": "32x32",
#         "href": "favicon/favicon-32x32.png",
#     },
#     {
#         "rel": "apple-touch-icon",
#         "sizes": "180x180",
#         "href": "favicon/apple-touch-icon.png",
#         "color": "#000000",
#     },
#     {
#         "rel": "android-chrome",
#         "sizes": "192x192",
#         "href": "favicon/android-chrome-192x192.png",
#     },
#     {
#         "rel": "android-chrome",
#         "sizes": "512x512",
#         "href": "favicon/android-chrome-512x512.png",
#     },
#     {
#         "rel": "manifest",
#         "href": "favicon/site.webmanifest",
#     },
#     {
#         "rel": "shortcut icon",
#         "href": "favicon/favicon.ico",
#     },
# ]

icon_links = [
    {
        "name": "GitHub",
        "url": "https://github.com/vdecaro/torchdyno",
        "icon": "fa-brands fa-github",
        "type": "fontawesome",
    }
]

html_theme_options = {
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button-field"],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["last-updated", "copyright"],
    "footer_center": ["sphinx-version"],
    "footer_end": ["theme-version"],
    "show_nav_level": 1,
    "navigation_depth": 4,
    "icon_links": icon_links,
    "external_links": external_links,
    "header_links_before_dropdown": 4,
    "logo": {
        "text": "TorchDyno",
        "alt_text": "TorchDyno - Home",
        "image_dark": "_static/images/logo_no_name.png",
        "image_light": "_static/images/logo_no_name.png",
    },
}

html_sidebars = {"**": ["sidebar-nav-bs", "sidebar-ethical-ads"]}

html_scaled_image_link = False


def filter_out_undoc_class_members(member_name, class_name, module_name):
    module = import_module(module_name)
    cls = getattr(module, class_name)
    if getattr(cls, member_name).__doc__:
        return f"~{class_name}.{member_name}"
    else:
        return ""


def filter_out_parent_class_members(member_name, class_name, module_name):
    module = import_module(module_name)
    cls = getattr(module, class_name)
    if member_name in cls.__dict__:
        return f"~{class_name}.{member_name}"
    else:
        return ""


FILTERS["filter_out_undoc_class_members"] = filter_out_undoc_class_members
FILTERS["filter_out_parent_class_members"] = filter_out_parent_class_members

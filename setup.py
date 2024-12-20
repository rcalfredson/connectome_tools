from setuptools import setup, find_packages

setup(
    name="contools-optimized",  # Updated name
    version="2.1.0",
    description="An optimized version of connectome_tools with performance enhancements and multi-processing",
    url="http://github.com/rcalfredson/connectome_tools",
    author="Michael Winding, updated by R.C. Alfredson",
    author_email="mwinding@alumni.nd.edu, robert.c.alfredson@gmail.com",
    license="MIT",
    packages=find_packages(include=["contools", "contools.*"]),
    install_requires=["anytree", "python-catmaid", "scipy", "tables", "upsetplot"],
    long_description=(
        "This is an optimized fork of connectome_tools, originally authored by Michael Winding. "
        "This version includes performance enhancements such as better memory management and multi-processing capabilities, "
        "maintained by R.C. Alfredson. For the original repository, see http://github.com/mwinding/connectome_tools."
    ),
)

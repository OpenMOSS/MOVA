import os

from setuptools import find_packages, setup

version_file = "dover/version.py"


def readme():
    if not os.path.exists("README.md"):
        return ""
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


def get_version():
    with open(version_file, "r") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def get_requirements(filename="requirements.txt"):
    here = os.path.dirname(os.path.realpath(__file__))
    req_path = os.path.join(here, filename)
    if not os.path.exists(req_path):
        return []
    with open(req_path, "r") as f:
        requires = [line.replace("\n", "") for line in f.readlines()]
    return requires


setup(
    name="dover",
    version=get_version(),
    description="Disentangled Video Quality Evaluator",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Teo (Timothy) Wu Hao Ning",
    author_email="realtimothyhwu@gmail.com",
    keywords="computer vision, video quality assessment",
    url="https://github.com/twowu/dover",
    include_package_data=True,
    packages=find_packages(exclude=("examplar_data_labels", "figs")),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    license="MIT License",
    setup_requires=["numpy"],
    install_requires=get_requirements(),
    ext_modules=[],
    cmdclass={},
    zip_safe=False,
)

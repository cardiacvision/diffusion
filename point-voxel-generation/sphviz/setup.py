import setuptools

setuptools.setup(
    name="sphtools",
    url="",
    classifiers=["Private :: Do not Upload"],
    python_requires=">=3.7",
    packages=setuptools.find_packages(),
    install_requires=[
        # sphtools dependencies
        "numpy",
        "matplotlib",
        "pyvistaqt",
        "imageio-ffmpeg",
        "PyQt5",
        "more-itertools",
        "cmocean",
        "pyacvd @ git+https://github.com/pyvista/pyacvd", # upstream changes merged, but no new release
        "pymeshfix",
        "pygem @ git+https://github.com/mathLab/PyGeM", # sparse_conv dependencies
        "scipy",
        "tqdm",
        "ffmpeg-python",
    ],
    entry_points={
        "console_scripts": [
            "sphviz = sphtools.visualize:main",
        ]
    },
)
"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from os import path
from setuptools import setup

PATH_HERE = path.abspath(path.dirname(__file__))

with open(path.join(PATH_HERE, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


with open(path.join(PATH_HERE, "pyproject.toml"), encoding="utf-8") as fp:
    requirements = [rq.rstrip() for rq in fp.readlines() if not rq.startswith("#")]

setup(
    name="scrapcore",
    version="0.0.1",
    package_data={
        "scrap_core": ["py.typed"],
        "scrap_core.datamodel": ["py.typed", "load_full_heat_from_oko.sql", "load_heat_plan_from_iars.sql"],
        "scrap_core.blendmodel": [
            "py.typed",
            "trained_models/blend_model_v1.pth",
            "trained_models/eob_model_v6.pth",
        ],
        "scrap_core.yieldmodel": ["py.typed"],
        "scrap_core.meltabilitymodel": ["py.typed"],
        "scrap_core.optimization": ["py.typed"],
        "scrap_core.correctiontechnologiesmodel": ["py.typed"],
        "scrap_core.evaluation": ["py.typed"],
        "scrap_core.blendmodel.eob_datamodule": ["py.typed"],
        "scrap_core.home_scrap_estimation": [
            "py.typed",
            "estimators/1PIT.json",
            "estimators/DSI.json",
            "estimators/HS.json",
            "estimators/HSB.json",
            "estimators/HST.json",
            "estimators/HSZ.json",
        ],
    },
    packages=[
        "scrap_core",
        "scrap_core.datamodel",
        "scrap_core.blendmodel",
        "scrap_core.home_scrap_estimation",
        "scrap_core.yieldmodel",
        "scrap_core.optimization",
        "scrap_core.meltabilitymodel",
        "scrap_core.correctiontechnologiesmodel",
        "scrap_core.evaluation",
        "scrap_core.datamodel.grades",
        "scrap_core.blendmodel.eob_datamodule",
        "scrap_core.pytorch_balanced_sampler",
        "scrap_core.weightmodel",
    ],
    description="Calculations for scrap model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch==1.9.0",
        "torchvision==0.10.0",
        "pytorch-lightning==1.3.5",
        "attrs==20.2.0",
        "cattrs==1.0.0",
        "numpy==1.19.1",
        "tqdm==4.48.2",
        "immutables==0.14",
        "pymssql==2.1.5",
        "python-dotenv==0.20.0",
        "jsonlines==1.2.0",
        "bayesian-optimization==1.1.0",
        "deap==1.3.1",
        "cachetools==4.2.1",
        "pyodbc==4.0.30",
        "spacecutter==0.2.0",
        "torchmetrics==0.3.2",
        "scipy==1.7.3",
        "returns==0.18.0",
    ],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "generate_blend_model_training_data=scrap_core.generate_blendmodel_training_data:main",
            "pig_iron=scrap_core.pigironanalysis:main",
            "train_eob_model=scrap_core.blendmodel.eob_model_training:main",
            "test_eob_model=scrap_core.blendmodel.eob_model_test:main",
            "train_home_scrap_estimators=scrap_core.home_scrap_estimation.train_home_scrap_estimators:main",
        ]
    },
)

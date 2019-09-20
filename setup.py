from setuptools import setup

setup(name='ShapExplainers',
    version='1.0.0',
    description='Wrappers for shapley value explanations for model interpretability',
    packages=['ShapExplainers'],
    install_requires=["typing",
                      "pandas",
                      "numpy == 1.15.4",
                      'scikit-learn',
                      "shap == 0.29.3",
                      ],
)
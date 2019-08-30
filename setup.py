from setuptools import setup

setup(name='Interpretability',
    version='1.0.0',
    description='Wrappers for shapley value explanations for model interpretability',
    packages=['Interpretability'],
    install_requires=["typing",
                      "pandas"
                      "numpy == 1.15.4",
                      'scikit-learn == 0.20.3',
                      "shap == 0.29.3",
                      ],
)
from setuptools import setup, find_packages

setup(
    name='probench',
    version='0.0.1',
    description='ProBench',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    url='https://github.com/Yan98/ProBench_eval',
    install_requires=[
        "datasets",
        "torch",
        "shortuuid",
        "vllm",
        "openai",
        "numpy",
        "regex",
        "Pillow",
        "pandas",
        "tiktoken",
        "scikit-learn",
    ]
)

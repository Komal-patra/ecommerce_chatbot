from setuptools import find_packages, setup

setup(
    name="ecommbot",
    version="0.0.1",
    author="Komal",
    author_email="komalfsds2022@gmail.com",
    packages=find_packages(),
    install_requires=['langchain-astradb','langchain ','langchain-openai','datasets','pypdf','python-dotenv','flask']
)
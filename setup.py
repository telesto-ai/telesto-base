import setuptools


with open("requirements.txt", "r") as req_f:
    install_requires = req_f.read().splitlines()


setuptools.setup(
    name="telesto-base",
    version="0.0.1",
    author="telesto.ai",
    author_email="contact@telesto.ai",
    description="Base tools for telesto.ai models",
    url="https://github.com/telesto-ai/telesto-base",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=install_requires
)
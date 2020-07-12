# telesto-base
Base image for telesto.ai models.

# Instructions

`telesto-base` contains a pip-installable Python package and a Docker image, allowing you to
easily package your models for telesto.ai competitions.

## telesto-base module
To install the module, you can simply use pip:
```
pip install git+https://github.com/telesto-ai/telesto-base.git@master
```
If you would like to use the latest not yet released version, you can install the one in the 
`develop` branch.
```
pip install git+https://github.com/telesto-ai/telesto-base.git@develop
```

## The base image
The base image contains the pre-installed `telesto-base` module. Your submissions will use this
as a base, so you'll only have to worry about the algorithms and not the packaging. To use it
locally, you can pull the image from Docker:
```
docker pull telestoai/model-api-base:stable
```

Alternatively, the image can also be built locally with the command 
```
docker build -t telestoai/model-api-base -f Dockerfile .
```

## An example model
If you are stuck on how to prepare your model for submission, we have prepared a concrete example
for you. The example is available in the [telesto-models](https://github.com/telesto-ai/telesto-models) repository with further instructions
on the usage.

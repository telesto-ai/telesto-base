# telesto-base
Base Docker image and tools for telesto.ai models.

# Instructions

`telesto-base` contains a pip-installable Python package and a Docker image, allowing you to
easily package your models for telesto.ai competitions.

## telesto-base package
To install the module, you can simply use pip:
```
pip install telesto-base
```
If you would like to use the latest not yet released version, you can install the one in the 
`develop` branch.
```
pip install git+https://github.com/telesto-ai/telesto-base.git@develop
```

## The base image
The base image contains the pre-installed `telesto-base` module. Your submissions will use this
as a base, so you'll only have to worry about the algorithms and not the packaging. To use it
locally, you can pull the image from Docker Hub:
```
docker pull telestoai/model-api-base:latest
```

Alternatively, the image can also be built locally with the command 
```
docker build -t telestoai/model-api-base -f Dockerfile .
```

## An example model
If you are stuck on how to prepare your model for submission, we have prepared a concrete example
for you. The example is available in the [telesto-models](https://github.com/telesto-ai/telesto-models) repository with further instructions
on the usage.

## Test classification model API

Build and start a container
```
docker build -t telestoai/model-api-base -f Dockerfile .
docker run -p 9876:9876 --name model-api-base --rm --env USE_FALLBACK_MODEL=1 \
    telestoai/model-api-base classification
```

Send a sample input
```
curl -X POST -H "Content-Type:application/json" --data-binary @tests/data/class/example-input.json -i \
    http://localhost:9876/
...
{
    "predictions": [
        {"probs": {"cat": 0.32015, "dog": 0.67985}, "prediction": "dog"},
        {"probs": {"cat": 0.81545, "dog": 0.18455}, "prediction": "cat"}
    ]
}
```

## Test segmentation model API

Build and start a container
```
docker build -t telestoai/model-api-base -f Dockerfile .
docker run -p 9876:9876 --name model-api-base --rm --env USE_FALLBACK_MODEL=1 \
    telestoai/model-api-base segmentation
```

Post a sample input
```
curl -X POST -H "Content-Type:application/json" --data-binary @tests/data/segm/example-input.json -i \
    http://localhost:9876/jobs
...
{
    "job_id": "b741bd19767441f6b7abd022744083c9"
}
```

Get the result
```
curl -H "Content-Type:application/json" -i http://localhost:9876/jobs/b741bd19767441f6b7abd022744083c9
...
{
    "mask": {
        "content": "<BASE_64_IMAGE>"
    }
}
```

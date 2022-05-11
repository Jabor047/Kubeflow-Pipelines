#!/bin/bash -e
image_name=jabor047/kubeflow-pipeline-reusable-components-feature-prep
image_tag=latest
full_image_name=${image_name}:${image_tag}

# for arm based MACs use the following:
docker buildx build -t "${full_image_name}" . --platform linux/amd64

# for x86 based machines use the following:
# docker build -t ${full_image_name} .

docker push ${full_image_name}

# Output the strict image name, which contains the sha256 image digest
docker inspect --format="{{index .RepoDigests 0}}" "${full_image_name}"
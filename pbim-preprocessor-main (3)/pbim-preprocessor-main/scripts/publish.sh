#!/bin/bash
USER=${1:-"s364083"}
TOKEN=${2:-$(cat ~/MA/.github_token)}
poetry config repositories.gitlab https://gitlab2.informatik.uni-wuerzburg.de/api/v4/projects/16019/packages/pypi
poetry publish -r gitlab -u $USER -p $TOKEN
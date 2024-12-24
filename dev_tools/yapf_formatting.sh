#!/usr/bin/env bash

for file in bin/*py py/squeze/*py py/squeze/*/*py
do
  echo "yapf --style google $file -i"
  yapf --style google $file -i
done

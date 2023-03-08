#!/bin/bash
pwd
for filename in tutorials/*.ipynb; do
    jupyter nbconvert "$filename" --to markdown --execute --output "../docs/$(basename "$filename" .ipynb).md"
done

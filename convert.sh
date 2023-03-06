#!/bin/bash
pwd
for filename in tutorials/*.ipynb; do

# jupyter nbconvert *.ipynb --to markdown --execute 
# jupyter nbconvert "$filename" --to markdown --execute --output "$(basename "$filename" .ipynb).md"
jupyter nbconvert "$filename" --to markdown --execute --output "../docs/$(basename "$filename" .ipynb).md"
# cp "$(basename "$filename" .ipynb).md" ../docs/
#cp -r . ../docs/
done

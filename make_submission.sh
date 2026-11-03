#!/bin/bash

# Zip files for upload
zip -r pr4_submission.zip . -i "./task*/*" "grade.py" -x "*.so" "**/__pycache__/*"

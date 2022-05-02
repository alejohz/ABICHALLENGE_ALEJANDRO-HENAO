#!/usr/bin/env bash


ACCESS_KEY_ID = 'AKIAYBSFEO5WE6WXM5G6'
SECRET_ACCESS_KEY = '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'
AWS_DEFAULT_REGION = 'us-east-1'

aws configure set aws_access_key_id $ACCESS_KEY_ID
aws configure set aws_secret_access_key $SECRET_ACCESS_KEY
aws configure set default.region $AWS_DEFAULT_REGION

python ahenao_web_app.py
open http://localhost:8501
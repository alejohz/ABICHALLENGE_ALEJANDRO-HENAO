###############
# BUILD IMAGE #
###############
FROM python:3.8.2-slim-buster AS build

# virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc
    
# add and install requirements
RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

RUN pip install awscli
ENV ACCESS_KEY_ID = 'AKIAYBSFEO5WE6WXM5G6'
ENV SECRET_ACCESS_KEY = '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'
ENV AWS_DEFAULT_REGION = 'us-east-1'
RUN aws configure set aws_access_key_id $ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $SECRET_ACCESS_KEY
RUN aws configure set default.region $AWS_DEFAULT_REGION

#################
# RUNTIME IMAGE #
#################
FROM python:3.8.2-slim-buster AS runtime

# setup user and group ids
ARG USER_ID=1000
ENV USER_ID $USER_ID
ARG GROUP_ID=1000
ENV GROUP_ID $GROUP_ID

# add non-root user and give permissions to workdir
RUN groupadd --gid $GROUP_ID user && \
          adduser user --ingroup user --gecos '' --disabled-password --uid $USER_ID && \
          mkdir -p /usr/src/app && \
          chown -R user:user /usr/src/app

# copy from build image
COPY --chown=user:user --from=build /opt/venv /opt/venv

# set working directory
WORKDIR /usr/src/app

# switch to non-root user
USER user

# disables lag in stdout/stderr output
ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
# Path
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 8501

# Run streamlit
CMD streamlit run ahenao_web_app.py
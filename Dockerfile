FROM jupyter/base-notebook as base

USER root

# Install requirements for pygraphviz and stuff for interacting with git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    graphviz \
    graphviz-dev\
    gnupg2 \
    git \
    openssh-client && \
    rm -rf /var/lib/apt/lists/* && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" | tee -a /etc/sudoers

USER ${NB_USER}

# Install in the default python3 environment
RUN pip install --no-cache-dir 'flake8' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

WORKDIR /workspaces
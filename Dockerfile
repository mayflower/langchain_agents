FROM jupyter/base-notebook as base

# Install in the default python3 environment
RUN pip install --no-cache-dir 'flake8' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# Install from the requirements.txt file
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

FROM base as devcontainer

USER root

# Install git and gpg for interacting with the git repository
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gnupg2 \
    git \
    openssh-client && \
    rm -rf /var/lib/apt/lists/* 
# Add sudo support for the nbuser (disabled)
# echo "${NB_USER} ALL=(ALL) NOPASSWD:ALL" | tee -a /etc/sudoers

USER ${NB_USER}

WORKDIR /workspaces
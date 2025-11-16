FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

USER root
RUN apt-get update && \
  apt-get install -y --no-install-recommends openssh-server && \
  rm -rf /var/lib/apt/lists/* && \
  mkdir -p /run/sshd && \
  ssh-keygen -A

ARG PORT=65142
RUN echo "Port ${PORT}" >> /etc/ssh/sshd_config

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  sudo \
  curl \
  tmux \
  git \
  build-essential \
  vim \
  htop \
  python3-dev \
  ninja-build \
  wget \
  ca-certificates \
  unzip \
  screen \
  less \
  nano && \
  rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -q \
  lightning \
  click \
  transformers \
  goatools \
  toml \
  wget \
  pydantic \
  loguru \
  wandb \
  tqdm \
  einops \
  obonet \
  fastobo \
  h5py \
  seaborn \
  scikit-learn \
  ipython \
  sentencepiece \
  sentence-transformers \
  tensorflow==2.18.0 \
  tf-keras==2.18.0

EXPOSE ${PORT}

CMD ["/usr/sbin/sshd","-D", "-e"]
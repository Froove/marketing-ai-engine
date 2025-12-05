FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Eviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# Mise à jour de base
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le projet
COPY . /app

# Installer les dépendances Python
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Par défaut : on ne lance rien. La commande sera fournie par le runner (Runpod/Vast/etc.)
CMD ["/bin/bash"]


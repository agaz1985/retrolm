FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools
RUN apt-get update && apt-get install -y \
    nasm \
    gcc \
    gcc-multilib \
    g++-multilib \
    make \
    curl \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install DJGPP from prebuilt binaries
RUN mkdir -p /djgpp && cd /djgpp && \
    curl -L https://github.com/andrewwutw/build-djgpp/releases/download/v3.4/djgpp-linux64-gcc1220.tar.bz2 -o djgpp.tar.bz2 && \
    tar xjf djgpp.tar.bz2 && \
    rm djgpp.tar.bz2

# Add DJGPP to PATH (note the double djgpp directory)
ENV PATH="/djgpp/djgpp/bin:${PATH}"

WORKDIR /project
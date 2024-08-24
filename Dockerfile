FROM ubuntu:latest

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install required packages
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        tmux \
        git \
        python3 \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH environment variable
ENV PYTHONPATH="/root/spar-2024:${PYTHONPATH}"

# Copy the startup script into the container
COPY runpod_startup.sh /startup.sh
RUN chmod +x /startup.sh

# Set the startup script as the entry point
ENTRYPOINT ["/startup.sh"]
CMD ["/bin/bash"]

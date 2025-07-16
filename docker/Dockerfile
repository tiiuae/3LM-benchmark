
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker
# FROM 654654449790.dkr.ecr.us-east-1.amazonaws.com/cuda121_nccl2215_pt220_efa1310_h100_train
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker
# FROM google/cloud-sdk:slim
# FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Install required packages and prerequisites
# RUN apt-get update && \
#     apt-get install -y apt-transport-https ca-certificates gnupg curl && \
#     rm -rf /var/lib/apt/lists/*

# # Add the Google Cloud SDK distribution URI as a package source
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
#     | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# # Import the Google Cloud public key for package verification
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
#     | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# # Update package lists and install the Google Cloud SDK
# RUN apt-get update && \
#     apt-get install -y google-cloud-sdk && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

# Install requirements
COPY modules/evaluators/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --no-cache-dir

# RUN apt-get update \
#  && apt-get install -y pigz
#  RUN apt-get update && \
 #     apt-get install -y apt-transport-https ca-certificates gnupg curl && \
 #     rm -rf /var/lib/apt/lists/*
 
 # # Add the Google Cloud SDK distribution URI as a package source
 # RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" \
 #     | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
 
 # # Import the Google Cloud public key for package verification
 # RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
 #     | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
 
 # # Update package lists and install the Google Cloud SDK
 # RUN apt-get update && \
 #     apt-get install -y google-cloud-sdk && \
 #     rm -rf /var/lib/

RUN apt update && apt install -y rsync pigz


CMD ["/bin/bash"]
ENV NLTK_DATA /opt/ml/nltk_data
# Create the NLTK data directory and download 'punkt' tokenizer data
RUN mkdir -p $NLTK_DATA && \
    python -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DATA')"

# Copy the source
COPY . .
RUN rm requirements.txt
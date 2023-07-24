FROM registry.access.redhat.com/ubi8/python-38

# Add application sources with correct permissions for OpenShift
# USER 0
# ADD app-src .
# RUN chown -R 1001:0 ./
# USER 1001

# Install the dependencies
COPY . .
USER root
RUN yum install -y git-lfs
RUN  git lfs install
RUN pip install -U "pip>=19.3.1" && \
    pip install bitsandbytes && \
    pip install huggingface_hub && \
    pip install -r requirements.txt

RUN huggingface-cli login --token xxxx

# Run the application
CMD python app.py
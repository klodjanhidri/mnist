FROM python:3.9
#FROM pytorch/pytorch:latest 

# Set the working directory inside the container
WORKDIR /workspace

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /workspace

RUN pip3 install --upgrade pip
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
ENV WATCHFILES_FORCE_POLLING=true
#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch torchvision torchaudio
# Install any dependencies if needed (uncomment if required)
EXPOSE 8000


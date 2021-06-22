 FROM python:3.8.8
 ADD example.py .
 RUN useradd --create-home --shell /bin/bash app_user
 WORKDIR /home/app_user
 COPY requirements.txt ./
 RUN pip install --no-cache-dir -r requirements.txt
 USER app_user
 COPY . .
 ENTRYPOINT ["python", "./example.py"]
 #CMD ["bash"]

# ---------

# FROM python:3.8.8

# ADD example.py .

# WORKDIR /app
# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# ENTRYPOINT ["python", "./example.py"]
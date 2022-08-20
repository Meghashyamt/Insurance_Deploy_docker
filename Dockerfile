FROM continuumio/anaconda3:4.4.0
COPY . /home/shyam/Downloads/Insurance-P-Deploy-main/
EXPOSE 5000
WORKDIR /home/shyam/Downloads/Insurance-P-Deploy-main/
RUN pip install scikit-learn==0.22.2
RUN pip install --upgrade pip
CMD python app.py

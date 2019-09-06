FROM kaggle/python

# our instrumentation uses astunparse sometimes
RUN pip install astunparse

# install an older version of sklearn
# a lot of these scripts won't run with the latest and greatest
RUN pip install scikit-learn==0.18

ADD . /opt/collection
WORKDIR /opt/collection

EXPOSE 5000

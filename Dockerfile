FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

COPY docker/requirements.txt /src/requirements.txt
RUN pip install -r /src/requirements.txt

COPY *.py /src/
RUN echo "#! $(which python)" | cat - /src/train.py > /src/train.tmp.py
RUN mv /src/train.tmp.py /src/train.py
RUN chmod +x /src/train.py
RUN ln -s /src/train.py /usr/local/bin/adain-train
RUN echo "#! $(which python)" | cat - /src/test.py > /src/test.tmp.py
RUN mv /src/test.tmp.py /src/test.py
RUN chmod +x /src/test.py
RUN ln -s /src/test.py /usr/local/bin/adain-test

COPY docker/bash.bashrc /etc/bash.bashrc
RUN chmod 777 -R /workspace
ENV XDG_CACHE_HOME=/workspace/.cache

CMD [ "bash" ]

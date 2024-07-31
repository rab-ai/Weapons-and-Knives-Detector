FROM python:3-alpine3.11
WORKDIR /app
COPY tmp/requirements.txt requirements.txt
RUN python -m /py && \
   /py/bin/pip install --upgrade pip && \
   apk add --update alpine-sdk && \
   apk add --update --no-cache postgresql-client && \
   apk add --update --no-cache --virtual .tmp-build-deps \
      build-base gcc python3-dev postgresql-dev musl-dev libffi-dev openssl-dev cargo  && \
   /py/bin/pip install -r /tmp/requirements.txt && \
   if [ $DEV = "true" ]; \
      then /py/bin/pip install -r /tmp/requirements.dev.txt ; \
   fi && \
   rm -rf /tmp && \
   apk del .tmp-build-deps && \
   adduser \
      --disabled-password \
      --no-create-home 
EXPOSE 4000
CMD python ./app.py
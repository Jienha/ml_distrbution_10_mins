# 1. image
FROM python:3.11

# 2. Code directory inside the our container:
WORKDIR /code

# 3. Container is an isolated env, so we need to copy all the necessary files that we need the container executes
COPY requirements.txt /code/requirements.txt

# 4. Install our requirements, therefore we run pip install inside the container
RUN pip install --no-cache-dir -r /code/requirements.txt

# 5. Copy our application folder inside our working directory code:
COPY ./app /code/app

# 6 expose some port because this is an isolated environment
# So we expose some endpoint where we can send the data or we can interact with this container
EXPOSE 8000

# 7. Command container is going to run after it is created
# "uvicorn" is server getway interface. It's the binning element that handles the web connection from the browser or API
# client  and then allows FastAPI to serve actual request. It also allows you to properly spawn many workers and scale you
# deployment properly the app.
CMD [ "uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]

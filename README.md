# Tutorial - ML distribution with Docker container


1. Build docker container:
    > $ docker build -t image_name .
---
2. Run container:
    > $ docker run --name countainer_name -p 8000:8000 image_name
---
3. Output will contain:
INFO:   Uvicorn running on http://0.0.0.0:8000
    - use this url in chrome to see the model frontend
    - use for testing the model
---
4. Query model 
    
    4.1 Via web interface (chrome):
        http://0.0.0.0:8000/docs -> test model
    
    4.2 Via python client:
        client.py
    
    4.3 Via curl request:
        
        $ curl -X POST "http://0.0.0.0:8000/predict" -H "accept: application/json" -H "Content-Type: application/json" -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
---
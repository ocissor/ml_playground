from fastapi import FastAPI, UploadFile, Form, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from backend.app.models import train_test_models
from typing import List
from pathlib import Path
import sys
sys.path.append(Path(__file__).parent)  # Add parent directory to sys.path
from backend.app.eda_plot import *
from backend.app.graph.graph import analyze_data_graph
from backend.app.data_storage import dataframe, conversation_history
import pickle

app = FastAPI(title="ML Playground Backend")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/train")
async def train_model(file: UploadFile, target_column: str = Form(...), model_name: str = Form(...), problem_type: str = Form("classification"), hyperparameter : str = Form(...)):
    # Read uploaded CSV into dataframe
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    result = train_test_models(df, target_column, model_name, hyperparameter, problem_type=problem_type)

    return {'model_outputs': result}

@app.post("/data_stats")
async def get_data_stats(file: UploadFile):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {'data_stats':df.describe().to_json()}

@app.post('/visualize_data')
async def visualize_data(file: UploadFile):
    contents = await file.read()
    if not contents:
        return {"error": "Uploaded file is empty"}
    df = pd.read_csv(io.BytesIO(contents))

    images = univariate_plots(df)
    return images


@app.post('/chat_with_data')
async def chat_with_data(file: UploadFile, user_input: str = Form(...), uuid: str = Form(...)):
    contents = await file.read()
    if not contents:
        return {"error": "Uploaded file is empty"}
    df = pd.read_csv(io.BytesIO(contents))
    if uuid not in conversation_history:
        conversation_history[uuid] = []

    graph_input = {
        'user_input': user_input,
        'messages': conversation_history[uuid],
        'uuid': uuid
    }

    output = analyze_data_graph.invoke(graph_input, config={"configurable": {"thread_id": uuid}})
    conversation_history[uuid] = output['messages']
    return Response(content=pickle.dumps(output), media_type="application/octet-stream")

@app.post('/upload_file')
async def save_data_to_dataframe(file: UploadFile, uuid: str = Form(...)):
    content = await file.read()
    if not content:
        return {"error":"Uploaded file is empty"}

    df = pd.read_csv(io.BytesIO(content))
    if uuid not in dataframe:
        dataframe[uuid] = df.to_json(orient="records")
    else:
        return {"message": "UUID already exists. Data not overwritten."}
    
    return {"message": "File uploaded and data saved successfully."}

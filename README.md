<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# ML_PLAYGROUND

<em></em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=default&logo=Streamlit&logoColor=white" alt="Streamlit">
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=default&logo=scikit-learn&logoColor=white" alt="scikitlearn">
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style=default&logo=FastAPI&logoColor=white" alt="FastAPI">
<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
<img src="https://img.shields.io/badge/Pytest-0A9EDC.svg?style=default&logo=Pytest&logoColor=white" alt="Pytest">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
<br>
<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style=default&logo=Plotly&logoColor=white" alt="Plotly">
<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=default&logo=OpenAI&logoColor=white" alt="OpenAI">
<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=default&logo=Pydantic&logoColor=white" alt="Pydantic">
<img src="https://img.shields.io/badge/LangGraph-3C75AF.svg?style=default&logo=LangChain&logoColor=white" alt="LangGraph">
<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=default&logo=YAML&logoColor=white" alt="YAML">

</div>
<br>

---
# ML_PLAYGROUND

Lightweight playground to prototype ML experiments: a FastAPI backend for data processing and model serving, and a Streamlit frontend for quick EDA, visualization and training workflows.

## Features
- FastAPI backend with modular agents and simple in-memory data storage
- Streamlit frontend for interactive EDA, training and visualization
- Graph utilities and plotting helpers
- Dockerfiles for backend and frontend for containerized development
- Pytest-based test scaffolding

## Project structure (high-level)
- backend/
  - app/
    - main.py — FastAPI app entrypoint (routes, startup)
    - data_storage.py — dataset and artifact storage abstraction
    - eda_plot.py — plotting helpers used by backend endpoints
    - models.py — model serialization / inference helpers
    - agents/
      - analyze_graphs_agent.py — graph analysis utilities
      - state.py — agent state management
    - graph/
      - graph.py — graph data structure / helpers
- frontend/
  - app/
    - home.py — Streamlit app entrypoint
    - train.py — training UI and triggers
    - visualization.py — plotting pages / components
    - helper.py — frontend helpers (API client, utils)
    - config.py — app configuration (API base URL, ports)
- Dockerfile.backend, Dockerfile.frontend — container builds
- requirements.txt — pinned Python dependencies
- setup.py — package metadata (optional)
- README.md — this file

## Getting started (local, development)

Prerequisites
- Devcontainer: Ubuntu 24.04.2 LTS (this workspace)
- Python 3.10+ and pip
- Recommended: create and activate a virtual environment

Install dependencies
```sh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the backend (development)
```sh
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Run the frontend (development)
```sh
streamlit run frontend/app/home.py --server.port 8501
```

Open the frontend in the host browser from the devcontainer:
```sh
"$BROWSER" "http://localhost:8501"
```

Testing
```sh
pytest -q
```

Docker (quick)
- Build backend image:
```sh
docker build -f Dockerfile.backend -t ml_playground-backend .
```
- Build frontend image:
```sh
docker build -f Dockerfile.frontend -t ml_playground-frontend .
```

## Usage notes
- Backend API base URL: http://localhost:8000 by default
- Frontend expects the backend at the configured API base; update frontend/app/config.py if needed
- For production, configure persistent storage and a model registry instead of in-memory storage

## Roadmap
- Improve model registry and persistent storage
- Add user authentication for multi-user experimentation
- Add CI workflow and published Docker images

## Contributing
- Create a branch per feature: `git checkout -b feature/your-feature`
- Run tests and add unit tests for changes
- Open a PR with a clear description

## License
See the LICENSE file for details.

## Acknowledgments
- Contributors, inspirations and references used to build this project.

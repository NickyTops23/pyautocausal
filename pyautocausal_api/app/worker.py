from celery import Celery
from celery.utils.log import get_task_logger
import pandas as pd
from pathlib import Path
import shutil

# --- pyautocausal Import ---
# This should work if pyautocausal is installed via poetry from the local path
from pyautocausal.pipelines.example_graph import simple_graph
from pyautocausal.persistence.notebook_export import NotebookExporter
# If the above import fails, double-check:
# 1. You ran `poetry install` in the `pyautocausal_api` directory.
# 2. The path `../pyautocausal` in `pyautocausal_api/pyproject.toml` is correct.
# 3. Your `pyautocausal` library has a proper structure (e.g., `pyautocausal/pyautocausal/__init__.py`
#    and `pyautocausal/pyautocausal/pipelines/example_graph.py`).

# Initialize Celery
# The first argument 'tasks' is the conventional name for the main module of tasks.
# We'll refer to this celery_app instance when running the worker.
celery_app = Celery(
    'tasks', # This is the name of the current module for Celery's purposes
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0',
    # Set the include to ensure tasks are found if worker.py is not the main module
    # when celery worker is started. For simple cases like this, it might not be strictly
    # necessary if you start celery with `-A app.worker.celery_app`
    include=['app.worker']
)

celery_app.conf.update(
    task_track_started=True,
    # It's good practice to use JSON for serialization if possible
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)

logger = get_task_logger(__name__)

# Define base paths for our local "blob storage"
# These paths are relative to where the worker is run.
# Assuming the worker is started from `pyautocausal_api/` root.
# Using .resolve() makes them absolute paths.
BASE_INPUT_PATH = Path("./local_job_files/inputs").resolve()
BASE_OUTPUT_PATH = Path("./local_job_files/outputs").resolve()

@celery_app.task(bind=True, name='app.worker.run_graph_job') # Explicit naming is good practice
def run_graph_job(self, job_id: str, original_filename: str):
    """
    Celery task to load data, run the simple_graph, and save outputs.
    `bind=True` makes `self` (the task instance) available for updating state.
    """
    logger.info(f"[{job_id}] Celery task 'run_graph_job' started for file: {original_filename}")
    # Note: Celery automatically sets state to 'STARTED' if task_track_started=True
    # self.update_state(state='STARTED', meta={'message': 'Job picked up by worker.'})

    input_file_path = BASE_INPUT_PATH / job_id / original_filename
    output_job_path = BASE_OUTPUT_PATH / job_id

    try:
        # Ensure output directory exists
        output_job_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{job_id}] Output directory prepared: {output_job_path}")

        # 1. Load data (assuming CSV for this example)
        if not input_file_path.exists():
            logger.error(f"[{job_id}] Input file not found: {input_file_path}")
            raise FileNotFoundError(f"Input file not found at {input_file_path} for job {job_id}")

        current_meta = {'message': f'Loading data from {original_filename}.'}
        self.update_state(state='PROCESSING', meta=current_meta) # Custom state
        logger.info(f"[{job_id}] {current_meta['message']}")

        try:
            # For PoC, assuming CSV. Add more robust handling for production.
            data = pd.read_csv(input_file_path)
        except Exception as e:
            logger.error(f"[{job_id}] Error reading input CSV ({input_file_path}): {e}", exc_info=True)
            raise ValueError(f"Could not parse input file '{original_filename}' for job {job_id}: {e}")

        # 2. Initialize and run the graph
        current_meta = {'message': 'Initializing and fitting the causal graph.'}
        self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")

        # The simple_graph function from your library
        # It expects output_path where it will save its results.
        graph = simple_graph(output_path=output_job_path)

        graph.fit(df=data) # This is the potentially long-running operation
        logger.info(f"[{job_id}] Graph fitting complete. Results should be in {output_job_path}")

        # --- Add Notebook Export Logic Here ---
        current_meta = {'message': 'Exporting graph to Jupyter Notebook.'}
        self.update_state(state='PROCESSING', meta=current_meta)
        logger.info(f"[{job_id}] {current_meta['message']}")
        try:
            exporter = NotebookExporter(graph)
            notebook_filename = "causal_analysis_notebook.ipynb" # Or any name you prefer
            notebook_full_path = output_job_path / notebook_filename
            exporter.export_notebook(notebook_full_path)
            logger.info(f"[{job_id}] Notebook exported successfully to {notebook_full_path}")
        except Exception as e:
            logger.error(f"[{job_id}] Failed to export notebook: {e}", exc_info=True)
            # Decide if this failure should fail the whole job or just be a warning.
            # For now, we'll let it continue, but the main results are already saved.
            # If notebook export is critical, you might want to re-raise or handle differently.

        # 3. Task completed successfully
        # The result of the task will be the absolute path to the output directory.
        return str(output_job_path.resolve()) # Celery automatically sets state to SUCCESS

    except Exception as e:
        logger.error(f"[{job_id}] Task 'run_graph_job' failed: {e}", exc_info=True)
        # Celery automatically sets state to FAILURE and stores the exception
        # You can do cleanup here if needed, e.g., remove partial output_job_path
        # if output_job_path.exists():
        #     shutil.rmtree(output_job_path)
        #     logger.info(f"[{job_id}] Cleaned up output directory {output_job_path} due to failure.")
        raise # Re-raise the exception to ensure Celery processes it correctly.

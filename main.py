# main.py
import sys
from uni.crew import Uni, setup_vectorstore
from pathlib import Path

# ✅ You no longer need to call run_tumour_agent here.
# The tumour classification is now a CrewAI task (classify_tumour)
# handled by the tumour_classifier agent in the crew pipeline.

def run():
    """Run the CrewAI pipeline end-to-end: classify tumour, retrieve guidelines, recommend treatment, and generate report."""
    
    # Step 1: Setup vectorstore (guidelines)
    setup_vectorstore()

    # Step 2: Define initial inputs
    # The new pipeline can now accept the image path as input.
    # classify_tumour task will use this path.
    image_path = Path("C:/Users/arish/Downloads/Cirdan1/pngs_v2/20170608_135006.isyntax.png")

    # These inputs are available to the crew:
    #   - image_path: used by classify_tumour task
    #   - (no precomputed label/confidence anymore, classification is done inside crew)
    inputs = {
        "image_path": str(image_path)
    }

    # Step 3: Kick off the crew pipeline
    try:
        Uni().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"⚠ An error occurred while running the crew: {e}")

if __name__ == "__main__":
    run()

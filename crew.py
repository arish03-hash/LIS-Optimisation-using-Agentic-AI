from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from fpdf import FPDF
from typing import List
from dotenv import load_dotenv
import os
# from langchain.llms import Ollama
# from crewai_tools import PDFSearchTool
from crewai_tools import RagTool
from pathlib import Path
from uni.crew_1 import tumour_classifier_tool
  # ✅ this exists in your version


load_dotenv()

# -------------------------
# Load Models
# -------------------------
# hf_token = os.getenv("HF_TOKEN")
model1 = os.getenv("MODEL1")
model2 = os.getenv("MODEL2")
provider = os.getenv("PROVIDER")
# base_url = os.getenv("API_BASE")
print("DEBUG MODEL1:", model1)
print("DEBUG MODEL2:", model2)
# print("DEBUG PROVIDER:", provider)
# print("DEBUG API_BASE:", base_url)

# # Initialize RAG Tool once
# rag_tool = RagTool(
#     name="Breast Cancer Guidelines RAG",
#     description="Retrieve clinical guidelines for breast cancer",
#     vectorstore="chroma",
#     path="./chroma_db",
#     config=dict(
#         llm=Ollama(model=model1),
#         embedder=dict(
#             provider=provider,
#             config=dict(
#                 model=model2,
#                 base_url=os.getenv("API_BASE"),
#             ),
#         ),
#     )
# )

# -------------------------
# Helper: PDF Writer
# -------------------------
import re
def save_pdf(content: str, filename="treatment_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # remove unsupported Unicode chars
    safe_content = re.sub(r"[^\x00-\xFF]", "", str(content))

    for line in safe_content.split("\n"):
        pdf.multi_cell(0, 10, line)

    pdf.output(filename)


def setup_vectorstore():
    """Add PDFs to ChromaDB only if not already embedded."""
    pdf_path = Path("../../knowledge/guidelines")

    if Path("./chroma_db").exists():
        print("✅ Using existing ChromaDB index (skipping re-embedding).")
    else:
        print("📚 No ChromaDB index found. Embedding PDFs...")
        rag_tool.add(str(pdf_path), data_type="directory")
        print("✅ Finished embedding PDFs into ChromaDB.")
from pathlib import Path

# def setup_vectorstore():
#     """Add PDFs to ChromaDB only if not already embedded."""
#     # Resolve path to knowledge/guidelines relative to this file (crew.py or wherever setup_vectorstore lives)
#     base_dir = Path(__file__).resolve().parent
#     guidelines_path = (base_dir / "../../knowledge/guidelines").resolve()

#     if not guidelines_path.exists():
#         raise FileNotFoundError(f"❌ Guidelines path not found: {guidelines_path}")

#     print(f"📚 Loading guidelines from: {guidelines_path}")

#     # Check if DB already exists
#     if Path("./chroma_db").exists():
#         print("✅ Using existing ChromaDB index (skipping re-embedding).")
#     else:
#         print("📚 No ChromaDB index found. Embedding PDFs...")
#         rag_tool.add(str(guidelines_path), data_type="directory")
#         print("✅ Finished embedding PDFs into ChromaDB.")

# -------------------------
# Tools
# -------------------------


# @tool
# def tumour_classifier_tool(image_path: str) -> dict:
#     """
#     Classify a histopathology breast image as tumour or no tumour.
#     Returns a dict with label and confidence.
#     """
#     agent = TumourAgent("C:/Users/arish/Downloads/finetuned_biomedclip.pth")
#     result = agent.run(image_path)

#     # Ensure JSON-compatible output
#     return {
#         "label": result.get("label"),
#         "confidence": float(result.get("confidence", 0))
#     }

# class TumourPredictionTool:
#     """Wraps the TumourAgent to be used as a CrewAI Tool."""
#     name = "Tumour Classifier"
#     description = "Classify histopathological breast tissue images as tumour or no tumour."

#     def __init__(self, checkpoint_path="C:/Users/arish/Downloads/finetuned_biomedclip.pth"):
#         self.agent = TumourAgent(checkpoint_path)

#     def run(self, image_path: str):
#         """Runs the tumour agent on an image path."""
#         return self.agent._predictor.predict(image_path)
# Create RAG tool with custom LLM and embedder
rag_tool = RagTool(
    config=dict(
        llm=dict(
            provider=provider, 
            config=dict(model=model1)
        ),
        embedder=dict(
            provider=provider, 
            config=dict(model=model2)
        ),
    )
)

# Add your directory of PDFs
# rag_tool.add("../../knowledge/guidelines", data_type="directory")

# pdf_search_tool = PDFSearchTool(
#     pdf=[
#         "../../knowledge/guidelines/ECCN Management.pdf",
#         "../../knowledge/guidelines/Guidelines for management.pdf",
#         "../../knowledge/guidelines/NCCN Guidelines V3.2021 pdf.pdf",
#         "../../knowledge/guidelines/NCCN.pdf"
#     ],
#     config=dict(
#         llm=dict(provider=provider, config=dict(model=model1)),
#         embedder=dict(provider=provider, config=dict(model=model2)),
#     ),
# )

# Shared LLM (fixed)
# my_llm = LLM(
#     provider=provider,
#     config=dict(model=model1),
# )
my_llm=LLM(model="ollama/llama3.1", base_url="http://localhost:11434")

class LangchainRagTool(RagTool):
    def __init__(self, llm, embedder, **kwargs):
        # convert LangChain LLM into dict
        llm_dict = {
            "provider": "ollama",
            "config": {"model": llm.model}
        }
        embedder_dict = embedder if isinstance(embedder, dict) else {
            "provider": "huggingface",
            "config": {"model": embedder}
        }
        super().__init__(config=dict(llm=llm_dict, embedder=embedder_dict), **kwargs)

# Usage
# my_llm = Ollama(model=model1)
# rag_tool = LangchainRagTool(
#     llm=my_llm,
#     embedder=model2,
#     name="Breast Cancer Guidelines RAG",
#     description="Retrieve clinical guidelines for breast cancer",
#     vectorstore="chroma",
#     path="./chroma_db",
# )



@CrewBase
class Uni:
    """Uni Crew for Breast Cancer Treatment Planning"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # -------------------------
    # Agents
    # -------------------------

    @agent
    def tumour_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config["tumour_classifier"],
            tools=[tumour_classifier_tool],
            llm=my_llm,
        )

    @agent
    def diagnostic_mapper(self) -> Agent:
        return Agent(
            config=self.agents_config["diagnostic_mapper"],
            llm=my_llm,
        )


    @agent
    def guideline_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config["guideline_retriever"],
            tools=[rag_tool],
            llm=my_llm,
        )

    @agent
    def treatment_recommender(self) -> Agent:
        return Agent(
            config=self.agents_config["treatment_recommender"],
            llm=my_llm,
        )

    @agent
    def validator(self) -> Agent:
        return Agent(
            config=self.agents_config["validator"],
            llm=my_llm,
        )

    @agent
    def patient_explainer(self) -> Agent:
        return Agent(
            config=self.agents_config["patient_explainer"],
            llm=my_llm,
        )

    # @agent
    # def main_agent(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config["main_agent"],
    #         llm=my_llm,
    #     )

    @agent
    def patient_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["patient_generator"],
            llm=my_llm
        )

    @agent
    def doctor_generator(self) -> Agent:
        return Agent(
            config=self.agents_config["doctor_generator"], 
            llm=my_llm
        )

    @agent
    def report_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["report_writer"], 
            llm=my_llm
        )

    @agent
    def manager(self) -> Agent:
        return Agent(
            config = self.agents_config["manager"],
            llm=my_llm,
        )


    # -------------------------
    # Tasks
    # -------------------------

    @task
    def classify_tumour(self) -> Task:
        return Task(
            config=self.tasks_config["classify_tumour"],
        )

    @task
    def map_diagnosis(self) -> Task:
        return Task(
            config=self.tasks_config["map_diagnosis"],
        )

    @task
    def retrieve_guidelines(self) -> Task:
        return Task(
            config=self.tasks_config["retrieve_guidelines"],
        )

    @task
    def recommend_treatment(self) -> Task:
        return Task(
            config=self.tasks_config["recommend_treatment"],
        )

    @task
    def validate_plan(self) -> Task:
        return Task(
            config=self.tasks_config["validate_plan"],
        )

    @task
    def explain_to_patient(self) -> Task:
        return Task(
            config=self.tasks_config["explain_to_patient"],
        )

    @task
    def generate_patient(self) -> Task:
        return Task(
            config=self.tasks_config["generate_patient"]
        )

    @task
    def generate_doctor(self) -> Task:
        return Task(
            config=self.tasks_config["generate_doctor"]
        )

    @task
    def write_report(self) -> Task:
        return Task(
            config=self.tasks_config["write_report"],
            callback=lambda output: save_pdf(str(output.raw), "treatment_report.pdf"),
            output_file="treatment_report.pdf"
        )

    # @task
    # def orchestrate_plan(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["orchestrate_plan"],
    #         output_file="treatment_plan.md",
    #     )

    

    # -------------------------
    # Crew
    # -------------------------

    @crew
    def crew(self) -> Crew:
        """Creates the Uni crew"""
        manager_agent = self.manager()
        return Crew(
            agents=[a for a in self.agents if a is not manager_agent],
            tasks=self.tasks,
            process=Process.sequential,
            manager_agent=manager_agent,
            verbose=True,
        )




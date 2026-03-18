import sys

# Extract --model from sys.argv
model_name = None
for i, arg in enumerate(sys.argv):
    if arg == "--model" and i + 1 < len(sys.argv):
        model_name = sys.argv[i + 1]
        sys.argv = sys.argv[:i] + sys.argv[i + 2:]
        break

if model_name is None:
    print("ERROR: --model argument is required")
    print("Usage: python pipeline_rag.py --model <model_name>")
    sys.exit(1)

from ontomap.ontology_matchers import MatcherCatalog
from ontomap.pipeline.om_pipeline import OMPipelines

if model_name not in MatcherCatalog["rag"]:
    print(f"Unknown model '{model_name}'. Available: {list(MatcherCatalog['rag'].keys())}")
    sys.exit(1)

config = {
    "approach": "rag",
    "encoder": "rag",
    "use-all-models": False,
    "models-to-consider": [model_name],
    "use-all-encoders": False,
    "approach-encoders-to-consider": ["label", "label-children", "label-parent"],
    "do-evaluation": False,
    "load-from-json": True,
    "root_dir": "datasets",
    "device": "cuda",
    "batch-size": 32,
    "nshots": 0,
    "outputs-dir": "outputs",
    "llm_confidence_th": 0.7,
}

# generation first
pipeline = OMPipelines(**config)
pipeline()

# evaluation afterwards
config["do-evaluation"] = True
pipeline_evaluation = OMPipelines(**config)
pipeline_evaluation()

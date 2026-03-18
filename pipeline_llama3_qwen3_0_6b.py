from ontomap.pipeline.om_pipeline import OMPipelines

config = {
    "approach": "rag",
    "encoder": "rag",
    "use-all-models": False,
    "models-to-consider": [
        "LLaMA3Qwen3RAG",
    ],
    "use-all-encoders": False,
    "approach-encoders-to-consider": ["label", "label-children", "label-parent"],
    "do-evaluation": False,
    "load-from-json": True,
    "batch_size": 16,
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

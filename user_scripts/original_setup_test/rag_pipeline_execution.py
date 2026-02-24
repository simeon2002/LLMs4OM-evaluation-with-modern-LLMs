from ontomap import OMPipelines

# All open-source RAG models (excluding ChatGPT/GPT-4)
rag_models = [
    'LLaMA7BAdaRAG',
    'LLaMA7BBertRAG',
    'MistralAdaRAG',
    'MistralBertRAG',
    'FalconAdaRAG',
    'FalconBertRAG',
    'VicunaAdaRAG',
    'VicunaBertRAG',
    'MPTAdaRAG',
    'MPTBertRAG',
    'MambaLLMAdaRAG',
    'MambaLLMBertRAG',
]

# Bio-ML datasets
bio_ml_datasets = [
    'ncit-doid',
    'omim-ordo',
    'snomed-fma',
    'snomed-ncit-neoplas',
    'snomed-ncit-pharm',
]

# Base configuration
base_config = {
    'approach': 'rag',
    'encoder': 'rag',
    'use-all-encoders': False,
    'approach-encoders-to-consider': ['label'],  # C representation
    'use-all-models': False,
    'load-from-json': False,
    'device': 'cuda',
    'do-evaluation': True,
    'batch-size': 16,
    'llm_confidence_th': 0.7,
    'ir_score_threshold': 0.9
}

# Run for each model on each dataset
for model in rag_models:
    for dataset in bio_ml_datasets:
        print(f"\n{'='*60}")
        print(f"Running {model} on {dataset}")
        print('='*60)
        
        args = {
            **base_config,
            'models-to-consider': [model],
            'outputs-dir': f'output-rag-bioml-{model}-{dataset}',
            'task': dataset  # Specify the bio-ml task
        }
        
        try:
            runner = OMPipelines(**args)
            runner()
            print(f"✓ {model} on {dataset} completed successfully")
        except Exception as e:
            print(f"✗ {model} on {dataset} failed: {str(e)}")

print("\n" + "="*60)
print("All experiments completed!")
print("="*60)
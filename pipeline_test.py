from ontomap import OMPipelines
from ontomap.ontology import ontology_matching
import traceback

rag_models = [
    'MistralBertRAG',
]

base_config = {
    'approach': 'rag',
    'encoder': 'rag',
    'use-all-encoders': False,
    'approach-encoders-to-consider': ['label'],
    'use-all-models': False,
    'load-from-json': True,
    'device': 'cuda',
    'do-evaluation': False,
    'batch-size': 128,
    'nshots': 0,
    'llm_confidence_th': 0.7,
    'ir_score_threshold': 0.9,
}

for model in rag_models:
    print(f"\n{'='*60}")
    print(f"TEST: Running {model} on ncit-doid.disease only")
    print('='*60)
    
    args = {
        **base_config,
        'models-to-consider': [model],
        'outputs-dir': f'output-test-{model}',
    }
    
    try:
        # Backup original
        original = dict(ontology_matching)
        
        # Filter to ONLY ncit-doid.disease
        ontology_matching.clear()
        # Get only the first dataset (ncit-doid.disease)
        ontology_matching['bio-ml'] = [original['bio-ml'][0]]
        
        runner = OMPipelines(**args)
        runner()
        
        # Restore
        ontology_matching.clear()
        ontology_matching.update(original)
        
        print(f"✓ Test completed successfully!")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()

print("\n" + "="*60)
print("Quick test done!")
print("="*60)
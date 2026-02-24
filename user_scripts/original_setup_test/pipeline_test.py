from ontomap import OMPipelines
import traceback

# Test with just 1 model on 1 dataset first
rag_models = [
    'MistralBertRAG',
]

# Base configuration
base_config = {
    'approach': 'rag',
    'encoder': 'rag',
    'use-all-encoders': False,
    'approach-encoders-to-consider': ['label'],
    'use-all-models': False,
    'load-from-json': False,
    'device': 'cuda',
    'do-evaluation': True,
    'batch-size': 16,
    'nshots': 0,  # ← MAKE SURE THIS IS HERE
    'llm_confidence_th': 0.7,
    'ir_score_threshold': 0.9,
    'tracks-to-consider': ['bio-ml'],
    'tasks-to-consider': ['ncit-doid']
}

# Run test
for model in rag_models:
    print(f"\n{'='*60}")
    print(f"TEST: Running {model} on ncit-doid only")
    print('='*60)
    
    args = {
        **base_config,
        'models-to-consider': [model],
        'outputs-dir': f'output-test-{model}'
    }
    
    try:
        runner = OMPipelines(**args)
        runner()
        print(f"✓ Test completed successfully!")
    except Exception as e:
        print(f"✗ Test failed: {str(e)}")
        traceback.print_exc()

print("\n" + "="*60)
print("Quick test done!")
print("="*60)
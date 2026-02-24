from ontomap.pipeline.om_pipeline import OMPipelines
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--models', type=str, nargs='+', required=True, help='One or more models to evaluate')
# parser.add_argument('--encoders', type=str, nargs='+', default=["label", "label-children", "label-parent"], help='One or more encoder types')
# args = parser.parse_args()

# print(args.models)
# print(args.encoders)

'''
        "MistralBertRAG",
        "FalconBertRAG",
        "LLaMA7BBertRAG",
        "VicunaBertRAG",
        # "MPTBertRAG", #TODO: ASK... THIS ONE CAN'T BE LOADED, BECAUSE REPO DOESN'T EXIST ANYMORE, WHAT TO DO? 
        "MambaBertRAG",
'''

config = {
    "approach": "rag",  # or whatever approach you want
    "encoder": 'rag',
    "use-all-models": False,
    "models-to-consider": [
        # "MistralBertRAG",
        "FalconBertRAG",
        # "LLaMA7BBertRAG",
        # "VicunaBertRAG",
        # # "MPTBertRAG", #TODO: ASK... THIS ONE CAN'T BE LOADED, BECAUSE REPO DOESN'T EXIST ANYMORE, WHAT TO DO? 
        # "MambaBertRAG",
    ],
    "use-all-encoders": False,
    "approach-encoders-to-consider": ["label", "label-children", "label-parent"],
    "do-evaluation": False,  # Set to True if you want to run evaluation after generation
    "load-from-json": True,
    "batch_size": 16,
    "root_dir": "datasets",
    'device': 'cuda',
    'batch-size': 32,
    'nshots': 0,
    'outputs-dir': 'outputs',
    'llm_confidence_th': 0.7,
}

# generation first
# import ipdb; ipdb.set_trace()
pipeline = OMPipelines(**config)
pipeline()  

# evaluation afterwards
config["do-evaluation"] = True 
pipeline_evaluation = OMPipelines(**config)
pipeline_evaluation()
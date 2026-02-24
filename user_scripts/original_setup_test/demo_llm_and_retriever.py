from ontomap.ontology import MouseHumanOMDataset
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder import IRILabelInRAGEncoder
from ontomap.ontology_matchers import MistralLLMBertRAG
from ontomap.postprocess import process

# Setting configurations for experimenting 'rag' on GPU with batch size of 16
config = BaseConfig(approach='rag').get_args(device='cuda', batch_size=16)
# set dataset directory
config.root_dir = "datasets"
# parse task source, target, and reference ontology
ontology = MouseHumanOMDataset().collect(root_dir=config.root_dir)

# init encoder (concept-representation)
encoded_inputs = IRILabelInRAGEncoder()(**ontology)

# init Mistral-7B + BERT
model = MistralLLMBertRAG(**config.MistralBertRAG)
# generate results
predicts = model.generate(input_data=encoded_inputs)

# post-processing
predicts, _ = process.postprocess_hybrid(predicts=predicts,
                                         llm_confidence_th=0.7,
                                         ir_score_threshold=0.9)
# evaluation
results = evaluator(track='anatomy',
                    predicts=predicts,
                    references=ontology["reference"])
print(results)
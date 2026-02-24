from ontomap.ontology import MouseHumanOMDataset
from ontomap.base import BaseConfig
from ontomap.evaluation.evaluator import evaluator
from ontomap.encoder.lightweight import IRILabelInLightweightEncoder
from ontomap.ontology_matchers.retrieval.models import BERTRetrieval
from ontomap.postprocess import process

# 1) Config: retrieval on CPU
config = BaseConfig(approach="retrieval").get_args(device="cpu")
config.root_dir = "datasets"  # relative to repo root

# 2) Load ontology task (mouse-human anatomy)
ontology = MouseHumanOMDataset().collect(root_dir=config.root_dir)

print(ontology)

# 3) Encode concepts (C representation)
# Instantiate the encoder
encoder = IRILabelInLightweightEncoder()
# Call the encoder with unpacked ontology dictionary
encoded_inputs = encoder(**ontology)

# 4) Run BERT-based retrieval
model = BERTRetrieval(**config.BERTRetrieval)
predicts = model.generate(input_data=encoded_inputs)

# 5) Post-process + evaluate
predicts = process.eval_preprocess_ir_outputs(predicts=predicts)
results = evaluator(
    track="anatomy",
    predicts=predicts,
    references=ontology["reference"],
)

print(results)



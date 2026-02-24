# -*- coding: utf-8 -*-
import os


from ontomap.base import BaseConfig
from ontomap.ontology import ontology_matching
from ontomap.utils import io

root_dataset_dir = ".."

if __name__ == "__main__":
    config = BaseConfig().get_args()
    for ontology, oms in ontology_matching.items():
        print(f"working on {ontology} track OM pairs!")
        for om in oms:
            print(f"\t {om.ontology_name} pairs is processing!")
            dataset = om().collect(root_dir=config.root_dir) # same collect funciton as used in om_pipelines whenever the load from json is set to false, which is the default value. So it will always parse the ontologies and generate the dataset.
            output_path = os.path.join(config.root_dir, om.working_dir, "om.json")
            io.write_json(output_path=output_path, json_data=dataset)

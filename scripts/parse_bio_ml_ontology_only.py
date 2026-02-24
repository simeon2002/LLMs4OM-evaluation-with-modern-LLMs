from ontomap.base import BaseConfig
from ontomap.ontology import ontology_matching
from ontomap.utils import io
import os

config = BaseConfig().get_args()

# Only parse bio-ml track
print("Parsing bio-ml datasets only...\n")
for om in ontology_matching['bio-ml']:
    print(f"{'='*60}")
    print(f"Processing {om.ontology_name}...")
    print('='*60)
    
    output_path = os.path.join(config.root_dir, om.working_dir, "om.json")
    
    # Check if om.json already exists
    if os.path.exists(output_path):
        print(f"✓ om.json already exists at:")
        print(f"  {output_path}")
        print(f"  Skipping parsing...\n")
    else:
        print(f"om.json not found. Parsing ontology...")
        try:
            dataset = om().collect(root_dir=config.root_dir)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            io.write_json(output_path=output_path, json_data=dataset)
            print(f"✓ Successfully saved to:")
            print(f"  {output_path}\n")
        except Exception as e:
            print(f"✗ Failed to parse: {str(e)}")
            import traceback
            traceback.print_exc()
            print()

print("Bio-ml datasets processing complete!")
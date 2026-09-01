import json
import time
import os
from pydantic import BaseModel, Field
from typing import List
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

class PropertyItem(BaseModel):
    property_family: str = Field(description="The family of the property, e.g., 'capability', 'action', 'state', 'function'")
    frame_type: str = Field(description="The frame type, e.g., 'can_verb', 'has_property', 'does_action'")
    predicate_lemma: str = Field(description="The base verb, noun, or adjective, e.g., 'fly', 'photosynthesize', 'cut'")
    positive_realization: str = Field(description="Positive phrasing for singular subjects, e.g., 'can fly', 'is used to cut'")
    negative_realization: str = Field(description="Negative phrasing for singular subjects, e.g., 'cannot fly', 'is not used to cut'")
    positive_plural_realization: str = Field(description="Positive phrasing for plural subjects, e.g., 'can fly', 'are used to cut'")
    negative_plural_realization: str = Field(description="Negative phrasing for plural subjects, e.g., 'cannot fly', 'are not used to cut'")
    allowed_subject_types: List[str] = Field(description="List of subclasses/subjects this applies to, e.g., ['bird', 'insect']. Ensure these are written in singular form and are subclasses of the root concept.")
    domain: str = Field(description="The domain name this property belongs to. Ensure this to be singular and exactly matches the 'domain' field in the input.")

class DomainPropertiesList(BaseModel):
    properties: List[PropertyItem]


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = 'gemini-flash-latest'
 

def get_properties_for_domain(domain_name: str) -> dict:
    """Calls Gemini to generate properties for a specific domain using Structured Output."""
    
    prompt = f"""
    You are an expert ontologist. 
    Generate a list of distinct and common properties for the domain '{domain_name}'.
    Ensure every item accurately reflects the structure and details of the requested schema.
    For the 'domain' field, always use exactly '{domain_name}'.
    """
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DomainPropertiesList,
                temperature=0.2, 
            ),
        )
        # Parse the JSON string returned by the model
        return response.parsed.model_dump()
    
    except Exception as e:
        print(f"Error processing domain '{domain_name}': {e}")
        return None

def main():

    input_filename = '../KnowledgeGraph/data_sources/domains.json'
    output_filename = '../KnowledgeGraph/data_sources/domain_properties.json'
    

    # if the input file doesn't exist
    if not os.path.exists(input_filename):
        print(f"Creating sample {input_filename} for testing...")
        sample_domains =[
            {"domain_name": "Animals"},
            {"domain_name": "Tools and Equipment"}
        ]
        with open(input_filename, 'w') as f:
            json.dump(sample_domains, f, indent=4)


    with open(input_filename, 'r') as file:
        domains_data = json.load(file)

    all_properties_output =[]

    print(f"Loaded {len(domains_data)} domains. Starting generation...\n")
    
    # have index
    for idx, item in enumerate(domains_data):
        # if idx>4:
        #     print("Reached 5 domains, stopping for now...")
        #     break

        domain_name = item.get("domain_name")
        if not domain_name:
            continue
            
        print(f"Fetching properties for domain: {domain_name}...")
        
        result = get_properties_for_domain(domain_name)
        
        if result and "properties" in result:
            all_properties_output.extend(result["properties"])
            print(f"Successfully generated {len(result['properties'])} properties for {domain_name}.")


    with open(output_filename, 'w') as out_file:
        json.dump(all_properties_output, out_file, indent=4)
        
    print(f"\nDone! All properties saved to {output_filename}")

if __name__ == "__main__":
    main()
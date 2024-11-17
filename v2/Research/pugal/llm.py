from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time
import json


# LangChain setup with a prompt template for comparing application details
validation_schema = {
    'application_number': {'datatype': 'Number', 'length': 6},
    'name': {'datatype': 'String'},
    'parent_name': {'datatype': 'String'},
    'address': {'datatype': 'String'},
    'taluk': {'datatype': 'String'},
    'pin_code': {'datatype': 'Number', 'length': 6},
    'district': {'datatype': 'String'},
    'state': {'datatype': 'String'},
    'DOB': {'datatype': 'String'},
    'gender': {'allowed_values': ['Male', 'Female', 'Other']},
    'nationality': {'datatype': 'String'},
    'nativity': {'datatype': 'String'},
    'aadhar_number': {'datatype': 'Number', 'length': 12},
    'annual_income': {'datatype': 'Number', 'min_value': 0},
    'civic_status': {'datatype': 'String'},
    'mother_tongue': {'datatype': 'String'},
    'first_graduate': {'datatype': 'Boolean'},
    'school_category': {'allowed_values': ['govt', 'govt-aided', 'CBSC', 'ICSC']},
    'school_name': {'datatype': 'String'},
    'permanent_register_number': {'datatype': 'Number'},
    'HSC_roll_no': {'datatype': 'Number'},
    'medium_of_instruction': {'datatype': 'String'},
    'HSC_mark': {'datatype': 'Number', 'min_value': 0, 'max_value': 600},
    'SSLC_mark': {'datatype': 'Number', 'min_value': 0, 'max_value': 500},
    'community_certificate_number': {'datatype': 'String'},
    'applied_for_neet': {'datatype': 'Boolean'},
    'applied_for_jee': {'datatype': 'Boolean'}
}

template = """
Your task is to extract the details of the applicant from the provided details:
Application Details: {user}

Please return the JSON formatted response:
- Output only the JSON response
- Don't include any additional text in the response

Important: Make it as accurate as possible.

Exact Output Schema (Strictly follow):
```json {validation_schema}```
"""

# Initialize Ollama with the Llama3.2 model
model = OllamaLLM(model="llama3.2")

def compare_application_and_documents(extracted_text, validation_schema):
    """Use Llama3.2 model to compare the extracted text and reference details."""
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Instead of LLMChain, use RunnableSequence and the invoke method
    chain = prompt | model  # Chaining the prompt and model

    # Prepare inputs
    inputs = {
        "user": extracted_text,
        'validation_schema': validation_schema
    }
    
    start_time = time.time()
    
    # Use invoke() to get the result
    result = chain.invoke(inputs)  # Changed run() to invoke()
    
    # Record the end time
    end_time = time.time()
    
    # Calculate the elapsed time
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.6f} seconds")
    
    return json.loads(result)

def validate_data(data, schema):
    """Validate extracted data against the schema."""
    errors = []
    for key, rules in schema.items():
        value = data.get(key)

        # Validate datatype
        if 'datatype' in rules:
            expected_type = rules['datatype']
            if expected_type == 'Number' and not isinstance(value, (int, float)):
                errors.append(f"{key} must be a number.")
            elif expected_type == 'String' and not isinstance(value, str):
                errors.append(f"{key} must be a string.")
            elif expected_type == 'Boolean' and not isinstance(value, bool):
                errors.append(f"{key} must be a boolean.")

        # Validate length
        if 'length' in rules and isinstance(value, str) and len(value) != rules['length']:
            errors.append(f"{key} must have a length of {rules['length']}.")

        # Validate min and max value
        if 'min_value' in rules and isinstance(value, (int, float)) and value < rules['min_value']:
            errors.append(f"{key} must be at least {rules['min_value']}.")
        if 'max_value' in rules and isinstance(value, (int, float)) and value > rules['max_value']:
            errors.append(f"{key} must not exceed {rules['max_value']}.")

        # Validate allowed values
        if 'allowed_values' in rules and value not in rules['allowed_values']:
            errors.append(f"{key} must be one of {rules['allowed_values']}.")

    return errors

def main():
    # Example extracted text (from OCR or document)
    extracted_text = """
    Application Number: 305994
    Name: RAJESH S
    Parent Name: SARAVANAN S
    Address: 107/D4, SOLARAJAPURAM STREET
    Taluk: RAJAPALAYAM
    Pin Code: 626117
    District: Virudhunagar
    State: Tamil Nadu
    DOB: 15-04-2005
    Gender: Male
    Nationality: Indian
    Nativity: Tamil Nadu
    Aadhar Number: 295206496531
    Annual Income: 96000
    Civic Status: Municipality
    Mother Tongue: Tamil
    First Graduate: true
    School Category: govt-aided
    School Name: XYZ School
    Permanent Register Number: 2111119945
    HSC Roll No: 5119714
    Medium of Instruction: Tamil
    HSC Mark: 513
    SSLC Mark: 424
    Community Certificate Number: FFDB678C6A687B86
    Applied for NEET: false
    Applied for JEE: false
    """
    print("Extracting applicant details...")
    result = compare_application_and_documents(extracted_text, validation_schema)
    print("Validation Results:")
    print(result)
    errors = validate_data(result, validation_schema)
    
    if errors:
        print("Errors found in the extracted data:")
        for error in errors:
            print(error)
    else:
        print("All data is valid!")

# Run the main function
if __name__ == "__main__":
    main()
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
Your task is to extract the applicant's details from the provided input.  
Input Details: {user}  

Return only the JSON object adhering strictly to the schema below.  
**Guidelines:**  
1. Use the schema to validate and structure the extracted information.  
2. Ensure all required fields are included and formatted correctly based on the schema.  
3. Do not include the schema itself or any explanation in the response.  
4. Respond with **only** the JSON object and no additional text.

Output format(Strictly follow):
{output_format}
Note: Do not add schema gain

Schema:
```json {validation_schema}```

Extract the data and return a validated JSON object based strictly on the schema.  
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
        'validation_schema': validation_schema,
        'output_format': '{ key: value }'
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
GOVERNMENT OF TAMIL NADU
ak pares 3, DIRECTORATE OF TECHNICAL EDUCATION
ee Sy TAMIL NADU ENGINEERING ADMISSION - 2022
Application Number: 305994
Personal Information
Name: RAJESH S Name of the Parent/Guardian: SARAVANAN S
Communication Address: 107/D4, SOLARAJAPURAM STREET, Permanent Address: 107/D4, SOLARAJAPURAM STREET,
AAVARAMPATTI, AAVARAMPATTI,
RAJAPALAYAM - 626117 RAJAPALAYAM - 626117
State: Tamil nadu District; Virudhunagar
Taluk: Rajapalayam Communication address pincode: 626117
Native District: Virudhunagar Civic status of Native Place: Municipality
Date of Birth (DD-MM-YYYY): 15-04-2005 Gender: Male
Mother Tongue: Tamil Nationality: Indian
Nativity: Tamil nadu Religion: Hindu
Name of the Community: BC Name of the Caste: Senaithalaivar, Senaikudiyar and Illaivaniar
Aadhar Number (optional): 295206496531
Special Reservation Information
Whether you are a candidate under quota for Eminent Sports person as per Ex-Servicemen (Only Army/Navy/ Air force services are Eligible): No
annexure-ll, item No.22 of information brochure?:
No
Differently Abled Person: No Differently Abled Type: -
TFC Center for certificate verification:
PAC Ramasamy Raja's Polytechnic College,Rajapalayam - 626 108
Scholarship Information
Parent Occupation: Self Employed Annual Income: 96000
Are you a First Graduate?: Yes Post Matric Scholarship (SC/SCA/ST/Converted Christians): No
School of Study Information
Category of School: Govt. Aided Civic status of school location (+2): Municipality
Have you studied VIII to XII in Tamil Nadu?: Yes Have you studied from VI to VIII in private school under RTE and IX to XII in
Government School?:
No
Have you studied VI to XII in Government school?: No
Class Year of Passing Name of the schoo! District State Block Category of
Govt.School
VI Std. 2016 N.a Annapparaja Memorial H S S Ra- Virudhunagar Tamil nadu Rajapalayam -
japalayam
japalayam
japalayam
japalayam
japalayam
japalayam
japalayam
Academic Information
Qualifying Examination: HSC Name of the Board of Examination:
Tamil nadu Board of Higher Secondary Education
Permanent register number: 2111119945 HSC Roll number: 5119714
Qualified Year: 2022 HSC Group: HSC Academic
Group Code: Physics/ Chemistry/ Maths/ Biology Medium of Instruction: Tamil
HSC maximum (total) marks: 600 HSC obtained marks: 513
SSLC maximum (total) marks: 500 SSLC obtained marks: 424
Have you applied for NEET ?: No Have you applied for JEE ?: No
Educational Management Information System(EMIS) Number: Community certificate number: FFDB678C6A687B86
332606127 7500257


Extracted Keywords:
 [('japalayam\njapalayam\njapalayam\njapalayam\njapalayam\njapalayam\njapalayam\nacademic information\nqualifying examination', 105.64285714285714), ('technical education\nee sy tamil nadu engineering admission', 54.666666666666664), ('higher secondary education\npermanent register number', 32.0), ('physics/ chemistry/ maths/ biology medium', 25.0), ('ra- virudhunagar tamil nadu rajapalayam', 18.4), ('hsc academic\ngroup code', 17.166666666666664), ('educational management information system', 16.142857142857142), ('army/navy/ air force services', 16.0), ('tamil nadu\nak pares 3', 15.666666666666666), ('626 108\nscholarship information\nparent occupation', 15.642857142857142), ('district state block category', 13.0), ('rajapalayam communication address pincode', 12.066666666666666), ('tamil nadu district', 10.666666666666666), ('tamil nadu religion', 10.666666666666666), ('study information\ncategory', 10.642857142857142), ('295206496531\nspecial reservation information', 10.142857142857142), ('tamil nadu board', 9.666666666666666), ('post matric scholarship', 9.5), ('virudhunagar civic status', 9.333333333333334), ('male\nmother tongue', 9.0), ('illaivaniar\naadhar number', 9.0), ('eminent sports person', 9.0), ('differently abled person', 9.0), ('differently abled type', 9.0), ('pac ramasamy raja', 9.0), ('employed annual income', 9.0), ('aided civic status', 9.0), ('tamil\nhsc maximum', 8.833333333333332), ('2111119945 hsc roll number', 8.666666666666666), ('tamil nadu', 7.666666666666666), ('600 hsc obtained marks', 7.666666666666666), ('500 sslc obtained marks', 7.5), ('community certificate number', 7.5), ('school\nvi std', 6.857142857142858), ('permanent address', 6.666666666666666), ('305994\npersonal information', 6.142857142857143), ('information brochure', 6.142857142857143), ('examination', 6.0), ('communication address', 5.666666666666666), ('tamil nationality', 5.666666666666666), ('2022 hsc group', 5.666666666666666), ('virudhunagar\ntaluk', 5.333333333333334), ('2022\napplication number', 5.0), ('626117\nnative district', 5.0), ('513\nsslc maximum', 5.0), ('certificate verification', 4.5), ('solarajapuram street', 4.0), ('native place', 4.0), ('15-04-2005 gender', 4.0), ('indian\nnativity', 4.0), ('tfc center', 4.0), ('polytechnic college', 4.0), ('sc/sca/st/converted christians', 4.0), ('class year', 4.0), ('annapparaja memorial', 4.0), ('5119714\nqualified year', 4.0), ('school location', 3.857142857142857), ('private school', 3.857142857142857), ('studied vi', 3.666666666666667), ('government school', 3.5238095238095237), ('municipality\ndate', 3.5), ('studied viii', 3.166666666666667), ('number', 3.0), ('hsc', 2.6666666666666665), ('626117\nstate', 2.5), ('rajapalayam', 2.4), ('626117 rajapalayam', 2.4), ('community', 2.0), ('vi', 2.0), ('board', 2.0), ('marks', 2.0), ('school', 1.8571428571428572), ('government', 1.6666666666666667), ('studied', 1.6666666666666667), ('municipality', 1.5), ('viii', 1.5), ('pes', 1.0), ('directorate', 1.0), ('rajesh', 1.0), ('parent/guardian', 1.0), ('saravanan', 1.0), ('107/d4', 1.0), ('aavarampatti', 1.0), ('birth', 1.0), ('dd-mm-yyyy', 1.0), ('hindu', 1.0), ('bc', 1.0), ('caste', 1.0), ('senaithalaivar', 1.0), ('senaikudiyar', 1.0), ('optional', 1.0), ('candidate', 1.0), ('quota', 1.0), ('ex-servicemen', 1.0), ('eligible', 1.0), ('annexure-ll', 1.0), ('item', 1.0), ('graduate', 1.0), ('govt', 1.0), ('xii', 1.0), ('rte', 1.0), ('ix', 1.0), ('passing', 1.0), ('schoo', 1.0), ('instruction', 1.0), ('total', 1.0), ('applied', 1.0), ('neet', 1.0), ('jee', 1.0), ('emis', 1.0), ('ffdb678c6a687b86\n332606127 7500257', 1.0), ('22', 0), ('96000', 0), ('+2', 0), ('2016', 0), ('424', 0)]
Extracted Text for Debugging:
 PES? GOVERNMENT OF TAMIL NADU
ak pares 3, DIRECTORATE OF TECHNICAL EDUCATION
ee Sy TAMIL NADU ENGINEERING ADMISSION - 2022
Application Number: 305994
Personal Information
RAJESH S Name of the Parent/Guardian: SARAVANAN S
Communication Address: 107/D4, SOLARAJAPURAM STREET, Permanent Address: 107/D4, SOLARAJAPURAM STREET,
AAVARAMPATTI, AAVARAMPATTI,
RAJAPALAYAM - 626117 RAJAPALAYAM - 626117
State: Tamil nadu District; Virudhunagar
Taluk: Rajapalayam Communication address pincode: 626117
Native District: Virudhunagar Civic status of Native Place: Municipality
Date of Birth (DD-MM-YYYY): 15-04-2005 Gender: Male
Mother Tongue: Tamil Nationality: Indian
Nativity: Tamil nadu Religion: Hindu
Name of the Community: BC Name of the Caste: Senaithalaivar, Senaikudiyar and Illaivaniar
Aadhar Number (optional): 295206496531
Special Reservation Information
Whether you are a candidate under quota for Eminent Sports person as per Ex-Servicemen (Only Army/Navy/ Air force services are Eligible): No
annexure-ll, item No.22 of information brochure?:
No
Differently Abled Person: No Differently Abled Type: -
TFC Center for certificate verification:
PAC Ramasamy Raja's Polytechnic College,Rajapalayam - 626 108
Scholarship Information
Parent Occupation: Self Employed Annual Income: 96000
Are you a First Graduate?: Yes Post Matric Scholarship (SC/SCA/ST/Converted Christians): No
School of Study Information
Category of School: Govt. Aided Civic status of school location (+2): Municipality
Have you studied VIII to XII in Tamil Nadu?: Yes Have you studied from VI to VIII in private school under RTE and IX to XII in
Government School?:
No
Have you studied VI to XII in Government school?: No
Class Year of Passing Name of the schoo! District State Block Category of
Govt.School
VI Std. 2016 N.a Annapparaja Memorial H S S Ra- Virudhunagar Tamil nadu Rajapalayam -
japalayam
japalayam
japalayam
japalayam
japalayam
japalayam
japalayam
Academic Information
Qualifying Examination: HSC Name of the Board of Examination:
Tamil nadu Board of Higher Secondary Education
Permanent register number: 2111119945 HSC Roll number: 5119714
Qualified Year: 2022 HSC Group: HSC Academic
Group Code: Physics/ Chemistry/ Maths/ Biology Medium of Instruction: Tamil
HSC maximum (total) marks: 600 HSC obtained marks: 513
SSLC maximum (total) marks: 500 SSLC obtained marks: 424
Have you applied for NEET ?: No Have you applied for JEE ?: No
Educational Management Information System(EMIS) Number: Community certificate number: FFDB678C6A687B86
332606127 7500257
    """

    print("Extracting applicant details...")
    result = compare_application_and_documents(extracted_text, validation_schema)
    print("\n\nValidation Results:")
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
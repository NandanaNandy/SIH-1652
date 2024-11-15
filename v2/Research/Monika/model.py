from langchain_ollama import OllamaLLM  # Correct import
from langchain_core.prompts import ChatPromptTemplate

# LangChain setup with a prompt template for comparing application details
template = """
You are tasked with comparing the following application details with the extracted document text:
1. Application Details: {reference_details}
2. Extracted Document Text: {extracted_text}

Please compare the two and provide the following:
- A summary of how well the extracted text matches the application details.
- If there are any discrepancies, explain them clearly.
"""

# Initialize Ollama with the Llama2 model
model = OllamaLLM(model="llama3.2")

def compare_application_and_documents(extracted_text, reference_details):
    """Use Llama3.2 model to compare the extracted text and reference details."""
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(template)
    
    # Instead of LLMChain, use RunnableSequence and the invoke method
    chain = prompt | model  # Chaining the prompt and model

    # Prepare inputs
    inputs = {
        "reference_details": reference_details,
        "extracted_text": extracted_text
    }

    # Use invoke() to get the result
    result = chain.invoke(inputs)  # Changed run() to invoke()
    return result

def main():
    # Example extracted text (from OCR or document) and reference details (from the application)
    extracted_text = """
    John Doe has submitted his application for the Software Engineer role. His resume mentions that he has 5 years of experience working with Python, JavaScript, and React. He completed his Bachelor's degree in Computer Science from XYZ University in 2018. John has also worked on multiple web development projects, including an e-commerce platform.
    """

    reference_details = """
    Name: John Doe
    Position Applied: Software Engineer
    Experience: 5 years
    Skills: Python, JavaScript, React
    Education: Bachelor's degree in Computer Science, XYZ University (2018)
    """

    # Check if the extracted text and reference details are available
    if extracted_text.strip() and reference_details.strip():
        print("Comparing extracted text with reference details using Llama2 model...")
        verification_result = compare_application_and_documents(extracted_text, reference_details)
        print("Verification Result:")
        print(verification_result)
    else:
        print("Error: Missing extracted text or reference details.")

# Run the main function
if __name__ == "__main__":
    main()

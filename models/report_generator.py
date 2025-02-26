from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_vertexai import VertexAI

class PathologyPredictionExplanation:
    def __init__(self, model_name: str, project: str, location: str, max_output_tokens: int = 512, temperature: float = 0.1, top_p: float = 0.8, top_k: int = 40):
        """
        Initialize the class with the necessary configurations for the Vertex AI model.

        Args:
            model_name (str): The name of the deployed Vertex AI model.
            project (str): The Google Cloud project ID.
            location (str): The location of the model deployment (e.g., 'us-central1').
            max_output_tokens (int): The maximum number of tokens for the output.
            temperature (float): The temperature for the AI model's response.
            top_p (float): Top probability sampling parameter.
            top_k (int): Top k sampling parameter.
        """
        self.llm = VertexAI(
            model_name=model_name,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            project=project,
            location=location,
        )
        self.template = """
        You are a digital pathology expert.
        A case has been predicted with the following output:
            Prediction: {predicted_label} with a confidence of {confidence:.2f}.
        Based on similar cases from our knowledge graph, here is the retrieved context:
        {context}

        Please provide a detailed explanation discussing why this prediction might be valid. Reference similar cases and include relevant clinical insights.
        """
        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["predicted_label", "confidence", "context"]
        )
    
    def generate_explanation(self, context_data: list, prediction: dict) -> str:
        """
        Generates a detailed explanation using an LLM, based on the prediction result and the retrieved context.
        
        Args:
            context_data (list): A list of strings or node representations from your knowledge graph.
            prediction (dict): The prediction output (with keys "predicted_class" and "probabilities").
        
        Returns:
            str: The generated explanation.
        """
        # Map prediction to human-readable text.
        predicted_label = "Non-cancer" if prediction["predicted_class"] == 0 else "Cancer"
        confidence = prediction["probabilities"][0][prediction["predicted_class"]]
        
        # Convert context data into a single text block.
        context_text = "\n".join([str(item) for item in context_data])
        
        # Create an LLMChain with the prompt template and the configured LLM.
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
        # Run the chain to get the explanation.
        explanation = llm_chain.run({
            "predicted_label": predicted_label,
            "confidence": confidence,
            "context": context_text
        })
        return explanation


# Example usage:
if __name__ == "__main__":
    # Initialize the class with appropriate parameters
    pathology_explainer = PathologyPredictionExplanation(
        model_name="gemini-2.0-flash-001",  # Example model name
        project="kgrag-dp-new",
        location="us-central1"
    )

    # Assume you have retrieved some context from your Neo4j knowledge graph.
    example_context = [
        "Case 1: A 45-year-old female with similar imaging features was diagnosed with an early-stage benign lesion.",
        "Case 2: A 50-year-old male presented with analogous histopathological findings and was managed conservatively."
    ]
    
    # Prediction example
    prediction_output = {
        "predicted_class": 0,
        "probabilities": [[0.9990513920783997, 0.0009485770133323967]]
    }
    
    # Generate the explanation
    explanation = pathology_explainer.generate_explanation(example_context, prediction_output)
    print("Generated Explanation:")
    print(explanation)

from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def get_purchase_probability(company_text: str):
    """
    Uses an AI model to determine the probability of a company buying a product, with detailed reasoning.

    Args:
        company_text: Text extracted from the company's website and other sources.

    Returns:
        A string containing the AI's probability and detailed reasoning.
    """
    if not company_text:
        return "Could not analyze company due to lack of text."

    prompt = f"""
    Our product is: {config.PRODUCT_DESCRIPTION}

    Here is the text from a company's website and other sources:
    ---
    {company_text}
    ---

    First, give the probability (from 0% to 100%) that this company would be interested in our product. Write it as:
    PERCENTAGE: [XX%]
    REASONING:
    Then, provide a detailed, step-by-step explanation of how you arrived at this number. Your explanation must include:
    - The key positive and negative factors you considered (e.g., company size, industry, hiring activity, product fit, budget, urgency, etc.)
    - How each factor increased or decreased the probability
    - A logical, non-random calculation or estimation process, show your calculations. be precise, you can evaluate the percentage yourself.
    - Why you did NOT choose a higher or lower number
    - Think very very very carefully.
    If you do not provide a PERCENTAGE in the exact format above, your answer will be considered incomplete.
    """

    try:
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful, thoughtful sales intelligence analyst. You always provide deep, step-by-step reasoning for your evaluations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error from AI model: {e}"

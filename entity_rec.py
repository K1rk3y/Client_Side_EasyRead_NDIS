from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional


load_dotenv()
api_key = os.getenv('API_KEY_O')
client = OpenAI(api_key=api_key)


def get_refinement_criteria() -> str:
    return """Imagine that you are an illustrator trying to produce a visualisation based on the given textual descriptions. You must first refine this description of this visualisation based on the following metrics:
    
    1. Be Specific and Detailed: The more specific your prompt, the better the image quality. Include details like the setting, objects, colors, mood, and any specific elements you want in the image.
    2. Mood and Atmosphere: Describe the mood or atmosphere you want to convey. Words like “serene,” “chaotic,” “mystical,” or “futuristic” can guide the AI in setting the right tone.
    3. Use Descriptive Adjectives: Adjectives help in refining the image. For example, instead of saying “a dog,” say “a fluffy, small, brown dog.”
    4. Consider Perspective and Composition: Mention if you want a close-up, a wide shot, a bird’s-eye view, or a specific angle. This helps in framing the scene correctly.
    5. Specify Lighting and Time of Day: Lighting can dramatically change the mood of an image. Specify if it’s day or night, sunny or cloudy, or if there’s a specific light source like candlelight or neon lights.
    6. Incorporate Action or Movement: If you want a dynamic image, describe actions or movements. For instance, “a cat jumping over a fence” is more dynamic than just “a cat.”
    7. Avoid Overloading the Prompt: While details are good, too many can confuse the AI. Try to strike a balance between being descriptive and being concise.
    8. Use Analogies or Comparisons: Sometimes it helps to compare what you want with something well-known, like “in the style of Van Gogh” or “resembling a scene from a fantasy novel.”
    9. Specify Desired Styles or Themes: If you have a particular artistic style or theme in mind, mention it. For example, “cyberpunk,” “art deco,” or “minimalist.”

    You also needs to reorient the description based on this art style: Minimalist.

    For each criterion, provide:
    - A score (1-5)
    - Specific issues identified
    - Suggested improvements

    Then, provide an improved version of the description that addresses these issues.
    Ensure the improved version of the translation is wrapped in double quotes."""


def refine_translation(client: OpenAI, current_text: str, original_input: str, 
                        model: str = "gpt-4-turbo") -> Tuple[str, Dict[str, Any]]:
    """Refine the given prompt based on evaluation criteria."""
    refinement_prompt = f"""Original Text: {original_input}

    Current Translation:
    {current_text}

    {get_refinement_criteria()}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in converting plain text to clear and concise prompts for diffusion model use."},
                {"role": "user", "content": refinement_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        feedback = response.choices[0].message.content
        improved_text = feedback.split("improved version")[-1].strip()
        
        return improved_text, {"full_feedback": feedback}
    
    except Exception as e:
        print(f"Error in refinement: {str(e)}")
        return current_text, {"error": str(e)}


def iterative_translation(input_text: str, n_iterations: int = 2, 
                         model: str = "gpt-4-turbo") -> List[Tuple[str, Dict[str, Any]]]:
    
    system_prompt = f"You are an expert in converting plain text to clear and concise prompts for diffusion model use."
    
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=0,
            max_tokens=2000,
            model=model
        )
        current_text = response.choices[0].message.content
    except Exception as e:
        print(f"Error in initial translation: {str(e)}")
        return []
    
    results = [(current_text, {"stage": "initial"})]
    
    # Refinement loop
    for i in tqdm(range(n_iterations), desc="Refining translation"):
        try:
            refined_text, feedback = refine_translation(
                client=client,
                current_text=current_text,
                original_input=input_text,
                model=model
            )
            
            results.append((refined_text, {
                "stage": f"refinement_{i+1}",
                "feedback": feedback
            }))
            current_text = refined_text
            
        except Exception as e:
            print(f"Error in iteration {i+1}: {str(e)}")
            break
    
    return results


def save_refinement_history(results: List[Tuple[str, Dict[str, Any]]], filename: str = "translation_history.txt") -> Optional[str]:
    last_quoted_line = None
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Translation Refinement History ===\n\n")
        for i, (text, metadata) in enumerate(results):
            f.write(f"\n--- Stage: {metadata['stage']} ---\n")
            
            # Check each line in the text for quotes
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    last_quoted_line = line
            
            f.write(text + "\n")
            
            if i > 0 and "feedback" in metadata:
                f.write("\nFeedback from previous version:\n")
                feedback = metadata["feedback"]["full_feedback"]
                f.write(feedback + "\n")
                
                # Also check feedback for quoted lines
                for line in feedback.split('\n'):
                    line = line.strip()
                    if line.startswith('"') and line.endswith('"'):
                        last_quoted_line = line
            
            f.write("\n" + "="*50 + "\n")
    
    return last_quoted_line


def translate(input_text):
    # Run iterative translation
    results = iterative_translation(input_text, n_iterations=1)
    opt = save_refinement_history(results)

    return opt

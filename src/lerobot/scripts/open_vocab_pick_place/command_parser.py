"""
Natural Language Command Parser for Pick-and-Place Tasks
Supports both Groq and OpenAI APIs
"""

import json
import os
from typing import List, Tuple, Optional
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    GROQ = "groq"
    OPENAI = "openai"


class CommandParser:
    """Parse natural language instructions into pick-and-place actions"""
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.GROQ,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize command parser.
        
        Args:
            provider: LLM provider to use (GROQ or OPENAI)
            api_key: API key for the provider
            model: Model name to use (defaults per provider)
        """
        self.provider = provider
        
        # Get API key from environment if not provided
        if api_key is None:
            if provider == LLMProvider.GROQ:
                api_key = os.getenv("GROQ_API_KEY")
            elif provider == LLMProvider.OPENAI:
                api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key is None:
            raise ValueError(f"API key not found for {provider.value}. Set environment variable or pass api_key parameter.")
        
        self.api_key = api_key
        
        # Initialize client based on provider
        if provider == LLMProvider.GROQ:
            from groq import Groq
            self.client = Groq(api_key=api_key)
            self.model = model or "llama-3.3-70b-versatile"
        elif provider == LLMProvider.OPENAI:
            import openai
            openai.api_key = api_key
            self.client = openai
            self.model = model or "gpt-3.5-turbo"
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for command parsing"""
        return """
You are a robotic command parser. Extract "pick" and "place" actions from natural language instructions.

Rules:
1. Output a JSON list of lists: [["object_name", "target_location"], ...]
2. Normalize strict location references:
   - "top right corner" -> "trc"
   - "top left corner" -> "tlc"
   - "middle" or "center" -> "mid"
   - "bottom right corner" -> "brc"
   - "bottom left corner" -> "blc"
3. CRITICAL: If the location is a named object (e.g., "next to the cup"), KEEP the object name (e.g., "cup").
4. Extract multiple pick-place pairs if present in the instruction.
5. Return ONLY the JSON array. No explanation or additional text.

Examples:
Input: "Pick the bottle and place it in the top right corner"
Output: [["bottle", "trc"]]

Input: "Move the cup to the middle, then put the phone next to the laptop"
Output: [["cup", "mid"], ["phone", "laptop"]]

Input: "Grab the red block and place it near the blue cube"
Output: [["red block", "blue cube"]]
"""
    
    def extract_commands(self, instruction: str) -> List[Tuple[str, str]]:
        """
        Parse natural language instruction into pick-place command pairs.
        
        Args:
            instruction: Natural language instruction
            
        Returns:
            List of (pick_object, place_location) tuples
        """
        try:
            if self.provider == LLMProvider.GROQ:
                response = self._query_groq(instruction)
            elif self.provider == LLMProvider.OPENAI:
                response = self._query_openai(instruction)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse JSON response
            cleaned_text = response.replace("```json", "").replace("```", "").strip()
            commands = json.loads(cleaned_text)
            
            # Convert to list of tuples
            return [tuple(cmd) for cmd in commands]
        
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {response}")
            print(f"Error: {e}")
            return []
        except Exception as e:
            print(f"Error in command parsing: {e}")
            return []
    
    def _query_groq(self, instruction: str) -> str:
        """Query Groq API"""
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f'Instruction: "{instruction}"'}
            ],
            model=self.model,
            temperature=0,
        )
        return chat_completion.choices[0].message.content
    
    def _query_openai(self, instruction: str) -> str:
        """Query OpenAI API"""
        response = self.client.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": f'Instruction: "{instruction}"'}
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    
    def validate_commands(
        self, 
        commands: List[Tuple[str, str]],
        available_objects: Optional[List[str]] = None
    ) -> bool:
        """
        Validate parsed commands.
        
        Args:
            commands: List of (pick, place) tuples
            available_objects: Optional list of detected objects to validate against
            
        Returns:
            True if commands are valid
        """
        if not commands:
            print("No commands parsed")
            return False
        
        valid_locations = ["trc", "tlc", "mid", "brc", "blc"]
        
        for pick, place in commands:
            if not pick or not place:
                print(f"Invalid command: pick='{pick}', place='{place}'")
                return False
            
            # If we have available objects, check pick object exists
            if available_objects is not None:
                if pick not in available_objects:
                    print(f"Pick object '{pick}' not in available objects: {available_objects}")
                    return False
                
                # Check place location is valid (either a location or another object)
                if place not in valid_locations and place not in available_objects:
                    print(f"Place location '{place}' not recognized")
                    return False
        
        return True


def main():
    """Test the command parser"""
    
    # Example using Groq (free tier available)
    try:
        parser = CommandParser(provider=LLMProvider.GROQ)
        
        test_instructions = [
            "Pick the cup and place it in the top right corner",
            "Move the bottle to the middle, then put the phone next to the laptop",
            "Grab the red block and place it near the blue cube",
            "Put the apple in the center",
        ]
        
        for instruction in test_instructions:
            print(f"\nInstruction: {instruction}")
            commands = parser.extract_commands(instruction)
            print(f"Parsed commands: {commands}")
            
            # Validate
            if parser.validate_commands(commands):
                print("✓ Commands are valid")
            else:
                print("✗ Commands are invalid")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to set GROQ_API_KEY or OPENAI_API_KEY environment variable")


if __name__ == "__main__":
    main()

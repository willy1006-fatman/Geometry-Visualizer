import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import json
import os
import logging

from RAG import generate_config_gemini
from RAG import knowledge_base_building

# Configure logging
logging.basicConfig(
    filename='geo_construction.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Load configuration file
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# Use file paths from configuration
knowledge_base_file = config['knowledge_base_file']

# Read FAISS index file and processed data file
faiss_index_file = config['faiss_index_file']
processed_data_file = config['processed_data_file']
keyword_column = config['keyword_column']
description_column = config['description_column']
command_column = config['command_column']
sheet_name = config['sheet_name']

search_top_k = config['search_top_k']
max_attemps = config['max_attemps']

gemini_model = config['gemini_model']

result_commands_file = config['result_commands_file']
split_prompt_file = config['split_prompt_file']
generate_prompt_file = config['generate_prompt_file']
split_result_file = config['split_result_file']
pseudo_code_file = config['pseudo_code_file']
valid_pseudo_code_file = config['valid_pseudo_code_file']

# Set gemini API key
genai.configure(api_key = config['gemini_api_key'])

ai_model = genai.GenerativeModel(gemini_model)

# 1. Load FAISS index and data
try:
    index = faiss.read_index(faiss_index_file)
    faiss_indices = index
except Exception as e:
    logging.error(f"Error loading FAISS index {faiss_index_file}: {e}")
    raise

try:
    with open(processed_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    logging.error(f"Error loading processed data file {processed_data_file}: {e}")
    raise

# 2. Load Sentence-BERT model
try:
    model = SentenceTransformer(config['model_name'])
except Exception as e:
    logging.error(f"Error loading Sentence-BERT model: {e}")
    raise

# 3. Define preprocessing function
def preprocess(text):
    text = text.strip()
    if config.get('preprocessing', {}).get('lowercase', False):
        text = text.lower()
    return text

# 4. Define search function using RAG
def search(query_sentence, top_k=search_top_k):
    processed_query = preprocess(query_sentence)
    query_embedding = model.encode([processed_query])[0]
    query_embedding = np.array([query_embedding]).astype('float32')

    try:
        distances, indices = faiss_indices.search(query_embedding, top_k)
    except Exception as e:
        logging.error(f"Error during FAISS search: {e}")
        raise

    results = []

    for idx in indices[0]:
        if idx is not None and idx < len(data):
            keywords = data[idx][keyword_column]
            descriptions = data[idx][description_column]
            commands = data[idx][command_column]
            item = 'Keyword: ' + keywords +  '\nDescription: ' + descriptions + ', Command: ' + commands
            results.append(item)

    return results

# 5. Define the function to split sentences, calling the gemini API
def split_sentence_into_steps(sentence):
    prompt = f"""I have a knowledge base with sentences and the corresponding GeoGebra commands.
I need you to extract objects and their relationships from geometry problems is natural language in order for my retriever to identify commands.
For each problem:

- **Objects**: List "all" mentioned objects (points, lines, circles, angles, triangles, quadrilaterals, macros, etc.). **Do not split macro objects into their individual components** (e.g., "triangle ABC" remains "triangle ABC", not "point A, point B, point C").
    - **Objects and their Relationships**:
        - **Type 1 Relationships**: Describe how each object interacts with other objects (e.g., "angle at vertex A of triangle ABC").
        - **Type 2 Relationships**: Describe how objects are derived from other objects (e.g., "circle T is the circumcircle of triangle ABC").
    - **Include Objects**: List objects contained within this object (those related to this object don't count) (e.g., for triangle ABC: "point A, point B, point C, line AB, line BC, line CA"). **This part will be used later in pseudocode generation to determine declaration dependencies**.

- **Important Rule**: 
    - **You need to strictly follow this rule: `include objects` should only list objects that are contained within the main object. Do not include objects that are related to the main object but not contained within it.**

- Exclude any numerical data or numeric relationships (e.g., "a equals b", "congruent angles").
- Use essential words only
- Replace "line segment" with "segment", do NOT replace "line" with "segment".
- Ensure output is in English.
- Return as a single Python string without quotes and code block.
- You can rename objects or relationships to more common terms if needed.

**Wrong Example:**

point N: 
    - Type 1: foot of perpendicular from point A to line CD 
    - Type 2: derived from perpendicular from point A 
    - include objects: point A, line CD

**Correct Example:**

point N: 
    - Type 1: foot of perpendicular from point A to line CD 
    - Type 2: derived from perpendicular from point A 
    - include objects:

**Reason:** 
Include objects doesn't count those objects that are related to this object but not in this object.

**Format your output as following example:**

**Problem:**
given triangle ABC, draw the internal angle bisector of angle BAC intersects line BC at point D

**Your Answer:**
point D:
    - Type 1:
    - Type 2: intersection of angle bisector of angle BAC with line BC
    - include objects:
angle BAC:
    - Type 1: angle at vertex A of triangle ABC
    - Type 2:
    - include objects:
triangle ABC:
    - Type 1:
    - Type 2:
    - include objects: point A, point B, point C, line AB, line BC, line CA
line BC:
    - Type 1: intersects at point D with angle bisector of angle BAC
    - Type 2: line connecting points B and C
    - include objects: point B, point C
angle bisector of angle BAC:
    - Type 1: intersects at point D with angle bisector of angle BAC
    - Type 2: angle bisector of angle BAC
    - include objects: 

**Another Problem Example with Macro:**

**Problem:**
Given two adjacent sides AB and AD, construct parallelogram ABCD.

**Your Answer:**
parallelogram ABCD:
    - Type 1:
    - Type 2:
    - include objects: point A, point B, point C, point D, segment AB, segment BC, segment CD, segment DA

**Here is my problem:**
{sentence}
"""

    with open(split_prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)

    # Call the gemini API to generate steps
    try:
        response = ai_model.generate_content(prompt)
        steps = response.text.strip()
        with open(split_result_file, 'w', encoding='utf-8') as f:
            f.write(steps)
        return steps
    except Exception as e:
        logging.error(f"Error calling gemini API during sentence splitting: {e}")
        print(f"Error calling gemini API during sentence splitting: {e}")
        return "ERROR"

def generate_pseudo_code(objects_relationships,last_time_pseudo_code,improve):
    prompt = f"""I am solving a geometry problem and want to convert it from literals into GeoGebra commands. Please generate the pseudocodes (step by step guides) for this problem.

Please generate a pseudocode that outlines the steps to construct the specified objects and their relationships in order. The pseudocode should include the following elements:
- **Declare Variables**: Define variables for each object involved. If an object is part of a macro command, declare the macro which includes all its constituent objects.
- **Implement Relationships**: Establish each relationship between the objects, ensuring that macro commands encapsulate their defining properties and constraints.
- **Include Objects Handling**: If an object has included objects listed under "include objects", treat it as a macro and do not declare the included objects separately.

**Important:**
- **Macros** (e.g., "parallelogram ABCD") should be treated as single entities. When declaring a macro, **do not** declare its included objects (e.g., points A, B, C, D) separately.
- Ensure that all necessary geometric constraints within macros are included in the pseudocode.

Additionally, generate a `keywords` section where each keyword corresponds to the object type extracted from each step. The keyword should represent the type of the object (e.g., point, line, triangle, segment, square) without the specific names and fully capture its relationship without being incomplete (e.g., N is the foot of perpendicular from point A to line CD/("foot of perpendicular from point to line" instead of just "foot of perpendicular")).

**Important Formatting Instructions:**
- For each step, output the step description followed by a "/" and then the corresponding keyword.
- Try to include only one relationship in one line(two object and one relationship of them, if it can't do it, ignore this instruction).
- Ensure that each step and its keyword are on the same line, separated by a "/".
- Do not include `steps:` or `keywords:` headers in the output.
- Only generate the list of steps with keywords, without any additional text. Do not use quotes or code blocks.
- If I give you some points to improve and the pseudo code you generated last time, also follow them to improve the ultimate result.

**Example:**

**Here is my problem:**
parallelogram ABCD:
    - Type 1:
    - Type 2:
    - include objects: point A, point B, point C, point D, line AB, line BC, line CD, line DA

**Your Answer:**
1. parallelogram ABCD/parallelogram

**Another Example Without Macro:**

**Here is my problem:**
point D:
    - Type 1:
    - Type 2: intersection of angle bisector of angle BAC with line BC
    - include objects:
angle BAC:
    - Type 1: angle at vertex A of triangle ABC
    - Type 2:
    - include objects:
triangle ABC:
    - Type 1:
    - Type 2:
    - include objects: point A, point B, point C, line AB, line BC, line CA
line BC:
    - Type 1: intersects at point D with angle bisector of angle BAC
    - Type 2: line connecting points B and C
    - include objects: point B, point C
angle bisector of angle BAC:
    - Type 1: intersects at point D with angle bisector of angle BAC
    - Type 2: angle bisector of angle BAC
    - include objects: 

**Your Answer:**
1. triangle ABC/triangle
2. line BC is a line connecting point B and C/line of two point
3. angbiBAC is the angle bisector of angle BAC./angle bisector of a angle
4. D is the intersection of angle bisector and line BC./intersection of two lines

**Here is my problem:**
{objects_relationships}

**This is your the pseudo code you generated last time**
{last_time_pseudo_code}

**Here is some advice to help you improve them**
{improve}
"""

    try:
        response = ai_model.generate_content(prompt)
        pseudo_code = response.text.strip()
        return pseudo_code
    except Exception as e:
        logging.error(f"Error calling gemini API during pseudocode generation: {e}")
        print(f"Error calling gemini API during pseudocode generation: {e}")
        return "ERROR"

# 7. Define the function to convert pseudocode to GeoGebra commands, calling the gemini API
def pseudo_code_to_commands(pseudo_code):
    lines = pseudo_code.split('\n')
    context = ''

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('/')
        if len(parts) != 2:
            logging.warning(f"Line format incorrect (missing '/'): {line}")
            continue

        step = parts[0].strip()
        keyword = parts[1].strip()

        search_results = search(keyword)

        context += f"{step}\n"
        for i, res in enumerate(search_results, start=1):
            context += f"({i}): {res}\n"
        context += '\n'

    # Updated prompt with additional instructions
    prompt = f"""I have the steps for a geometry problem.
Please convert these steps into corresponding GeoGebra commands.

For each step, you have several alternative command options, each with its own purpose (marked as keyword), sometimes they may be the same but uses different command(most of the time it depends on whether some objects is declared or not), you need to choose the one that fits your given condition(some of them might needs objects that isn't given to you, you should choose another one that uses given condition with higher priority. Note that for polygons or bigger objects, small objects inside (e.g. points, segments) might not be given to you, feel free to declare them by yourself).
When selecting commands, you need to consider both the intent of the pseudocode step and the intent of each command's keyword.
This means you must ensure that the chosen command aligns with the intent of the pseudocode step.
If the commands are appropriate and there isn't a missing object inside bigger ones you already declared, you may choose come up with commands with the same logic as the given commands.
If there are some points that haven't been declared, you can consider declaring them by yourself.

Please:
- Avoid annotations; no additional text is needed.
- Ensure every name is unique.
- Choose object coordinates freely.
- If information is insufficient, explain why.
- Single capital letters(sometimes also with numbers right behind) in commands denote variables; replace them based on the original question and pseudocode.
For example : Given the command quadrilateralABCD = Polygon(A,B,C,D), and the problem of drawing a quadrilateral ACDB. You should output `quadrilateral ACDB = Polygon(A,C,D,B)`
- Use only the given commands unless no command is appropriate

Output each GeoGebra command on an individual line with plain text without quotes and code block, no empty lines.

**Example:**

**Here is my problem:**
quadrilateral ABCD
(1): Keyword: a quadrilateral
Description: quadrilateral ABCD, Command: quadrilateralABCD = Polygon(A,B,C,D)
(2): Keyword: quadrilateral with no two sides parallel
Description: quadrilateral ABCD, Command: A = (0,0); B = (5,5); C = (7,4); D = (10,0); quadrilateralABCD = Polygon(A,B,C,D)

**Your Answer:**
A = (0,0)
B = (5,5)
C = (7,4)
D = (10,0)
quadrilateralABCD = Polygon(A,B,C,D)

**Another Example Without Macro:**

**Here is my problem:**
1. triangle ABC
(1): Keyword: a triangle
Description: triangle ABC, Command: triangleABC = Polygon(A,B,C)
(2): Keyword: an equilateral triangle
Description: equilateral triangle ABC, Command: C=B+Rotate(Vector(A,B),((2 π)/(3)));regtangleABC=Polygon(A,B,C)

2. line BC is a line connecting point B and C
(1): Keyword: a point on a line segment
Description: point A on segment BC, Command: A = Point(Segment(B,C))
(2): Keyword: a point on a line segment
Description: point A on line BC, Command: A = Point(Line(A,B))

3. angbiBAC is the angle bisector of angle BAC.
(1): Keyword: an angle bisector of an angle
Description: angle bisector of angle ABC, Command: angbiABC = AngleBisector(A,B,C)
(2): Keyword: an angle
Description: angle ABC, Command: angB = Angle(A,B,C), lineAB = Line(A,B), lineBC = Line(B,C)

4. D is the intersection of angle bisector and line BC.
(1): Keyword: the intersection of two lines
Description: P is the intersection of line L1 and line L2, Command: P = Intersect(L1,L2)
(2): Keyword: a point on a line segment
Description: point A on segment BC, Command: A = Point(Segment(B,C))

**Your Answer:**
A = (0,0)
B = (3,4)
C = (5,1)
triangle ABC = Polygon(A, B, C)
lineBC = line(B,C)
angbiBAC = AngleBisector(B,A,C)
D = Intersect(angbiBAC,lineBC)

**Here is my problem:**
{context}
"""

    with open(generate_prompt_file, 'w', encoding='UTF-8') as f:
        f.write(prompt)

    try:
        response = ai_model.generate_content(prompt)
        commands = response.text.strip()
        return commands
    except Exception as e:
        logging.error(f"Error calling gemini API during pseudocode to command conversion: {e}")
        print(f"Error calling gemini API during pseudocode to command conversion: {e}")
        return "ERROR"

# 8. Define the function to validate pseudocode, calling the gemini API
def validate_pseudo_code(initial_question,objects_relationships, pseudo_code):
    prompt = f"""I have the following geometry problem with objects and their relationships:

{objects_relationships}

Please verify if the following pseudocode correctly includes all objects and relationships.
You need to notice two points:
1. All variables are declared.
   Note:
       - There are three ways to declare variables:
           1. Declare at the beginning (e.g., `point A`)
           2. Declare within relationships (e.g., `angbiBAC is the angle bisector of angle BAC`)
           3. Declare within a macro object (e.g., point A,B,C is declared in `Triangle ABC`)
2. Each unique relationship should be implemented in the pseudocode.
   Note:
       - A unique relationship means some same relationships can be considered as one.

Also here is the original problem, please check if there are any relationship that doesn't be notice.

{initial_question}

Respond with "Valid" if it does, or "Wrong" if it doesn't.

If your validation is wrong, besides returns wrong, also explain why and give some advice to let me give back to the pseudo code generator to help it improve it, including missing relationships.

return in text without additional description, quotes and code block.

Here is the pseudocode:
{pseudo_code}
"""


    try:
        response = ai_model.generate_content(prompt)
        validation = response.text.strip().lower()
        with open(valid_pseudo_code_file, 'w', encoding='utf-8') as f:
            f.write(validation)
        return validation
    except Exception as e:
        logging.error(f"Error calling gemini API during pseudocode validation: {e}")
        print(f"Error calling gemini API during pseudocode validation: {e}")
        return "ERROR"

# 9. Define the main function to generate GeoGebra commands from a query sentence
def generate_response(query_sentence):
    # Use gemini API to split the sentence into objects and relationships
    steps = split_sentence_into_steps(query_sentence)

    if steps == "ERROR":
        return "Error generating steps."

    # The split result is already in the desired format
    objects_relationships = steps


    # Validate pseudocode

    last_time_pseudo_code = "You haven't generate pseudo code before!"

    improve = ""

    for i in range(max_attemps):
        # Generate pseudocode
        pseudo_code = generate_pseudo_code(objects_relationships,last_time_pseudo_code,improve)

        with open(pseudo_code_file, 'w', encoding='utf-8') as f:
            f.write(pseudo_code)
        
        if "wrong" in pseudo_code.lower():
            return "Error generating code, please check your input"
        
        validation = validate_pseudo_code(objects_relationships,objects_relationships, pseudo_code)
        logging.info(f"Validation result: {validation}")
        print(f"Validation result: {validation}")
        
        if "valid" in validation.lower():
            break

        print(i)
        
        last_time_pseudo_code = pseudo_code
        improve = validation

    if "valid" in validation.lower():
        print("valid")

    # Convert pseudocode to GeoGebra commands
    ggb_commands = pseudo_code_to_commands(pseudo_code)
    if ggb_commands == "ERROR":
        return "Error converting pseudocode to commands."

    with open(result_commands_file, 'w', encoding='utf-8') as f:
        f.write(ggb_commands)
    logging.info(f"Generated GeoGebra Commands:\n{ggb_commands}")

    return ggb_commands

# 10. Main Execution (if applicable)
if __name__ == "__main__":
    # Example usage
    sample_question = "Given two adjacent sides AB and AD, construct parallelogram ABCD."
    response = generate_response(sample_question)
    print("GeoGebra Commands:")
    print(response)
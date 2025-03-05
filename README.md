# Geometry Visualizer  

## Description  
YTP project by 這什麼CSES樹.  

Geometry Visualizer is an application that generates geometric diagrams from text input with 80% accuracy for high school-level problems.  

It is built using RAG, a custom knowledge base, and a unique combination of LLMs we developed. The application uses GeoGebra to generate diagrams from scripts and Gemini to generate scripts from prompts.  

## Installation and Setup  

1. Install the required packages by running the following command in the terminal:  
   ```sh
   pip install -r requirements.txt
   ```  

2. Replace `YOUR_API_KEY` in line 4 of `/RAG/generate_config_gemini.py` with your Gemini API key.  

3. Select a UI template by modifying line 13 of `app.py`:
   - For a simplified UI, use `front_webpage_simple.html`.
   - For the original version, use `front_webpage.html`.  

4. Start the application by running:  
   ```sh
   python app.py
   ```  

## Usage Guide  

### Simplified Version  

1. Before running `app.py`, set `TEMPLATE.html` to `front_webpage_simple.html`.  
2. When the UI loads in your browser, it should look like this:  
   ![image](https://github.com/user-attachments/assets/8b878626-41c5-46a2-aeee-7d83a5019132)  
3. Enter a description in the text box on the right.  
4. Click the **Go** button to generate the diagram.  
5. The diagram will appear on the left canvas.  

### Original Version  

1. Before running `app.py`, set `TEMPLATE.html` to `front_webpage.html`.  
2. When the UI loads in your browser, it should look like this:  
   ![image](https://github.com/user-attachments/assets/ac188491-387f-4385-a238-ccf853c7c271)  
3. Enter a description in the text box in the upper right.  
4. Click **Generate Script** to create the GeoGebra script.  
5. (Optional) Modify the generated script in the text box below.  
6. Click **Run Script** to generate the diagram from the script.  
7. The diagram will appear on the left canvas.  

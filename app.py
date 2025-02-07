import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

class AIAssistant:
    def __init__(self):
        try:
            # Get API key from environment variable
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")

            # Configure Gemini API
            genai.configure(api_key=api_key)

            # Generation configuration
            generation_config = {
                'temperature': 0.7,
                'top_p': 1.0,
                'max_output_tokens': 2048
            }

            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]

            # Initialize Gemini model
            self.model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                generation_config=generation_config,
                safety_settings=safety_settings
            )

            print("Initialization complete!")
        except Exception as e:
            print(f"Initialization error: {str(e)}")
            raise

    def extract_pdf_text(self, pdf_file):
        if pdf_file is None:
            return ""
        
        try:
            pdf_reader = PdfReader(pdf_file.name)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text[:10000]
        except Exception as e:
            print(f"PDF processing error: {str(e)}")
            return f"PDF processing error: {str(e)}"

    def generate_response(self, prompt, language, tone, mode, topic, current_level, available_time, learning_method, goal, pdf_file):
        try:
            # Extract PDF context
            pdf_context = self.extract_pdf_text(pdf_file)

            # Formulate input text based on mode
            if mode == "Chat":
                input_text = (
                    f"Respond in {language} with a {tone} tone: {prompt}\n\n"
                    f"Additional context from PDF (if available):\n{pdf_context}"
                )
            else:  # Study plan mode
                input_text = (
                    f"Create a study plan in {language} with a {tone} tone.\n"
                    f"Topic: {topic}\n"
                    f"Current Level: {current_level}\n"
                    f"Available Time: {available_time} hours per week\n"
                    f"Learning Method: {learning_method}\n"
                    f"Goal: {goal}\n"
                    f"Additional context from PDF (if available):\n{pdf_context}"
                )

            # Generate response
            response = self.model.generate_content(input_text)
            return response.text

        except Exception as e:
            print(f"Error occurred: {str(e)}")
            return f"An error occurred: {str(e)}"

# Create assistant instance
assistant = AIAssistant()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Educational Assistant")
    gr.Markdown("Ask questions and receive detailed answers")

    with gr.Row():
        with gr.Column():
            language = gr.Dropdown(
                label="Select response language",
                choices=["English", "Русский", "Español", "Deutsch"],
                value="English"
            )
            tone = gr.Dropdown(
                label="Select response tone",
                choices=["formal", "informal", "humorous", "serious"],
                value="formal"
            )
            mode = gr.Dropdown(
                label="Select operation mode",
                choices=["Chat", "Study plan"],
                value="Chat"
            )
            topic = gr.Textbox(
                label="Topic or skill to study",
                visible=False
            )
            current_level = gr.Dropdown(
                label="Current knowledge level",
                choices=["beginner", "intermediate", "advanced"],
                visible=False
            )
            available_time = gr.Number(
                label="Hours per week available for study",
                visible=False
            )
            learning_method = gr.Dropdown(
                label="Preferred learning method",
                choices=["visual", "auditory", "practical", "reading"],
                visible=False
            )
            goal = gr.Textbox(
                label="Specific learning goal or target skill level",
                visible=False
            )
            pdf_file = gr.File(
                label="Upload PDF for additional context (optional)",
                file_types=[".pdf"]
            )

        with gr.Column():
            output = gr.Textbox(
                label="Response",
                lines=10
            )
            prompt = gr.Textbox(
                lines=4,
                placeholder="Enter your question...",
                label="Question"
            )
            with gr.Row():
                submit_button = gr.Button("Get response")
                clear_button = gr.Button("Clear history")

    # Event handlers
    def update_fields(selected_mode):
        if selected_mode == "Study plan":
            return [gr.update(visible=True)] * 5 + [gr.update(visible=False)]
        else:
            return [gr.update(visible=False)] * 5 + [gr.update(visible=True)]

    mode.change(
        update_fields,
        inputs=[mode],
        outputs=[topic, current_level, available_time, learning_method, goal, prompt]
    )

    submit_button.click(
        fn=assistant.generate_response,
        inputs=[prompt, language, tone, mode, topic, current_level, 
                available_time, learning_method, goal, pdf_file],
        outputs=output
    )

    def clear_history():
        return "", "", None

    clear_button.click(
        fn=clear_history,
        inputs=[],
        outputs=[prompt, output, pdf_file]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
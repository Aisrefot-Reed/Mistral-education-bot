import os
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load environment variables
load_dotenv()

class AIAssistant:
    def __init__(self):
        self.client = InferenceClient(
            token=os.getenv('HF_API_KEY')  # Retrieve API key from environment variables
        )
        self.model = "mistralai/Mistral-7B-Instruct-v0.3"

    def generate_response(self, prompt, language, tone, mode, topic, current_level, available_time, learning_method, goal, pdf_file):
        try:
            # Process PDF file if uploaded
            pdf_text = ""
            if pdf_file is not None:
                pdf_reader = PdfReader(pdf_file.name)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

            # Formulate message for the model
            if mode == "Chat":
                user_content = f"{prompt}\n\nAdditional context from PDF:\n{pdf_text}" if pdf_text else prompt
            elif mode == "Study plan":
                user_content = (
                    f"Topic: {topic}\n"
                    f"Current level: {current_level}\n"
                    f"Available time: {available_time}\n"
                    f"Learning method: {learning_method}\n"
                    f"Goal: {goal}\n"
                )

            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful AI assistant that provides {tone} responses in {language}. Default language: English. Always respond in the selected language, but use English if no language is explicitly selected."
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            # Generate response using the model
            response = self.client.text_generation(
                prompt=messages[-1]["content"],
                model=self.model,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

            return response

        except Exception as e:
            print(f"Error occurred: {str(e)}")  # Log the error
            return f"An error occurred: {str(e)}"

# Create an instance of the assistant
assistant = AIAssistant()

# Define the Gradio interface
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
            submit_button = gr.Button("Get response")

    # Function to update field visibility based on selected mode
    def update_fields(selected_mode):
        if selected_mode == "Study plan":
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # Update field visibility when mode changes
    mode.change(
        update_fields,
        inputs=[mode],
        outputs=[topic, current_level, available_time, learning_method, goal, prompt]
    )

    # Define action when button is clicked
    submit_button.click(
        fn=assistant.generate_response,
        inputs=[prompt, language, tone, mode, topic, current_level, available_time, learning_method, goal, pdf_file],
        outputs=output
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()
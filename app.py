import os
import gradio as gr
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Загрузка переменных окружения
load_dotenv()

class AIAssistant:
    def __init__(self):
        self.client = InferenceClient(
            token=os.getenv('HF_API_KEY')  # Получение API-ключа из переменных окружения
        )
        self.model = "mistralai/Mistral-7B-Instruct-v0.3"

    def generate_response(self, prompt, language, tone, pdf_file):
        try:
            # Обработка PDF-файла, если он загружен
            pdf_text = ""
            if pdf_file is not None:
                pdf_reader = PdfReader(pdf_file.name)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

            # Формирование сообщения для модели
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful AI assistant that provides {tone} responses in {language}."
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nAdditional context from PDF:\n{pdf_text}" if pdf_text else prompt
                }
            ]

            # Генерация ответа с использованием модели
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
            print(f"Error occurred: {str(e)}")  # Логирование ошибки
            return f"Произошла ошибка: {str(e)}"

# Создание экземпляра ассистента
assistant = AIAssistant()

# Определение интерфейса Gradio
with gr.Blocks() as demo:
    gr.Markdown("# AI Educational Assistant")
    gr.Markdown("Задавайте вопросы и получайте подробные ответы")

    with gr.Row():
        with gr.Column():
            language = gr.Dropdown(
                label="Выберите язык ответа",
                choices=["English", "Русский", "Español", "Deutsch"],
                value="English"
            )
            tone = gr.Dropdown(
                label="Выберите тон ответа",
                choices=["формальный", "неформальный", "юмористический", "серьезный"],
                value="формальный"
            )
            pdf_file = gr.File(
                label="Загрузите PDF для дополнительного контекста (необязательно)",
                file_types=[".pdf"]
            )
            prompt = gr.Textbox(
                lines=4,
                placeholder="Введите ваш вопрос...",
                label="Вопрос"
            )
            submit_button = gr.Button("Получить ответ")

        with gr.Column():
            output = gr.Textbox(
                label="Ответ",
                lines=10
            )

    # Определение действия при нажатии кнопки
    submit_button.click(
        fn=assistant.generate_response,
        inputs=[prompt, language, tone, pdf_file],
        outputs=output
    )

# Запуск приложения
if __name__ == "__main__":
    demo.launch()

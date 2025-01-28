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

    def generate_response(self, prompt, language, tone, mode, topic, current_level, available_time, learning_method, goal, pdf_file):
        try:
            # Обработка PDF-файла, если он загружен
            pdf_text = ""
            if pdf_file is not None:
                pdf_reader = PdfReader(pdf_file.name)
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text()

            # Формирование сообщения для модели
            if mode == "Chat":
                user_content = f"{prompt}\n\nAdditional context from PDF:\n{pdf_text}" if pdf_text else prompt
            elif mode == "Study plan":
                user_content = (
                    f"Тема: {topic}\n"
                    f"Текущий уровень: {current_level}\n"
                    f"Доступное время: {available_time}\n"
                    f"Метод обучения: {learning_method}\n"
                    f"Цель: {goal}\n"
                )

            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful AI assistant that provides {tone} responses in {language}."
                },
                {
                    "role": "user",
                    "content": user_content
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
            mode = gr.Dropdown(
                label="Выберите режим работы",
                choices=["Chat", "Study plan"],
                value="Chat"
            )
            topic = gr.Textbox(
                label="Тема или навык для изучения",
                visible=False
            )
            current_level = gr.Dropdown(
                label="Текущий уровень знаний",
                choices=["начальный", "средний", "продвинутый"],
                visible=False
            )
            available_time = gr.Number(
                label="Часы в неделю, доступные для обучения",
                visible=False
            )
            learning_method = gr.Dropdown(
                label="Предпочтительный метод обучения",
                choices=["визуальный", "слуховой", "практический", "чтение"],
                visible=False
            )
            goal = gr.Textbox(
                label="Конкретная цель обучения или целевой уровень навыков",
                visible=False
            )
            pdf_file = gr.File(
                label="Загрузите PDF для дополнительного контекста (необязательно)",
                file_types=[".pdf"]
            )

        with gr.Column():
            output = gr.Textbox(
                label="Ответ",
                lines=10
            )
            prompt = gr.Textbox(
                lines=4,
                placeholder="Введите ваш вопрос...",
                label="Вопрос"
            )
            submit_button = gr.Button("Получить ответ")

    # Функция для обновления видимости полей на основе выбранного режима
    def update_fields(selected_mode):
        if selected_mode == "Study plan":
            return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    # Обновление видимости полей при изменении режима
    mode.change(
        update_fields,
        inputs=[mode],
        outputs=[topic, current_level, available_time, learning_method, goal, prompt]
    )

    # Определение действия при нажатии кнопки
    submit_button.click(
        fn=assistant.generate_response,
        inputs=[prompt, language, tone, mode, topic, current_level, available_time, learning_method, goal, pdf_file],
        outputs=output
    )

# Запуск приложения
if __name__ == "__main__":
    demo.launch()

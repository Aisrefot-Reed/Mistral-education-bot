import gradio as gr
from huggingface_hub import InferenceClient
import pdfplumber

# Используемую модель оставляем без изменений
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Функция для работы с моделью
def respond(message, history: list[tuple[str, str]], system_message, max_tokens, temperature, top_p):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})
    response = ""

    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Функция для обработки PDF-файлов
def process_pdf(file):
    try:
        with pdfplumber.open(file.name) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return f"Содержимое PDF:\n{text[:1000]}..."  # Ограничиваем вывод 1000 символами
    except Exception as e:
        return f"Ошибка при обработке PDF: {e}"

# Функция для генерации Study Plan
def generate_study_plan(topic, level, time, method, goal):
    prompt = (
        f"Создай учебный план по теме '{topic}' для уровня '{level}', "
        f"с учетом {time} часов в неделю, предпочтительным методом обучения '{method}'. "
        f"Цель обучения: '{goal}'."
    )
    return prompt  # Можете заменить это на вызов модели, если потребуется.

# Создание интерфейса
with gr.Blocks() as demo:
    with gr.Row():
        # Левая панель (30% ширины)
        with gr.Column(scale=3):
            mode_choice = gr.Radio(
                choices=["Chat", "Study plan", "Option 2"], value="Chat", label="Выберите режим"
            )

            # Окно для "Study plan", по умолчанию скрыто
            with gr.Group(visible=False) as study_plan_panel:
                topic = gr.Textbox(label="Тема для изучения")
                level = gr.Radio(
                    choices=["Начальный", "Средний", "Продвинутый"], label="Текущий уровень знаний"
                )
                time = gr.Number(label="Часы в неделю, доступные для обучения")
                method = gr.Radio(
                    choices=["Визуальный", "Слуховой", "Практический", "Чтение"],
                    label="Предпочтительный метод обучения",
                )
                goal = gr.Textbox(label="Цель обучения")
                submit_button = gr.Button("Создать учебный план")
                study_plan_output = gr.Textbox(label="Ваш учебный план", interactive=False)
                
                # Генерация Study Plan
                submit_button.click(
                    generate_study_plan,
                    inputs=[topic, level, time, method, goal],
                    outputs=study_plan_output,
                )

            # Кнопка загрузки PDF
            pdf_upload = gr.File(label="Загрузите PDF", file_types=[".pdf"])
            pdf_output = gr.Textbox(label="Результат обработки PDF", interactive=False)
            pdf_upload.change(process_pdf, inputs=pdf_upload, outputs=pdf_output)

        # Правая панель (70% ширины)
        with gr.Column(scale=7):
            chat_input = gr.Textbox(label="Введите сообщение", placeholder="Введите ваш запрос...")
            chat_output = gr.Textbox(label="Ответ", placeholder="Ответ модели...", interactive=False)
            send_button = gr.Button("Отправить")

            # Логика для респондера
            def handle_mode(message, system_message, mode):
                if mode == "Study plan":
                    return "Для генерации учебного плана используйте левую панель."
                return list(respond(message, [], system_message, 512, 0.7, 0.95))

            # Связываем ввод с обработкой
            send_button.click(
                handle_mode,
                inputs=[chat_input, "Вы системный промпт", mode_choice],
                outputs=chat_output,
            )

    # Управление видимостью "Study plan"
    def toggle_mode(mode):
        return {"visible": mode == "Study plan"}

    mode_choice.change(toggle_mode, inputs=[mode_choice], outputs=[study_plan_panel])

# Запуск интерфейса
if __name__ == "__main__":
    demo.launch()



if __name__ == "__main__":
    demo.launch()

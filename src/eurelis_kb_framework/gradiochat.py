import textwrap

import gradio as gr

memory = {"documents": []}


def markdown_from_memory() -> str:
    documents = memory.get("documents", [])
    result = "# Sources\n"
    for document in documents:
        result += textwrap.shorten(document.page_content, 200)

        result += "\n\n"
        result += " > " + document.metadata.get("source", "")
        result += "\n\n"
        result += f"```{document.metadata}```\n\n"

        result += "\n\n---\n\n"

    return result


def define_chatbot(wrapper):
    define_chatbot.chain = wrapper.get_chain()
    # example with filter: define_chatbot.chain = wrapper.get_chain(retriever_kwargs={'search_kwargs': {'filter': {'article_num': "R141-38-4"}}})
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                msg = gr.Textbox()
                clear = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown(markdown_from_memory, every=2)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            memory["documents"] = []
            chain_answer = define_chatbot.chain(history[-1][0])
            memory["documents"] = chain_answer.get("source_documents", [])

            bot_message = chain_answer["answer"]

            history[-1][1] = bot_message

            return history

        def clear_lambda():
            memory["documents"] = []
            define_chatbot.chain = wrapper.get_chain()
            return None

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(clear_lambda, None, chatbot, queue=False).then()

    demo.queue(max_size=10)

    return demo

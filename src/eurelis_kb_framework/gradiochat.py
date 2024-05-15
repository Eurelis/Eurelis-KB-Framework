import textwrap
from typing import Optional, Dict, Any

import gradio as gr  # type: ignore[import-not-found]

memory: Dict[str, Any] = {"documents": [], "selfcheck": None}


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


def selfcheck_from_memory() -> Optional[dict]:
    return memory.get("selfcheck", None)


def define_chatbot(wrapper, selfcheck: bool = False):
    setattr(define_chatbot, "chain", wrapper.get_chain())

    if selfcheck:
        from eurelis_kb_framework.addons.checker.chat_checker import ChatChecker
        from eurelis_kb_framework.addons.checker.check_input_callback import (
            CheckInputCallback,
        )

        setattr(define_chatbot, "selfcheck", ChatChecker(wrapper.lazy_get_llm()))
        setattr(
            define_chatbot,
            "callbacks",
            [
                CheckInputCallback(
                    getattr(define_chatbot, "selfcheck"), method=None, language="en"
                )
            ],
        )
    else:
        setattr(define_chatbot, "callbacks", [])

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()

                if selfcheck:
                    gr.JSON(selfcheck_from_memory, every=2)

                msg = gr.Textbox()
                clear = gr.Button("Clear")

            with gr.Column(scale=1):
                gr.Markdown(markdown_from_memory, every=2)

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            memory["documents"] = []
            memory["selfcheck"] = None
            chain = getattr(define_chatbot, "chain")
            chain_answer = chain.invoke(
                history[-1][0], {"callbacks": getattr(define_chatbot, "callbacks")}
            )

            if selfcheck:
                memory["selfcheck"] = chain_answer.get("selfcheck")

            memory["documents"] = chain_answer.get("source_documents", [])

            bot_message = chain_answer["answer"]

            history[-1][1] = bot_message

            return history

        def clear_lambda():
            memory["documents"] = []
            setattr(define_chatbot, "chain", wrapper.get_chain())
            return None

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(clear_lambda, None, chatbot, queue=False).then()

    demo.queue(max_size=10)

    return demo

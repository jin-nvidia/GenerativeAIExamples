from typing import Any, Dict, List, Tuple, Literal, Callable, Annotated, Literal, Optional
import uuid
import sys 
import os
import logging
import gradio as gr
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from ui_assets.css.css import css, theme



def get_config_with_new_thread_id():
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }
    return config

def launch_demo_ui(assistant_graph, server_port, NVIDIA_API_KEY):
    # Establish a connection to the Riva server
    _LOGGER = logging.getLogger()
        

    global config 
    config = get_config_with_new_thread_id()

    def _print_event(event: dict, _printed: set, max_length=1500):
        return_print = ""
        current_state = event.get("dialog_state")
        if current_state:
            print("Currently in: ", current_state[-1])
            return_print += "Currently in: "
            return_print += current_state[-1]
            return_print += "\n"
        message = event.get("messages")
        latest_msg_chatbot = ""
        if message:
            if isinstance(message, list):
                message = message[-1]
            if message.id not in _printed:
                msg_repr = message.pretty_repr()
                msg_repr_chatbot = str(message.content)
                if len(msg_repr) > max_length:
                    msg_repr = msg_repr[:max_length] + " ... (truncated)"
                    msg_repr_chatbot = msg_repr_chatbot[:max_length] + " ... (truncated)"
                return_print += msg_repr
                latest_msg_chatbot = msg_repr_chatbot
                print(msg_repr)
                _printed.add(message.id)
        return_print += "\n"
        return return_print, latest_msg_chatbot

    def interact(query: str, chat_history: List[Tuple[str, str]], full_response: str):
        _printed = set()
        # example with a single tool call
        events = assistant_graph.stream(
            {"messages": ("user", query)}, config, stream_mode="values"
        )
        
        latest_response = ""
        for event in events:
            return_print, latest_msg =  _print_event(event, _printed)
            if full_response is not None:
                full_response += return_print
            else:
                full_response = return_print
            if latest_msg != "":
                latest_response = latest_msg

        yield "", chat_history + [[query, latest_response]], full_response, latest_response

    def new_thread():
        global config
        config = get_config_with_new_thread_id()


    with gr.Blocks(title = "Welcome to the Healthcare Contact Center", theme=theme, css=css) as demo:
        gr.Markdown("# Welcome to the Healthcare Contact Center")

        # session specific state across runs
        state = gr.State()
        

        with gr.Row(equal_height=True):
            chatbot = gr.Chatbot(label="Healthcare Contact Center Agent", elem_id="chatbot", show_copy_button=True)
            latest_response = gr.Textbox(label = "Latest Response", visible=False)
            full_response = gr.Textbox(label = "Full Response Log", visible=False, elem_id="fullresponsebox", lines=25)
    

        
        # input
        with gr.Row():
            with gr.Column(scale = 10):
                msg = gr.Textbox(label = "Input Query", show_label=True, placeholder="Enter text and press ENTER", container=False,)
        
        # buttons
        with gr.Row():
            submit_btn = gr.Button(value="Submit")
            _ = gr.ClearButton([msg, chatbot], value="Clear UI")
            full_resp_show = gr.Button(value="Show Full Response")
            full_resp_hide = gr.Button(value="Hide Full Response", visible=False)
        with gr.Row():
            new_thread_btn = gr.Button(value="Clear Chat Memory")
        
        # hide/show context
        def _toggle_full_response(btn: str) -> Dict[gr.component, Dict[Any, Any]]:
            if btn == "Show Full Response":
                out = [True, False, True]
            if btn == "Hide Full Response":
                out = [False, True, False]
            return {
                full_response: gr.update(visible=out[0]),
                full_resp_show: gr.update(visible=out[1]),
                full_resp_hide: gr.update(visible=out[2]),
            }

        full_resp_show.click(_toggle_full_response, [full_resp_show], [full_response, full_resp_show, full_resp_hide])
        full_resp_hide.click(_toggle_full_response, [full_resp_hide], [full_response, full_resp_show, full_resp_hide])

        msg.submit(interact, [msg, chatbot, full_response], [msg, chatbot, full_response, latest_response])
        submit_btn.click(interact, [msg, chatbot, full_response], [msg, chatbot, full_response, latest_response])

        new_thread_btn.click(new_thread, [],[])

        

    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=server_port,
        favicon_path="ui_assets/css/faviconV2.png",
        allowed_paths=[
            "ui_assets/fonts/NVIDIASansWebWOFFFontFiles/WOFF2/NVIDIASans_W_Rg.woff2",
            "ui_assets/fonts/NVIDIASansWebWOFFFontFiles/WOFF2/NVIDIASans_W_Bd.woff2",
            "ui_assets/fonts/NVIDIASansWebWOFFFontFiles/WOFF2/NVIDIASans_W_It.woff2",
        ]
    )
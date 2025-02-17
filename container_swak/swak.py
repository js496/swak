import os
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
from huggingface_hub import snapshot_download
import gradio as gr
import threading
import psutil
import pathlib
from pathlib import Path
import re
import logging
from fastapi.middleware.cors import CORSMiddleware

# logging.basicConfig(
#     filename="logfile.log",
#     level=logging.DEBUG,
#     filemode='a',
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger('my_logger')
# logger.setLevel(logging.DEBUG)
# file_handler = logging.FileHandler('logfile.log')
# file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# logger.debug('[DEBUG]')
# logger.info('[INFO]')
# logger.warning('[WARNING]')
# logger.error('[ERROR]')
# logger.critical('[CRITICAL]')
# logger = logging.getLogger()

logging.basicConfig(filename='logfile.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def load_log_file():
    try:
        with open('logfile.log', 'r') as file:
            # log_contents = file.read()
            # return log_contents
            lines = file.readlines()
            return ''.join(lines[-20:])

    except Exception as e:
        return f'{e}'
   
current_models_data = []
current_available_models_data = []
db_gpu_data = []
db_gpu_data_len = ''

HF_TOKEN = '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'
LOCAL_DIR = './models'

local = os.name == 'nt'
# script_dir = pathlib.Path(__file__).parent.resolve()
script_dir = Path(__file__).parent.resolve()
print("Script directory:", script_dir)


def get_available_models():
    global script_dir
    global current_available_models_data
    res_model_arr = []
    
    try:
        models_dir = script_dir / 'models'
        
        if not models_dir.exists() or not models_dir.is_dir():
            print(f"No 'models' directory found in {script_dir}")
            return f'no models found'
            return gr.update(choices=[], value=None, show_label=True, label="No models found!")

        model_dict = {}
        for first_level_folder in models_dir.iterdir(): # pipeline_tag
            first_level_path_arr = re.split(r"[\\/]+", str(first_level_folder))
            current_pipeline_tag = first_level_path_arr[-1]
            if first_level_folder.is_dir():
                current_sub_folder_path = script_dir / 'models' / first_level_folder 
                for second_level_folder in current_sub_folder_path.iterdir(): # creater
                    second_level_path_arr = re.split(r"[\\/]+", str(second_level_folder))
                    current_model_creator = second_level_path_arr[-1]
                    current_sub_sub_folder_path = script_dir / 'models' / first_level_folder / second_level_folder
                    for third_level_folder in current_sub_sub_folder_path.iterdir(): # model
                        third_level_path_arr = re.split(r"[\\/]+", str(third_level_folder))
                        current_model_name = third_level_path_arr[-1]
                        current_model_id = f'{current_model_creator}/{current_model_name}'                        
                        model_dict = {}
                        model_dict["name"] = str(current_model_id)
                        model_dict["pipeline"] = str(current_pipeline_tag)
                        model_dict["path"] = str(third_level_folder)
                        res_model_arr += [model_dict]


        current_available_models_data = res_model_arr.copy()
        current_available_models_names = [m["name"] for m in res_model_arr]
        if res_model_arr:
            return gr.update(choices=current_available_models_names, value=current_available_models_names[0], show_label=True, label=f'Found {len(res_model_arr)} models!')
        else:
            return gr.update(choices=[], value=None, show_label=True, label="No models found!")

    except Exception as e:
        logging.exception(f'Exception occured: {e}', exc_info=True)
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
        return gr.update(choices=[], value=None, show_label=True, label=f"Error: {e}")


def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')


def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

def get_info(selected_id):    
    global current_models_data    
    res_model_data = {
        "search_data" : "",
        "model_id" : selected_id,
        "pipeline_tag" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }

    try:
        for item in current_models_data:
            if item['id'] == selected_id:
                
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
                
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"]

def available_get_info(selected_id):    
    global current_available_models_data    
    print(f'[available_get_info] selected_id {selected_id}')
    res_model_data = {
        "model_id" : selected_id,
        "pipeline_tag" : "",
        "model_path" : ""
    }

    try:
        for item in current_available_models_data:
            if item["name"] == selected_id:
                print(f'found model! {item}')
                res_model_data["pipeline_tag"] = item["pipeline"]
                res_model_data["model_path"] = item["path"]
                return res_model_data["pipeline_tag"], res_model_data["model_path"]
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["pipeline_tag"], res_model_data["model_path"]
    

def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "config_data" : "",
            "model_id" : selected_id,
            "size" : 0,
            "gated" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:                        
                    safetensors_json = vars(model_info.safetensors)
                    res_model_data['size'] = safetensors_json["total"]
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                url = f'https://huggingface.co/{selected_id}/resolve/main/config.json'
                response = requests.get(url)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json                     
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            if res_model_data["size"] == 0:
                try:
                    res_model_data["size"] = calculate_model_size(res_model_data["config_data"]) 
                except Exception as get_config_json_err:
                    res_model_data["size"] = 0
    
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]
    
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]

# def gr_load_check(selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated):
#     if selected_model_pipeline_tag != '' and selected_model_transformers == 'True' and selected_model_private == 'False' and selected_model_gated == 'False':
#         return gr.update(visible=False), gr.update(value=f'Download {selected_model_id[:12]}...', visible=True)
#     else:
#         return gr.update(visible=True), gr.update(visible=False)


def gr_load_check(selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated):
    if selected_model_pipeline_tag != '' and selected_model_transformers == 'True' and selected_model_private == 'False' and selected_model_gated == 'False':
        return gr.update(visible=True), gr.update(value=f'Download {selected_model_id[:12]}...', visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=True)



rx_change_arr = []
def check_rx_change(current_rx_bytes):
    try:                
        try:
            int(current_rx_bytes)
        except ValueError:
            return '0'
        global rx_change_arr
        rx_change_arr += [int(current_rx_bytes)]
        if len(rx_change_arr) > 4:
            last_value = rx_change_arr[-1]
            same_value_count = 0
            for i in range(1,len(rx_change_arr)):
                if rx_change_arr[i*-1] == last_value:
                    same_value_count += 1
                    if same_value_count > 10:
                        return f'Count > 10 Download finished'
                else:
                    return f'Count: {same_value_count} {str(rx_change_arr)}'
            return f'Count: {same_value_count} {str(rx_change_arr)}'        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'Error rx_change_arr {str(e)}'


def download_from_hf_hub(selected_model_id,selected_model_pipeline_tag):
    try:
        print(f'trying to download ...')
        model_path = snapshot_download(
            repo_id=selected_model_id,
            local_dir=f'{LOCAL_DIR}/{selected_model_pipeline_tag}/{selected_model_id}',
            token=HF_TOKEN,
            force_download=True  # Resume in case of interruption
        )
        # print(f'trying to find config ...')
        # config_path = os.path.join(model_path, "config.json")
        return f'download result: {model_path}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'

prev_bytes_recv = 0
def get_download_speed():
    try:
        global prev_bytes_recv
        print(f'trying to get download speed ...')
        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - prev_bytes_recv
        prev_bytes_recv = bytes_recv
        download_speed_kb = download_speed / 1024
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)      
        bytes_received_mb = bytes_recv / (1024 ** 2)
        return f'{download_speed_mbit_s:.2f} MBit/s (total: {bytes_received_mb:.2f})'
        # return f'{download_speed_kb:.2f} KB/s (total: {bytes_received_mb:.2f})'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'download error: {e}'

def update_visibility_model_info():
    return {
        selected_model_pipeline_tag: gr.update(visible=True),
        selected_model_transformers: gr.update(visible=True),
        selected_model_private: gr.update(visible=True),
        selected_model_downloads: gr.update(visible=True),
        selected_model_size: gr.update(visible=True),
        selected_model_gated: gr.update(visible=True)
    }

with gr.Blocks() as app:
    gr.Markdown(
        """
        # Welcome!
        Download Hugging Face model and deploy
        
        **Note**: _[https://router.huggingface.co/models](https://router.huggingface.co/models)_        
        """)
    

    
    inp = gr.Textbox(placeholder="Type in a Hugging Face model or tag", show_label=False, autofocus=True)
    btn = gr.Button("Search")
    out = gr.Textbox(visible=False)

    

    
    model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False, visible=False)

    with gr.Row():
        selected_model_id = gr.Textbox(label="id",visible=False)
        selected_model_container_name = gr.Textbox(label="container_name",visible=False)
        
    with gr.Row():       
        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag", visible=False)
        selected_model_transformers = gr.Textbox(label="transformers", visible=False)
        selected_model_private = gr.Textbox(label="private", visible=False)
        
    with gr.Row():
        selected_model_size = gr.Textbox(label="size", visible=False)
        selected_model_gated = gr.Textbox(label="gated", visible=False)
        selected_model_downloads = gr.Textbox(label="downloads", visible=False)
    
    selected_model_search_data = gr.Textbox(label="search_data", visible=False)
    selected_model_hf_data = gr.Textbox(label="hf_data", visible=False)
    selected_model_config_data = gr.Textbox(label="config_data", visible=False)
    gr.Markdown(
        """
        <hr>
        """
    )            

    

    inp.submit(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)
    btn.click(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)


    info_textbox = gr.Textbox(value="Interface not possible for selected model. Try another model or check 'pipeline_tag', 'transformers', 'private', 'gated'", show_label=False, visible=False)
    btn_dl = gr.Button("Download", visible=False)
    
    model_dropdown.change(get_info, model_dropdown, [selected_model_search_data,selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_container_name]).then(get_additional_info, model_dropdown, [selected_model_hf_data, selected_model_config_data, selected_model_id, selected_model_size, selected_model_gated]).then(update_visibility_model_info, None, [selected_model_pipeline_tag, selected_model_transformers,selected_model_private,selected_model_downloads,selected_model_size,selected_model_gated]).then(gr_load_check, [selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated],[info_textbox,btn_dl]) 




    create_response = gr.Textbox(label="Create response...", show_label=True, visible=False)  
    timer_dl_box = gr.Textbox(label="Dowmload progress:", visible=False)
    
    btn_interface = gr.Button("Load Download Interface",visible=False)
    @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
    def show_split(text_pipeline, text_model):
        if len(text_model) == 0:
            gr.Markdown("Error pipeline_tag or model_id")
        else:
            # select correct local path
            available_model_path = script_dir / 'models' / text_pipeline / text_model
            print(f'available_model_path: {available_model_path}')
            gr.Interface.from_pipeline(pipeline(text_pipeline, model=available_model_path), title=None)


    timer_dl = gr.Timer(1,active=False)
    timer_dl.tick(get_download_speed, outputs=timer_dl_box)

    btn_dl.click(lambda: gr.update(label="Starting download ...",visible=True), None, create_response).then(lambda: gr.update(visible=True), None, timer_dl_box).then(lambda: gr.Timer(active=True), None, timer_dl).then(download_from_hf_hub, [model_dropdown,selected_model_pipeline_tag], create_response).then(lambda: gr.Timer(active=False), None, timer_dl).then(lambda: gr.update(label="Download finished!"), None, create_response).then(lambda: gr.update(visible=True), None, btn_interface)
    
    
    available_model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False, visible=True)
    with gr.Row():        
        def get_model_folders():
            try:
                models_dir = script_dir / 'models'
                if not models_dir.exists() or not models_dir.is_dir():
                    print(f"No 'models' directory found in {script_dir}")
                    return {}

                # Create a dictionary to hold the first-level folders and their subfolders
                model_dict = {}

                # Iterate over first-level directories in the 'models' directory
                for first_level_folder in models_dir.iterdir():
                    if first_level_folder.is_dir():
                        # List all subdirectories for each first-level folder
                        subfolders = [subfolder.name for subfolder in first_level_folder.iterdir() if subfolder.is_dir()]
                        # Store the subfolders array in the dictionary
                        model_dict[first_level_folder.name] = subfolders

                return model_dict
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'download error: {e}'

        available_model_pipeline_tag_box = gr.Textbox(label="pipeline_tag_box", visible=True)
        available_model_path_box = gr.Textbox(label="path_box", visible=True)
        
        available_model_dropdown.change(available_get_info, available_model_dropdown, [available_model_pipeline_tag_box,available_model_path_box])

        
        
        
    available_btn_interface = gr.Button("Load Available Interface",visible=False)
    @gr.render(inputs=[available_model_pipeline_tag_box, available_model_path_box], triggers=[available_btn_interface.click])
    def available_load_interface(available_text_pipeline, available_model_path):
        if len(available_model_path) == 0:
            gr.Markdown("Error available pipeline_tag or model_id")
        else:
            print(f'available_model_path: {available_model_path}')
            gr.Interface.from_pipeline(pipeline(available_text_pipeline, model=available_model_path), title=None)



    
    log_box = gr.Textbox(label="Log", visible=True)
    log_timer = gr.Timer(1,active=False)
    log_timer.tick(load_log_file, outputs=log_box)
    
    #fin
    # app.load(get_model_folders, inputs=None, outputs=models_textbox)
    # app.load(available_search_models, outputs=[available_model_selected])
    # app.load(get_available_models, inputs=None, outputs=available_model_selected)
    app.load(lambda: gr.Timer(active=True), None, log_timer).then(get_available_models, inputs=None, outputs=available_model_dropdown).then(lambda: gr.update(visible=True), None, available_btn_interface)


# app.queue().launch(share=False,
#                         debug=False,
#                         server_name="0.0.0.0",
#                         server_port=5555,
#                         ssl_verify=False)
# app.queue().launch(share=False,
#                         debug=False,
#                         server_name="0.0.0.0",
#                         ssl_certfile="cert.pem",
#                         ssl_keyfile="key.pem",
#                         server_port=5555,
#                         ssl_verify=False)
# app.launch(server_name="0.0.0.0", server_port=5555, ssl_verify=False)
# app.launch(server_name="0.0.0.0", server_port=5555)
app.launch(server_name="0.0.0.0", server_port=int(os.getenv("CONTAINER_PORT")))
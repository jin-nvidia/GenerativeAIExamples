# Agentic Healthcare Front Desk

![](./images/architecture_diagram.png)

An agentic healthcare front desk can assist patients and the healthcare professionals in various scanarios: it can assist with the new patient intake process, going over each of the fields in a enw patient form with the patients; it can assist with the appointment scheduling process, looking up available appointments and booking them for patients after conversing with the patient to find out their needs; it can help look up the patient's medications and general information on the medications, and more.

The front desk assistant contains agentic LLM NIM with tools calling capabilities implemented in the LangGraph framework.

Follow along this repository to see how you can create your own Healthcare front desk that combines NVIDIA NIM, RIVA ASR and RIVA TTS.

We will offer two options for interacting with the agentic healthcare front desk: with a text-based Gradio UI or with a voice-based web interface powered by [NVIDIA ace-controller](https://github.com/NVIDIA/ace-controller).


> [!NOTE]  
> If you're utilizing the NVIDIA AI Endpoints for the LLM, which is the default for this repo, latency can vary depending on the traffic to the endpoints. 



## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Run Instructions](#run-instructions)
4. [Customization](#customization)


## Introduction
In this repository, we demonstrate the following:
* A customer care agent in Langgraph that has three specialist assistants: patient intake, medication assistance, and appointment making, with corresponding tools.
* A customer care agent in Langgraph for patient intake only. 
* A customer care agent in Langgraph for appointment making only.
* A customer care agent in Langgraph for medication lookup only.
* A Gradio based UI that allows us to use voice or typing in text to converse with any of the four agents.
* A chain server that serves the graph via FastAPI.

The agentic tool calling capability in each of the customer care assistants is powered by LLM NIMs - NVIDIA Inference Microservices. With the agentic capability, you can write your own tools to be utilized by LLMs.

## Prerequisites
### Hardware 
There are no local GPU requirements for running any application in this repo. The LLMs utilized in LangGraph in this repo are by default set to calling NVIDIA AI Endpoints since `BASE_URL` is set to the default value of `"https://integrate.api.nvidia.com/v1"` in [vars.env](./vars.env), and require a valid NVIDIA API KEY. As seen in the [graph definitions](./graph_definitions/):
```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA
assistant_llm = ChatNVIDIA(model=llm_model, ...)
```
You can experiment with other LLMs available on build.nvidia.com by changing the `LLM_MODEL` values in [vars.env](./vars.env), for passing into `ChatNVIDIA` in the Python files in the directory [`graph_definitions/`](./graph_definitions/).

If instead of calling NVIDIA AI Endpoints with an API key, you would like to host your own LLM NIM instance, please refer to the [Docker tab of the LLM NIM](https://build.nvidia.com/meta/llama-3_1-70b-instruct?snippet_tab=Docker) on how to host, and changed the `BASE_URL` parameter in [vars.env](./vars.env) to [point to your own instance](https://python.langchain.com/docs/integrations/chat/nvidia_ai_endpoints/#working-with-nvidia-nims) when specifying `ChatNVIDIA` in the Python files in the directory [`graph_definitions/`](./graph_definitions/). For the hardware configuration of self hosting the LLM, please refer to the [documentation for LLM support matrix](https://docs.nvidia.com/nim/large-language-models/latest/support-matrix.html).

### NVIDIA API KEY
You will need an NVIDIA API KEY to call NVIDIA AI Endpoints.  You can use different model API endpoints with the same API key, so even if you change the LLM specification in `ChatNVIDIA(model=llm_model)` you can still use the same API KEY.

a. Navigate to [https://build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover).

b. Find the **Llama 3.1 70B Instruct** card and click the card.

![Llama 3 70B Instruct model card](./images/llama31-70b-instruct-model-card.png)

c. Click **Get API Key**.

![API section of the model page.](./images/llama31-70b-instruct-get-api-key.png)
Log in if you haven't already.

d. Click **Generate Key**.

![Generate key window.](./images/api-catalog-generate-api-key.png)

e. Click **Copy Key** and then save the API key. The key begins with the letters ``nvapi-``.

![Key Generated window.](./images/key-generated.png)


### Software

- Linux operating systems (Ubuntu 22.04 or later recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)



## Run Instructions

As illustrated in the diagrams in the beginning, in this repo, we could run two types of applications, one is a FastAPI-based chain server, the other one is a simple text-based Gradio UI for the healthcare agent. In this documentation, we will be showing how to use the Gradio UI. For the steps to enable the voice-based interface powered by ace-controller, please see the [`healthcare_voice_agent`](https://github.com/NVIDIA/ace-controller/tree/develop/examples/healthcare_voice_agent) example in the [`ace-controller` repository](https://github.com/NVIDIA/ace-controller).

Regardless of the type of application you'd like to run, first, please add your API Keys.

### 1. Add Your API keys Prior to Running
In the file `vars.env`, add two API keys of your own:
```
NVIDIA_API_KEY="nvapi-" 
TAVILY_API_KEY="tvly-"
```
Note the Tavily key is only required if you want to run the full graph or the medication lookup graph. Get your API Key from the [Tavily website](https://app.tavily.com/). This is used in the tool named `medication_instruction_search_tool` in [`graph.py`](./graph_definitions/graph.py) or [`graph_medication_lookup_only.py`](./graph_definitions/graph_medication_lookup_only.py).

### 2. Running the simple text Gradio UI
To spin up a simple Gradio based web UI that allows us to converse with one of the agents via voice or typing, run one of these following services.

##### 2.1 The patient intake agent 
Run the patient intake only agent.

```sh
# to run the container with the assumption we have done build:
docker compose up -d patient-intake-ui
# or to build at this command:
docker compose up --build -d patient-intake-ui
```

Note this will be running on port 7860 by default. If you need to run on a different port, modify the [`docker-compose.yaml`](./docker-compose.yaml) file's `patient-intake-ui` section and replace all mentions of 7860 with your own port number.

[Launch the web UI](#25-launch-the-web-ui) on your Chrome browser, you should see this interface:
![](./images/example_ui.png)

To bring down the patient intake UI:
```sh
docker compose down patient-intake-ui
```


##### 2.2 The appointment making agent 
Run the appointment making only agent.
```sh
# to run the container with the assumption we have done build:
docker compose up -d appointment-making-ui
# or to build at this command:
docker compose up --build -d appointment-making-ui
```

Note this will be running on port 7860 by default. If you need to run on a different port, modify the [`docker-compose.yaml`](./docker-compose.yaml) file's `appointment-making-ui` section and replace all mentions of 7860 with your own port number.

[Launch the web UI](#25-launch-the-web-ui) on your Chrome browser, you should see the same web interface as above.

To bring down the appointment making UI:
```sh
docker compose down appointment-making-ui
```

##### 2.3 The full agent 
Run the full agent comprising of three specialist agents.
```sh
# to run the container with the assumption we have done build:
docker compose up -d full-agent-ui
# or to build at this command:
docker compose up --build -d full-agent-ui
```

Note this will be running on port 7860 by default. If you need to run on a different port, modify the [`docker-compose.yaml`](./docker-compose.yaml) file's `full-agent-ui` section and replace all mentions of 7860 with your own port number.

[Launch the web UI](#25-launch-the-web-ui) on your Chrome browser, you should see the same web interface as above.

To bring down the full agent UI:
```sh
docker compose down full-agent-ui
```

##### 2.4 The medication lookup agent 
Run the medication lookup only agent.

```sh
# to run the container with the assumption we have done build:
docker compose up -d medication-lookup-ui
# or to build at this command:
docker compose up --build -d medication-lookup-ui
```

Note this will be running on port 7860 by default. If you need to run on a different port, modify the [`docker-compose.yaml`](./docker-compose.yaml) file's `medication-lookup-ui` section and replace all mentions of 7860 with your own port number.

[Launch the web UI](#25-launch-the-web-ui) on your Chrome browser, you should see the same web interface as above.

To bring down the medication lookup UI:
```sh
docker compose down medication-lookup-ui
```

##### 2.5 Launch the web UI

Go to your web browser, here we have tested with Google Chrome, and type in `<your machine's ip address>:<port number>`. The port number would be `7860` by default, or your modified port number if you changed the port number in [docker-compose.yaml](./docker-compose.yaml). 

## Customization
To customize for your own agentic LLM in LangGraph with your own tools, the [LangGraph tutorial on customer support](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/) is helpful, where you'll find detailed explanations and steps of creating tools and agentic LLM in LangGraph. Afterwards, you can create your own file similar to the graph files in [`graph_definitions/`](./graph_definitions/) which can connect to the simple text Gradio UI by calling [`launch_demo_ui`](./graph_definitions/graph_patient_intake_only.py#L184), or can be imported by the [chain server](./chain_server/chain_server.py#L31).


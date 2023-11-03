from langchain.llms import BaseLLM

from eurelis_kb_framework.base_factory import ParamsDictFactory


class HuggingFaceFactory(ParamsDictFactory[BaseLLM]):
    OPTIONAL_PARAMS = {
        "batch_size",
        "cache",
        "metadata",
        "model_id",
        "model_kwargs",
        "pipeline_kwargs",
        "tags",
        "verbose",
    }

    def __init__(self):
        super().__init__()
        self.task = "text-generation"
        self.model_id = None

    def set_model_id(self, model_id: str):
        self.model_id = model_id

    def set_task(self, task: str):
        self.task = task

    def build(self, context) -> BaseLLM:
        from langchain.llms import HuggingFacePipeline

        arguments = self.get_optional_params()

        return HuggingFacePipeline.from_model_id(self.model_id, self.task, **arguments)

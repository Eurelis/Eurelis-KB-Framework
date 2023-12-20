from typing import Dict, Any, Optional
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler


class MarkdownHtmlCallback(BaseCallbackHandler):
    def __init__(self, input_field: str = "answer", output_field: str = "answer_html"):
        super().__init__()
        self._input_field = input_field
        self._output_field = output_field

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        import markdown

        if parent_run_id is None and self._input_field in outputs:  # last chain_end
            input = outputs.get(self._input_field)

            output = markdown.markdown(input)

            outputs[self._output_field] = output

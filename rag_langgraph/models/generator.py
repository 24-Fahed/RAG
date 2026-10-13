"""Generator model used only for HyDE document generation."""

import logging

logger = logging.getLogger(__name__)


class Generator:
    """LLM-backed HyDE generator."""

    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None

        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            logger.info(
                "Loaded HyDE generator model=%s tokenizer_class=%s model_class=%s chat_template=%s",
                model_path,
                type(self.tokenizer).__name__ if self.tokenizer is not None else None,
                type(self.model).__name__ if self.model is not None else None,
                bool(getattr(self.tokenizer, "chat_template", None)),
            )
        except Exception as exc:
            raise RuntimeError(f"Could not load generator {model_path}") from exc

    def _preview(self, text: str, limit: int = 200) -> str:
        return text.replace("\n", "\\n")[:limit]

    def _build_hyde_messages(self, query: str) -> list[dict[str, str]]:
        user_content = (
            "Write one short neutral factual passage that could appear in a scientific or encyclopedic document "
            "related to the query. Do not answer with yes or no. Do not address the user. Do not judge whether "
            "the claim is correct. Focus on entities, concepts, and relationships useful for retrieval.\n\n"
            f"Query: {query}"
        )
        return [
            {
                "role": "system",
                "content": "You write concise neutral passages for retrieval augmentation.",
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_hyde_prompt(self, query: str) -> str:
        return (
            "Write one short neutral factual passage related to the query for retrieval. "
            "Do not answer yes or no. Do not judge the claim.\n\n"
            f"Query: {query}\n\nPassage:"
        )

    def _build_inputs_from_messages(self, messages: list[dict[str, str]]):
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_count = len(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
        raw_inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        )
        if isinstance(raw_inputs, torch.Tensor):
            inputs = {
                "input_ids": raw_inputs,
                "attention_mask": torch.ones_like(raw_inputs),
            }
        else:
            inputs = raw_inputs
        return prompt, prompt_token_count, inputs

    def _build_inputs(self, query: str):
        messages = self._build_hyde_messages(query)
        plain_prompt = self._build_hyde_prompt(query)
        use_chat_template = bool(
            self.tokenizer is not None
            and hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None)
        )

        if use_chat_template:
            prompt, prompt_token_count, inputs = self._build_inputs_from_messages(messages)
        else:
            prompt = plain_prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_token_count = len(self.tokenizer(prompt, add_special_tokens=True)["input_ids"])

        input_token_count = int(inputs["input_ids"].shape[1])
        truncated = prompt_token_count > input_token_count
        return inputs, prompt, input_token_count, truncated, use_chat_template

    def _move_inputs_to_device(self, inputs):
        if hasattr(self.model, "device"):
            if hasattr(inputs, "to"):
                return inputs.to(self.model.device)
            return {key: value.to(self.model.device) for key, value in inputs.items()}
        return inputs

    def _normalize_generate_output(self, outputs):
        if hasattr(outputs, "sequences") and outputs.sequences is not None:
            return outputs.sequences
        if hasattr(outputs, "shape"):
            return outputs
        raise TypeError(f"Unsupported generate output type: {type(outputs)!r}")

    def generate_hyde(self, query: str, max_out_len: int = 100) -> str:
        if self.model is None:
            raise ValueError("A local generator model_path is required")

        inputs, prompt, input_token_count, truncated, used_chat_template = self._build_inputs(query)

        logger.info(
            "Generator request mode=hyde model=%s query_chars=%s prompt_tokens=%s truncated=%s chat_template=%s prompt_preview=%s",
            self.model_path,
            len(query),
            input_token_count,
            truncated,
            used_chat_template,
            self._preview(prompt),
        )

        inputs = self._move_inputs_to_device(inputs)

        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_out_len,
                return_dict_in_generate=False,
                use_cache=False,
            )

        sequences = self._normalize_generate_output(outputs)
        generated_ids = sequences[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        logger.info(
            "Generator result mode=hyde model=%s generated_tokens=%s output_chars=%s output_preview=%s",
            self.model_path,
            int(generated_ids.shape[0]),
            len(response),
            self._preview(response),
        )

        return response


def get_generator(model_path: str = "", *, provider: str = "local"):
    """Generation length belongs to generate_hyde(), never to the shared factory."""
    from rag_langgraph.models.runtime import registry
    if provider == "siliconflow":
        from rag_langgraph.models.remote_generator import SiliconFlowGenerator
        return registry.get(("generator", provider, model_path), lambda: SiliconFlowGenerator(model_path))
    if provider != "local":
        raise ValueError(f"Unknown generation provider: {provider}")
    return registry.get(("generator", provider, model_path), lambda: Generator(model_path))

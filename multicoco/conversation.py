"""
Conversation management for multimodal dialogue systems.

This module provides conversation templates and management utilities for
handling dialogue with multimodal AI models, supporting various conversation
formats and separator styles.
"""

import dataclasses
import logging
from enum import auto, Enum
from typing import List, Any, Optional, Dict, Union

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class SeparatorStyle(Enum):
    """Different separator styles for conversation formatting."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    INTERNVL = auto()


@dataclasses.dataclass
class Conversation:
    """
    A class that manages conversation history and formatting.

    This class maintains conversation state and provides methods for formatting
    conversations according to different styles and templates.

    Attributes:
        system: System prompt/message
        roles: List of role names (e.g., ["user", "assistant"])
        messages: List of [role, message] pairs
        offset: Starting offset for conversation display
        sep_style: Separator style for formatting
        sep: Primary separator string
        sep2: Secondary separator string (optional)
        version: Template version identifier
        skip_next: Flag to skip next message processing
    """

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: Optional[str] = None
    version: str = "Unknown"
    skip_next: bool = False

    def get_prompt(self) -> str:
        """
        Generate formatted prompt string based on separator style.

        Returns:
            Formatted conversation prompt

        Raises:
            ConfigurationError: If separator style is invalid
        """
        if self.sep_style == SeparatorStyle.SINGLE:
            return self._format_single_separator()
        elif self.sep_style == SeparatorStyle.TWO:
            return self._format_two_separators()
        elif self.sep_style == SeparatorStyle.MPT:
            return self._format_mpt_style()
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            return self._format_llama2_style()
        elif self.sep_style == SeparatorStyle.PLAIN:
            return self._format_plain_style()
        elif self.sep_style == SeparatorStyle.INTERNVL:
            return self._format_internvl_style()
        else:
            raise ConfigurationError(f"Invalid separator style: {self.sep_style}")

    def _format_single_separator(self) -> str:
        """Format conversation with single separator style."""
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += f"{role}: {message}{self.sep}"
            else:
                ret += f"{role}:"
        return ret

    def _format_two_separators(self) -> str:
        """Format conversation with alternating two separators."""
        seps = [self.sep, self.sep2]
        ret = self.system + seps[0]

        for i, (role, message) in enumerate(self.messages):
            current_sep = seps[i % 2]
            if message:
                if current_sep is not None:
                    ret += f"{role}: {message}{current_sep}"
                else:
                    ret += f"{role}: {message}"
            else:
                ret += f"{role}:"
        return ret

    def _format_mpt_style(self) -> str:
        """Format conversation in MPT style."""
        ret = self.system + self.sep
        for role, message in self.messages:
            if message:
                ret += f"{role}{message}{self.sep}"
            else:
                ret += role
        return ret

    def _format_llama2_style(self) -> str:
        """Format conversation in LLaMA-2 style with special tokens."""
        def wrap_sys(msg: str) -> str:
            return f"<<SYS>>\n{msg}\n<</SYS>>\n\n"

        def wrap_inst(msg: str) -> str:
            return f"[INST] {msg} [/INST]"

        ret = ""
        for i, (role, message) in enumerate(self.messages):
            if i == 0:
                if not message:
                    raise ConfigurationError("First message should be a user message in LLaMA-2 style")
                message = wrap_sys(self.system) + message if self.system else message

            if message:
                if isinstance(message, list):
                    message = ' '.join(message)

                if i % 2 == 0:  # User message
                    message = wrap_inst(message)
                    ret += self.sep + message
                else:  # Assistant message
                    ret += f" {message} {self.sep2}"
            else:
                ret += " "
        return ret

    def _format_plain_style(self) -> str:
        """Format conversation in plain style."""
        seps = [self.sep, self.sep2]
        ret = self.system

        for i, (role, message) in enumerate(self.messages):
            current_sep = seps[i % 2]
            if message:
                ret += message + current_sep
        return ret

    def _format_internvl_style(self) -> str:
        """Format conversation in InternVL style."""
        seps = [self.sep, self.sep2]
        ret = self.system

        for i, (role, message) in enumerate(self.messages):
            if message:
                if isinstance(message, list):
                    message = ' '.join(message)

                # Ensure separator is not None before concatenation
                current_sep = seps[i % 2]
                if current_sep is not None:
                    ret += role + message + current_sep
                else:
                    ret += role + message
            else:
                ret += role
        return ret

    def append_message(self, role: str, message: str) -> None:
        """
        Add a new message to the conversation.

        Args:
            role: Role of the message sender
            message: Message content
        """
        self.messages.append([role, message])
        logger.debug(f"Added message from {role}: {message[:50]}...")

    def to_gradio_chatbot(self) -> List[List[Optional[str]]]:
        """
        Convert conversation to Gradio chatbot format.

        Returns:
            List of [user_message, bot_message] pairs for Gradio
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                if ret:  # Ensure we have a previous entry to update
                    ret[-1][-1] = msg
        return ret

    def copy(self) -> 'Conversation':
        """
        Create a deep copy of the conversation.

        Returns:
            New Conversation instance with copied data
        """
        return Conversation(
            system=self.system,
            roles=self.roles.copy(),
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary format.

        Returns:
            Dictionary representation of the conversation
        """
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

    def __len__(self) -> int:
        """Return the number of messages in the conversation."""
        return len(self.messages)

    def __str__(self) -> str:
        """Return string representation of the conversation."""
        return f"Conversation(version={self.version}, messages={len(self.messages)})"


# Predefined conversation template for InternVL
conv_internvl_v1_1 = Conversation(
    system="A helper assistant that can help me with a variety of tasks. ",
    roles=["<human>", "<bot>"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.INTERNVL,
    sep=" ",
    sep2="</s>",
    version="internvl_v1.1"
)


def get_conv_template(name: str) -> Conversation:
    """
    Get a conversation template by name.

    Args:
        name: Name of the conversation template

    Returns:
        Copy of the requested conversation template

    Raises:
        ConfigurationError: If template name is not found
    """
    templates = {
        "internvl_v1.1": conv_internvl_v1_1,
        "internvl2.5": conv_internvl_v1_1,  # Use same template for v2.5
    }

    if name not in templates:
        available = list(templates.keys())
        raise ConfigurationError(f"Unknown conversation template '{name}'. Available: {available}")

    template = templates[name]
    logger.debug(f"Retrieved conversation template: {name}")
    return template.copy()


# Utility function for quick template testing
def create_sample_conversation() -> Conversation:
    """
    Create a sample conversation for testing purposes.

    Returns:
        Sample conversation with example messages
    """
    conv = get_conv_template("internvl_v1.1")
    conv.append_message("<human>", "Hello, how are you?")
    conv.append_message("<bot>", "I'm doing well, thank you for asking!")
    return conv


if __name__ == "__main__":
    # Demo conversation template
    template = get_conv_template("internvl_v1.1")
    print("Template prompt:", template.get_prompt())

    # Demo with sample conversation
    sample = create_sample_conversation()
    print("Sample conversation:", sample.get_prompt()) 